from abc import ABC

import torch
import numpy as np
import random


class FREEDOMModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_layers,
                 num_ui_layers,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 rows,
                 cols,
                 modalities,
                 top_k,
                 multimodal_features,
                 adj,
                 mm_weight,
                 dropout,
                 lr_sched,
                 random_seed,
                 name="FREEDOM",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.modalities = modalities
        self.top_k = top_k
        self.n_layers = num_layers
        self.n_ui_layers = num_ui_layers
        self.adj = adj
        self.mm_weight = mm_weight
        self.masked_adj = None
        self.dropout = dropout
        self.lr_sched = lr_sched

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gu.to(self.device)
        self.Gi.to(self.device)

        rows = torch.from_numpy(rows)
        cols = torch.from_numpy(cols)
        self.edges_indices = torch.stack([rows, cols]).type(torch.LongTensor).to(self.device)
        self.edges_values = self._normalize_adj_m(self.edges_indices, torch.Size((self.num_users,
                                                                                  self.num_items))).to(self.device)

        # multimodal features
        self.Gim = torch.nn.ParameterDict()
        Sim = dict()
        self.projection_m = torch.nn.ModuleDict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            self.Gim[m] = torch.nn.Embedding.from_pretrained(
                torch.tensor(multimodal_features[m_id], dtype=torch.float32, device=self.device),
                freeze=False).weight
            self.Gim[m].to(self.device)
            self.projection_m[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                   out_features=self.embed_k_multimod)
            self.projection_m[m].to(self.device)
            current_sim = self.build_sim(self.Gim[m].detach())
            indices, Sim[m] = self.build_knn_neighbourhood(current_sim)

        self.mm_adj = self.mm_weight[0] * Sim[self.modalities[0]]
        for m_id, m in enumerate(self.modalities[1:]):
            self.mm_adj += self.mm_weight[m_id + 1] * Sim[m]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    @staticmethod
    def build_sim(context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_knn_neighbourhood(self, sim):
        _, knn_ind = torch.topk(sim, self.top_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.top_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size).to(self.device)

    @staticmethod
    def compute_normalized_laplacian(indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    @staticmethod
    def _normalize_adj_m(indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edges_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edges_values, degree_len)
        # random sample
        keep_indices = self.edges_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_users, self.num_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.num_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, (self.num_users + self.num_items,
                                                                             self.num_users + self.num_items)).to(self.device)

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.lr_sched[0] ** (epoch / self.lr_sched[1]))
        return scheduler

    def propagate_embeddings(self, adj):
        h = self.Gi.weight
        for layer in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.Gu.weight, self.Gi.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj.to(self.device), ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def forward(self, inputs, **kwargs):
        gum, gim = inputs
        gamma_u_m = torch.squeeze(gum).to(self.device)
        gamma_i_m = torch.squeeze(gim).to(self.device)

        xui = torch.sum(gamma_u_m * gamma_i_m, 1)

        return xui, gamma_u_m, gamma_i_m

    def predict(self, gum, gim, **kwargs):
        return torch.matmul(gum.to(self.device), torch.transpose(gim.to(self.device), 0, 1))

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = torch.nn.functional.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def train_step(self, batch):
        users = batch[0]
        pos_items = batch[1]
        neg_items = batch[2]

        ua_embeddings, ia_embeddings = self.propagate_embeddings(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                      neg_i_g_embeddings)
        mf_multimodal_loss = 0.0
        for m_id, m in enumerate(self.modalities):
            proj_features = self.projection_m[m](self.Gim[m].to(self.device))
            mf_current_loss = self.bpr_loss(ua_embeddings[users], proj_features[pos_items], proj_features[neg_items])
            mf_multimodal_loss += mf_current_loss
        self.optimizer.zero_grad()
        loss = batch_mf_loss + self.l_w * mf_multimodal_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
