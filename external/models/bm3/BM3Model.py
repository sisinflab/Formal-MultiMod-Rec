from abc import ABC

import torch
import numpy as np
import random
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import LGConv
import torch_geometric


class EmbLoss(torch.nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class BM3Model(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 reg_weight,
                 cl_weight,
                 dropout,
                 n_layers,
                 adj,
                 modalities,
                 multimodal_features,
                 multimod_embed_k,
                 learning_rate_scheduler,
                 random_seed,
                 name="BM3",
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
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.multimod_embed_k = multimod_embed_k
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.cl_weight = cl_weight
        self.dropout = dropout
        self.n_layers = n_layers
        self.adj = adj
        self.modalities = modalities

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gu.to(self.device)
        self.Gi.to(self.device)

        self.predictor = torch.nn.Linear(self.embed_k, self.embed_k).to(self.device)
        self.reg_loss = EmbLoss()
        torch.nn.init.xavier_normal_(self.predictor.weight)

        self.F = torch.nn.ParameterList()
        self.proj = torch.nn.ModuleList()
        for m_id, m in enumerate(self.modalities):
            self.F.append(torch.nn.Embedding.from_pretrained(torch.tensor(
                multimodal_features[m_id], device=self.device, dtype=torch.float32),
                freeze=False))
            linear = torch.nn.Linear(in_features=multimodal_features[m_id].shape[-1],
                                     out_features=self.multimod_embed_k)
            torch.nn.init.xavier_normal_(linear.weight)
            self.proj.append(linear)
        self.F.to(self.device)
        self.proj.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        propagation_network_list = []

        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=False), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        lr_scheduler = learning_rate_scheduler
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

    def propagate_embeddings(self):
        h = self.Gi.weight
        ego_embeddings = torch.cat((self.Gu.weight, self.Gi.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for layer in range(0, self.n_layers):
            all_embeddings += [list(
                self.propagation_network.children()
            )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def forward(self, inputs, **kwargs):
        gu, fi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        effe_i = torch.squeeze(fi).to(self.device)

        xui = torch.sum(gamma_u * effe_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        u_online, i_online = self.predictor(gu), self.predictor(gi)
        score_mat_ui = torch.matmul(u_online, i_online.transpose(0, 1))
        return score_mat_ui

    def train_step(self, batch):
        u_online_ori, i_online_ori = self.propagate_embeddings()
        proj_features_online = []
        proj_features_target = []

        for m_id, m in enumerate(self.modalities):
            proj_features_online.append(self.proj[m_id](self.F[m_id].weight.to(self.device)))

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = torch.nn.functional.dropout(u_target, self.dropout)
            i_target = torch.nn.functional.dropout(i_target, self.dropout)

            for m_id, m in enumerate(self.modalities):
                proj_features_target.append(proj_features_online[m_id].clone())
                proj_features_target[m_id] = torch.nn.functional.dropout(proj_features_target[m_id], self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = batch[0], batch[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_singlemod, loss_multimod = [], []
        for m_id, m in enumerate(self.modalities):
            current_online = self.predictor(proj_features_online[m_id])
            current_online = current_online[items, :]
            current_target = proj_features_target[m_id][items, :]
            loss_singlemod += [1 - cosine_similarity(current_online, i_target.detach(), dim=-1).mean()]
            loss_multimod += [1 - cosine_similarity(current_online, current_target.detach(), dim=-1).mean()]

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
            self.cl_weight * (sum(loss_singlemod) + sum(loss_multimod)).mean()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
