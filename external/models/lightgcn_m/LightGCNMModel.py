from abc import ABC

from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random


class LightGCNMModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 adj,
                 modalities,
                 multimodal_features,
                 aggregation,
                 normalize,
                 random_seed,
                 name="LightGCNM",
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
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.adj = adj
        self.normalize = normalize

        self.Gu = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)

        self.modalities = modalities
        self.aggregation = aggregation

        self.F = torch.nn.ParameterList()
        if self.aggregation == 'concat':
            total_multimodal_features = 0
            for m_id, m in enumerate(self.modalities):
                self.F.append(torch.nn.Embedding.from_pretrained(torch.tensor(
                    multimodal_features[m_id], device=self.device, dtype=torch.float32),
                    freeze=False))
                total_multimodal_features += multimodal_features[m_id].shape[-1]
            self.proj = torch.nn.Linear(in_features=total_multimodal_features, out_features=self.embed_k)
        else:
            self.proj = torch.nn.ModuleList()
            for m_id, m in enumerate(self.modalities):
                self.F.append(torch.nn.Embedding.from_pretrained(torch.tensor(
                    multimodal_features[m_id], device=self.device, dtype=torch.float32),
                    freeze=False))
                self.proj.append(
                    torch.nn.Linear(in_features=multimodal_features[m_id].shape[-1], out_features=self.embed_k))
        self.F.to(self.device)
        self.proj.to(self.device)

        propagation_network_list = []

        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=self.normalize), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        if self.aggregation == 'concat':
            F_proj = torch.nn.functional.normalize(
                self.proj(torch.concat([self.F[m_id].weight
                                        for m_id in range(len(self.F))], dim=-1).to(self.device)
                          ), p=2, dim=1).to(self.device)
        elif self.aggregation == 'mean':
            F_proj = [torch.nn.functional.normalize(self.proj[m_id](self.F[m_id].weight).to(self.device), p=2, dim=1)
                      for m_id in range(len(self.F))]
            F_proj = torch.mean(torch.stack(F_proj, dim=-1).to(self.device), dim=-1)
        elif self.aggregation == 'sum':
            F_proj = [torch.nn.functional.normalize(self.proj[m_id](self.F[m_id].weight).to(self.device), p=2, dim=1)
                      for m_id in range(len(self.F))]
            F_proj = torch.sum(torch.stack(F_proj, dim=-1).to(self.device), dim=-1)
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), F_proj.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = torch.mean(torch.stack(all_embeddings, 0), dim=0)
        # all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, fi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, fi

    def forward(self, inputs, **kwargs):
        gu, fi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        effe_i = torch.squeeze(fi).to(self.device)

        xui = torch.sum(gamma_u * effe_i, 1)

        return xui

    def predict(self, gu, fi, **kwargs):
        return torch.sigmoid(torch.matmul(gu.to(self.device), torch.transpose(fi.to(self.device), 0, 1)))

    def train_step(self, batch):
        gu, fi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user[:, 0]], fi[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], fi[neg[:, 0]]))
        loss = torch.mean(torch.nn.functional.softplus(xu_neg - xu_pos))
        reg_loss = self.l_w * (1 / 2) * (self.Gu.weight[user[:, 0]].norm(2).pow(2) +
                                         fi[pos[:, 0]].norm(2).pow(2) +
                                         fi[neg[:, 0]].norm(2).pow(2)) / float(batch[0].shape[0])
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
