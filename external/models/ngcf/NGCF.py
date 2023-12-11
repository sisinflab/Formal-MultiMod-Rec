"""
Module description:

"""

from tqdm import tqdm
import numpy as np
import torch
import random
import os

from elliot.utils.write import store_recommendation
from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .NGCFModel import NGCFModel

import math

from torch_sparse import SparseTensor

from torch_sparse import mul, fill_diag, sum


def apply_norm(edge_index, add_self_loops=True):
    adj_t = edge_index
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sum(adj_t, dim=1)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    norm_adj_t = mul(adj_t, deg_inv.view(-1, 1))
    return norm_adj_t


class NGCF(RecMixin, BaseRecommenderModel):
    r"""
    Neural Graph Collaborative Filtering

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3331184.3331267>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        weight_size: Tuple with number of units for each embedding propagation layer
        node_dropout: Tuple with dropout rate for each node
        message_dropout: Tuple with dropout rate for each embedding propagation layer

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NGCF:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          l_w: 0.1
          weight_size: (64,)
          node_dropout: ()
          message_dropout: (0.1,)
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 3, int, None),
            ("_weight_size", "weight_size", "weight_size", 64, int, None),
            ("_node_dropout", "node_dropout", "node_dropout", 0.0, float, None),
            ("_message_dropout", "message_dropout", "message_dropout", 0.5, float, None),
            ("_normalize", "normalize", "normalize", True, bool, None)
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        self._sampler = Sampler(self._data.i_train_dict, self._batch_size, self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        self.edge_index = np.array([row, col])

        self.adj = SparseTensor(row=torch.cat([torch.tensor(self.edge_index[0], dtype=torch.int64),
                                               torch.tensor(self.edge_index[1], dtype=torch.int64)], dim=0),
                                col=torch.cat([torch.tensor(self.edge_index[1], dtype=torch.int64),
                                               torch.tensor(self.edge_index[0], dtype=torch.int64)], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        if self._normalize:
            self.adj = apply_norm(self.adj, add_self_loops=True)

        self.users = list(range(self._num_users))
        self.items = list(range(self._num_items))

        self._model = NGCFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            weight_size=self._weight_size,
            n_layers=self._n_layers,
            message_dropout=self._message_dropout,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "NGCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device())
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = self.adj.to_torch_sparse_coo_tensor().coalesce().indices()
        v = self.adj.to_torch_sparse_coo_tensor().coalesce().values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = SparseTensor(row=i[0],
                           col=i[1],
                           value=v * (1. / (1 - rate)),
                           sparse_sizes=(self._num_users + self._num_items,
                                         self._num_users + self._num_items))
        return out

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            self._model.train()
            if self._node_dropout > 0:
                sampled_adj = self.sparse_dropout(self.adj,
                                                  self._node_dropout,
                                                  self.adj.nnz())
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for _ in range(n_batch):
                    user, pos, neg = self._sampler.step()
                    steps += 1
                    if self._node_dropout > 0:
                        loss += self._model.train_step((user, pos, neg), sampled_adj)
                    else:
                        loss += self._model.train_step((user, pos, neg), self.adj)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            gu, gi = self._model.propagate_embeddings(self.adj)
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(gu[offset: offset_stop], gi)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
