"""
Module description:

"""

from ast import literal_eval as make_tuple

from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler_full as csf
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .GRCNModel import GRCNModel

from torch_sparse import SparseTensor


class GRCN(RecMixin, BaseRecommenderModel):
    r"""
    Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3394171.3413556>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        num_layers: Number of propagation layers
        num_routings: Number of routing iterations
        factors: Number of latent factors
        factors_multimod: Tuple with number of units for each modality
        batch_size: Batch size
        l_w: Regularization coefficient
        modalities: Tuple of modalities
        aggregation: Type of aggregation
        weight_mode: Type of weight
        pruning: Whether to pruning or not
        has_act: Whether to use activation or not
        fusion_mode: Type of multimodal fusion

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        GRCN:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          num_layers: 3
          num_routings: 10
          factors: 64
          factors_multimod: (64,64)
          batch_size: 256
          l_w: 0.1
          modalities: (visual,textual)
          aggregation: concat
          weight_mode: max
          pruning: True
          has_act: False
          fusion_mode: concat
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_num_layers", "num_layers", "num_layers", 3, int, None),
            ("_num_routings", "num_routings", "num_routings", 10, int, None),
            ("_factors_multimod", "factors_multimod", "factors_multimod", 64, int, None),
            ("_modalities", "modalities", "modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_aggregation", "aggregation", "aggr", 'concat', str, None),
            ("_weight_mode", "weight_mode", "w_mod", 'max', str, None),
            ("_pruning", "pruning", "prun", True, bool, None),
            ("_has_act", "has_act", "act", False, bool, None),
            ("_fusion_mode", "fusion_mode", "f_mod", 'concat', str, None),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        self.set_modalities(self._modalities)

        np.random.seed(self._seed)

        self._sampler = csf.Sampler(self._data.i_train_dict, self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        _, counts = np.unique(row, return_counts=True)
        ptr = [0]
        c = counts.tolist()
        for i in range(len(c)):
            ptr += [ptr[i] + c[i]]
        ptr = torch.tensor(np.array(ptr), dtype=torch.int64)
        _, counts_full = np.unique(np.concatenate((row, np.array(col))), return_counts=True)
        ptr_full = [0]
        c_full = counts_full.tolist()
        for i in range(len(c_full)):
            ptr_full += [ptr_full[i] + c_full[i]]
        ptr_full = torch.tensor(np.array(ptr_full), dtype=torch.int64)
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))
        self.adj_user = SparseTensor(row=edge_index[0],
                                     col=edge_index[1], sparse_sizes=(self._num_users + self._num_items,
                                                                      self._num_users + self._num_items))

        self._model = GRCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            embed_k_multimod=self._factors_multimod,
            l_w=self._l_w,
            num_layers=self._num_layers,
            num_routings=self._num_routings,
            modalities=self._modalities,
            aggregation=self._aggregation,
            weight_mode=self._weight_mode,
            pruning=self._pruning,
            has_act=self._has_act,
            fusion_mode=self._fusion_mode,
            multimodal_features=self.get_all_features(),
            adj=self.adj,
            adj_user=self.adj_user,
            rows=row,
            cols=col,
            ptr=ptr,
            ptr_full=ptr_full,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "GRCN" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        edge_index = np.array([row, col]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            self._model.train()
            np.random.shuffle(edge_index)
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index, self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
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
