"""
This is the implementation of the normalized Discounted Cumulative Gain metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import typing as t

import pandas as pd
import numpy as np

from elliot.evaluation.metrics.base_metric import BaseMetric
from elliot.evaluation.metrics.metrics_utils import ProxyMetric


class clustered_nDCG(BaseMetric):
    r"""
    normalized Discounted Cumulative Gain

    This class represents the implementation of the nDCG recommendation metric.

    For further details, please refer to the `link <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`_

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}


    :math:`K` stands for recommending :math:`K` items.

    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.

    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.

    :math:`U^{te}` is for all users in the test set.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [nDCG]
    """

    def __init__(self, recommendations, config, params, eval_objects, additional_data):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects, additional_data)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.discounted_relevance
        self._rel_threshold = self._evaluation_objects.relevance._rel_threshold

        # self._item_clustering_path = self._additional_data.get("item_clustering_file", False)
        # if self._item_clustering_path:
        #     self._item_clustering = pd.read_csv(self._item_clustering_path, sep="\t", header=None)
        #     self._item_n_clusters = self._item_clustering[1].nunique()
        #     self._item_clustering = dict(zip(self._item_clustering[0], self._item_clustering[1]))
        #     self._item_clustering_name = self._additional_data['item_clustering_name']
        # else:
        #     self._item_n_clusters = 1
        #     self._item_clustering = {}
        #     self._item_clustering_name = ""

        self._user_clustering_path = self._additional_data.get("user_clustering_file", False)
        if self._user_clustering_path:
            self._user_clustering = pd.read_csv(self._user_clustering_path, sep="\t", header=None, names=['user', 'group'])
            self._user_clustering = {k: v["user"].tolist() for k, v in self._user_clustering.groupby(by=["group"])}
            self._user_n_clusters = len(self._user_clustering)
            # self._user_clustering = dict(zip(self._user_clustering[0], self._user_clustering[1]))
            self._user_clustering_name = self._additional_data['user_clustering_name']
        else:
            self._user_n_clusters = 1
            self._user_clustering = {}
            self._user_clustering_name = ""

        self._values_dict = {}

        self.process()

    def name(self):
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return f"nDCG_users:{self._user_clustering_name}"

    def compute_idcg(self, user, cutoff: int) -> float:
        """
        Method to compute Ideal Discounted Cumulative Gain
        :param gain_map:
        :param cutoff:
        :return:
        """
        gains: t.List = sorted(list(self._relevance.get_user_rel_gains(user).values()))
        n: int = min(len(gains), cutoff)
        m: int = len(gains)
        return sum(map(lambda g, r: gains[m - r - 1] * self._relevance.logarithmic_ranking_discount(r), gains, range(n)))

    def compute_user_ndcg(self, user_recommendations: t.List, user, cutoff: int) -> float:
        """
        Method to compute normalized Discounted Cumulative Gain
        :param sorted_item_predictions:
        :param gain_map:
        :param cutoff:
        :return:
        """
        idcg: float = self.compute_idcg(user, cutoff)
        dcg: float = sum(
            [self._relevance.get_rel(user, x) * self._relevance.logarithmic_ranking_discount(r)
             for r, x in enumerate([item for item, _ in user_recommendations]) if r < cutoff])
        return dcg / idcg if dcg > 0 else 0

    def __user_ndcg(self, user_recommendations: t.List, user, cutoff: int):
        """
        Per User normalized Discounted Cumulative Gain
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param user_gain_map: dict of discounted relevant items in the form {user1:{item1:value1,...},...}
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the value of the nDCG metric for the specific user
        """

        ndcg: float = self.compute_user_ndcg(user_recommendations[:cutoff], user, cutoff)

        return ndcg


    def process(self):
        """
        Evaluation function
        :return: the overall value of Bias Disparity - Bias Source
        """
        for user_group, users in self._user_clustering.items():
            self._values_dict[user_group] = np.average([self.__user_ndcg(u_r, u, self._cutoff) for u, u_r in self._recommendations.items() if len(self._relevance.get_user_rel(u)) and u in users])


        self._metric_objs_list = []
        for u_group in range(self._user_n_clusters):
            self._metric_objs_list.append(ProxyMetric(name= f"nDCG_users:{self._user_clustering_name}-{u_group}",
                                                      val=self._values_dict[u_group],
                                                      needs_full_recommendations=False))
    def get(self):
        return self._metric_objs_list
