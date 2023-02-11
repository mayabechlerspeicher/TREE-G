from functools import lru_cache
import numpy as np
from numpy.linalg import matrix_power
from typing import List
from gta_graph.aggregator_graph_level import Aggregator
from scipy import sparse
from scipy.sparse import diags


class GraphData:
    def __init__(self, adj_mat: np.array, features: np.array, label: float):
        if adj_mat is not None:
            n1, n2 = np.shape(adj_mat)
            if n1 != n2:
                raise ValueError("graph must be a square matrix")
            t1, t2 = np.shape(features)
            if n1 != t1:
                raise ValueError("the number of rows of features does not match the number of nodes in the graph")

        self.adj_mat = adj_mat
        self.features = features
        self.label = label
        self.adj_powers = None

    def compute_walks(self, max_walk_len):
        walks = []
        for walk_len in range(max_walk_len + 1):
            walks.append(np.linalg.matrix_power(self.adj_mat, walk_len))
        self.adj_powers = np.array(walks)

    @lru_cache
    def propagate_with_attention(self, walk_len: int, attention_set: str, attention_type: int):
        """
        Attention type 1 - zero out columns (target attention)
        Attention type 2 - zero out rows (source attention)
        Attention type 3 - zero out rows and columns (source + target attention)
        Attention type 4 - diagonal only (diagonal(cycles) attention)
        :param walk_len: The length of the walk in the range [0, max_walk_len]
        :param attention_set: This is a string in order to support chache
        :param attention_type: The type of attention in the range [1,4]
        :return:
        """
        if attention_set == '[]':
            attention_set = []
        else:
            attention_set = [int(x) for x in attention_set[1:-1].split(',')]
        if attention_type == 1:
            masked_ad = np.zeros_like(self.adj_mat)
            masked_ad[:, attention_set] = self.adj_powers[walk_len][:, attention_set]
            return np.matmul(masked_ad, self.features)
        elif attention_type == 2:
            masked_ad = np.zeros_like(self.adj_mat)
            masked_ad[attention_set, :] = self.adj_powers[walk_len][attention_set, :]
            return np.matmul(masked_ad, self.features)[attention_set, :]
        elif attention_type == 3:
            masked_ad = np.zeros_like(self.adj_mat)
            masked_ad[:, attention_set] = self.adj_powers[walk_len][:, attention_set]
            masked_ad[attention_set, :] = self.adj_powers[walk_len][attention_set, :]
            return np.matmul(masked_ad, self.features)[attention_set, :]
        elif attention_type == 4:
            diag = np.zeros(shape=(self.adj_mat.shape[0]))
            diag[attention_set] = self.adj_powers[walk_len].diagonal()[attention_set]
            masked_ad = np.zeros_like(self.adj_mat)
            np.fill_diagonal(masked_ad, diag)
            return np.matmul(masked_ad, self.features)[attention_set, :]
        else:
            print('Invalid attention type ' + str(attention_type) + '. Valid attentions for graph-level tasks are 1-5')
            return None

    def get_latent_feature_vector(self, walk_lens: List[int],
                                  available_attentions: List[np.array],
                                  aggregators: List[Aggregator],
                                  attention_types: List[int] = [1, 2, 3, 4]) -> np.array:
        latent_feature_vector = []
        for walk_len in walk_lens:
            for attention in available_attentions:
                for attention_type in attention_types:
                    pa = self.propagate_with_attention(walk_len, str(attention), attention_type)
                    for agg in aggregators:
                        for col_att in pa.T:
                            if attention == []:
                                latent_feature_vector.append(agg.get_score(np.empty(0)))
                            else:
                                latent_feature_vector.append(agg.get_score(col_att))
        return np.array(latent_feature_vector)

    def get_number_of_nodes(self):
        return np.shape(self.adj_mat)[0]

    def get_number_of_features(self):
        return np.shape(self.features)[1]

    def get_score_and_generated_attentions(self, walk_len: int, attention_set: List[int], attention_type: int, agg,
                                           feature_index: int, thresh: float):

        pa = self.propagate_with_attention(walk_len, str(attention_set), attention_type)
        if attention_set == []:
            cola = np.empty(0)
        else:
            cola = pa[:, feature_index]
        return agg.get_score(cola), agg.get_generated_attentions(cola, thresh), attention_set

    def get_label(self):
        return self.label


class SparseGraphData(GraphData):
    def __init__(self, edge_index: np.array, features: np.array, label: np.generic = np.NAN):
        super().__init__(None, None, label)
        self.sparse_adj = sparse.csr_matrix(edge_index)
        self.sparse_features = sparse.csr_matrix(features)
        self.sparse_adj_powers = np.NAN
        self._num_features = features.shape[1]
        self._num_nodes = features.shape[0]

    def compute_walks(self, max_walk_len):
        walks = []
        for walk_len in range(max_walk_len + 1):
            powered = self.sparse_adj ** walk_len
            if not (type(powered) == sparse.csr_matrix):
                powered = powered.tocsr()
            walks.append(powered)
        self.sparse_adj_powers = np.array(walks)

    def propagate_with_attention(self, walk_len, attention_set, attention_type=2):
        n_nodes = self.get_number_of_nodes()
        not_in_attention = np.array(list(set(range(n_nodes)) - set(attention_set)))
        if attention_type == 1:
            masked_sparse_power = self.sparse_adj_powers[walk_len].tolil()
            for i in not_in_attention:
                masked_sparse_power[:, i] = 0
            propagated_features = masked_sparse_power * self.sparse_features
            return propagated_features.toarray()
        elif attention_type == 2:
            masked_sparse_power = self.sparse_adj_powers[walk_len].tolil()
            for i in not_in_attention:
                masked_sparse_power[i, :] = 0
            propagated_features = masked_sparse_power * self.sparse_features
            return propagated_features.toarray()[attention_set, :]
        elif attention_type == 3:
            masked_sparse_power = self.sparse_adj_powers[walk_len].tolil()
            for i in not_in_attention:
                masked_sparse_power[:, i] = 0
                masked_sparse_power[i, :] = 0
            propagated_features = masked_sparse_power * self.sparse_features
            return propagated_features.toarray()[attention_set, :]
        elif attention_type == 4:
            masked_sparse_power = self.sparse_adj_powers[walk_len]
            diag = masked_sparse_power.diagonal()
            sparse_power_diag = diags(diag, 0)
            propagated_features = sparse_power_diag * self.sparse_features
            return propagated_features.toarray()[attention_set, :]

        else:
            print('Invalid attention type ' + str(attention_type) + '. Valid attentions for graph-level tasks are 1-5')
            return None

    def get_number_of_nodes(self):
        return self._num_nodes

    def get_number_of_features(self):
        return self._num_features
