import numpy as np
from typing import List
from scipy import sparse
from scipy.sparse import diags
from functools import lru_cache


class GraphData:
    def __init__(self, adj_mat: np.array, features: np.array, labels: np.generic = np.NAN):
        if adj_mat is not None:
            n1, n2 = np.shape(adj_mat)
            if n1 != n2:
                raise ValueError("graph must be a square matrix")
            t1, t2 = np.shape(features)
            if n1 != t1:
                raise ValueError("the number of rows of features does not match the number of nodes in the graph")

        self.adj_mat = adj_mat
        self.features = features
        self.labels = labels
        self.adj_powers = None

    def compute_walks(self, max_walk_len):
        walks = []
        for walk_len in range(max_walk_len + 1):
            walks.append(np.linalg.matrix_power(self.adj_mat, walk_len))
        self.adj_powers = np.array(walks)

    @lru_cache
    def propagate_with_attention(self, walk_len: int, attention_set: str, attention_type: int):
        if attention_set == '[]':
            attention_set = []
        else:
            attention_set = [int(x) for x in attention_set[1:-1].split(',')]
        if attention_type == 1:
            masked_ad = np.zeros_like(self.adj_mat)
            masked_ad[:, attention_set] = self.adj_powers[walk_len][:, attention_set]
            return np.matmul(masked_ad, self.features)[attention_set, :]
        elif attention_type == 4:
            diag = self.adj_powers[walk_len].diagonal()
            masked_ad = np.zeros_like(self.adj_mat)
            np.fill_diagonal(masked_ad, diag)
            return np.matmul(masked_ad, self.features)[attention_set, :]
        else:
            print(
                'Invalid attention type ' + str(attention_type) + '. Valid attentions for node-level tasks are 1 and 4')
            return None

    def get_feature_vectors_for_all_vertices(self, walk_lens: List[int],
                                             available_attentions: List[np.array],
                                             attention_types: List[int] = [1, 4],
                                             ) -> np.array:
        feature_list = []
        for walk_len in walk_lens:
            for attention in available_attentions:
                for attention_type in attention_types:
                    p = self.propagate_with_attention(walk_len=walk_len, attention_set=attention,
                                                      attention_type=attention_type)
                    feature_list.append(p)
        all_features_all_vertices = np.concatenate(feature_list, axis=1)
        return all_features_all_vertices

    def get_number_of_nodes(self):
        return np.shape(self.adj_mat)[0]

    def get_number_of_features(self):
        return np.shape(self.features)[1]

    def get_generated_attentions(self,
                                 attention_type: int,
                                 walk_len: int,
                                 attention_set: List[int],
                                 feature_index: int,
                                 threshold: float) -> List[List[int]]:

        p = self.propagate_with_attention(walk_len=walk_len, attention_set=attention_set, attention_type=attention_type)
        col = p[:, feature_index]
        gt_attention = [i for i in attention_set if (col[i] > threshold)]
        lte_attention = [i for i in attention_set if (col[i] <= threshold)]
        return [gt_attention, lte_attention]


class SparseGraphData(GraphData):
    def __init__(self, sparse_adj_mat: sparse.csr_matrix, features: np.array, labels: np.generic = np.NAN):
        super().__init__(None, features, labels)
        self.sparse_adj = sparse_adj_mat
        self.sparse_features = sparse.csr_matrix(features)
        self.sparse_adj_powers = np.NAN

    def compute_walks(self, max_walk_len):
        walks = []
        for walk_len in range(max_walk_len + 1):
            powered = self.sparse_adj ** walk_len
            if not (type(powered) == sparse.csr_matrix):
                powered = powered.tocsr()
            walks.append(powered)
        self.sparse_adj_powers = np.array(walks)

    def propagate_with_attention(self, walk_len, attention_set, attention_type):
        n_nodes = self.get_number_of_nodes()
        not_in_attention = np.array(list(set(range(n_nodes)) - set(
            attention_set)))
        if attention_type == 1:
            sparse_power = self.sparse_adj_powers[walk_len].tolil()
            sparse_features_masked = self.sparse_features.tolil()
            sparse_features_masked[:, not_in_attention] = 0
            propagated_features = sparse_power * sparse_features_masked
            return propagated_features.toarray()
        elif attention_type == 4:
            masked_sparse_power = self.sparse_adj_powers[walk_len]
            diag = masked_sparse_power.diagonal()
            sparse_power_diag = diags(diag, [0])
            propagated_features = sparse_power_diag * self.sparse_features
            return propagated_features.toarray()
        else:
            print(
                'Invalid attention type ' + str(attention_type) + '. Valid attentions for node-level tasks are 1 and 4')
            return None
