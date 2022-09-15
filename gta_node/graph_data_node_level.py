import numpy as np
from typing import List
from scipy import sparse
from scipy.sparse import diags

class GraphData:
    def __init__(self, adj_mat: np.array, features: np.array, label: np.generic = np.NAN):
        n1, n2 = np.shape(adj_mat)
        if n1 != n2:
            raise ValueError("graph must be a square matrix")
        t1, t2 = np.shape(features)
        if n1 != t1:
            raise ValueError("the number of rows of features does not match the number of nodes in the graph")

        self.adj_mat = adj_mat
        self.features = features
        self.label = label
        self.adj_powers = np.NAN

    def compute_walks(self, max_walk_len):
        walks = []
        for walk_len in range(max_walk_len + 1):
            walks.append(np.linalg.matrix_power(self.adj_mat, walk_len))
        self.adj_powers = np.array(walks)

    def propagate_with_attention(self, walk_len, attention_set, attention_type):
        n_nodes = len(self.features)
        not_in_attention = np.array(list(set(range(n_nodes)) - set(attention_set)))
        if attention_type == 1:
            masked_ad = self.adj_powers[walk_len].copy()
            for i in not_in_attention:
                masked_ad[:, i] = 0
            return np.matmul(masked_ad, self.features)
        elif attention_type == 4:
            # ad = np.linalg.matrix_power(adj_copy, graph_depth)
            ad = self.adj_powers[walk_len]
            diag = ad.diagonal()
            masked_ad = np.zeros_like(ad)
            np.fill_diagonal(masked_ad, diag)
            return np.matmul(masked_ad, self.features)
        else:
            print(
                'Invalid attention type ' + str(attention_type) + '. Valid attentions for node-level tasks are 1 and 4')
            return None

    def get_feature_vector(self, walk_lens: List[int],
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

    @staticmethod
    def get_index(latent_feature_index: int, sizes: List[int]) -> List[int]:
        """
        Translate the chosen index of the latent feature vector for each example to the chosen walk len, attention_depth,feature
        :param latent_feature_index: the chosen "latent feature" index
        :param sizes: #walks_lens, #available_attetnions, # attention_types, #features
        :return: a list with the sizes in the following order:  walk_len, attention index in available attentions, attention_type_index, feature index
        """
        indices = []
        for n in range(0, len(sizes)):
            s = sizes[len(sizes) - 1 - n]
            i = latent_feature_index % s
            latent_feature_index = int((latent_feature_index - i) / s)
            indices.insert(0, i)
        return indices

    def get_number_of_nodes(self):
        return np.shape(self.adj_mat)[0]

    def get_number_of_features(self):
        return np.shape(self.features)[1]

    def get_generated_attentions(self, index_in_feature_vector: int,
                                 threshold: np.generic,
                                 depths: List[int],
                                 attention_types: List[int],
                                 attentions: List[np.array]) -> List[List[int]]:

        walk_len_index, active_attention_index, attention_type_index, feature_index = \
            self.get_index(index_in_feature_vector,
                           [len(depths), len(attentions), len(attention_types), self.features.shape[1]])
        depth = depths[walk_len_index]
        attention_type = attention_types[attention_type_index]
        attention = attentions[active_attention_index]
        p = self.propagate_with_attention(walk_len=depth, attention_set=attention, attention_type=attention_type)
        col = p[:, feature_index]
        gt_attention = [i for i in attention if (col[i] > threshold)]
        lte_attention = [i for i in attention if (col[i] <= threshold)]
        return [gt_attention, lte_attention]


class SparseGraphData(GraphData):
    def __init__(self, adj_mat: np.array, features: np.array, label: np.generic = np.NAN):
        super().__init__(adj_mat, features, label)
        self.sparse_adj = sparse.csr_matrix(adj_mat)
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
