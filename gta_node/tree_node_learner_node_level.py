import numpy as np
from typing import List, NamedTuple
from gta_node.graph_data_node_level import GraphData
from sklearn.tree import DecisionTreeRegressor


def intersect(lst1, lst2):
    if isinstance(lst1, np.ndarray):
        lst1 = lst1.tolist()
    if isinstance(lst2, np.ndarray):
        lst2 = lst2.tolist()
    return list(set(lst1) & set(lst2))


class TreeNodeLearnerParams(NamedTuple):
    graph: GraphData = 0,
    walks_lens: List[int] = [0, 1, 2],
    max_attention_depth: int = 2,
    max_number_of_leafs: int = 10,
    min_leaf_size: int = 10,
    min_gain: float = 0.0,
    attention_types: List[int] = [1, 4],


class TreeNodeLearner:
    def __init__(self,
                 params: TreeNodeLearnerParams,
                 active: List[int],
                 parent: "TreeNodeLearner" = None):

        self.to_be_attention_type = None
        self.params = params
        self.active = active
        self.parent = parent

        self.walk_len = None
        self.feature_index = None
        self.generated_attentions = None
        self.all_predictions = None
        self.active_attention = None
        self.value = None
        self.potential_gain_ = None
        self.available_attentions = None
        self.attention_type = None
        self.feature_dimension = None
        self.lte = None
        self.gt = None

        self.latent_feature_index = None
        self.thresh_ = None
        self.lte_value = None
        self.gt_value = None
        self.to_be_generated_attentions = None
        self.to_be_feature_index = None
        self.to_be_active_attention = None
        self.to_be_depth = None
        self.active_gt = None
        self.active_lte = None

    def get_attentions_indices(self):
        attention_indices = [(-1, -1)]
        p = self
        for i in range(0, self.params.max_attention_depth):
            if p.parent is None:
                break
            p = p.parent
            att = [(i, j) for j in range(0, len(p.generated_attentions[0]))]
            attention_indices += att
        return attention_indices

    def set_available_attentions(self):
        p = self
        available_attentions = []
        for _ in range(0, self.params.max_attention_depth):
            if p.parent is None:
                break
            p = p.parent
            available_attentions = p.generated_attentions + available_attentions
        available_attentions = [list(range(0, self.params.graph.get_number_of_nodes()))] + available_attentions
        self.available_attentions = available_attentions

    def get_feature_matrix(self):  # attention logic
        p = self
        available_attentions = []
        for _ in range(0, self.params.max_attention_depth):
            if p.parent is None:
                break
            p = p.parent
            available_attentions = p.generated_attentions + available_attentions
        available_attentions = [list(range(0, self.params.graph.get_number_of_nodes()))] + available_attentions
        return self.params.graph.get_feature_vector(self.params.walks_lens,
                                                    available_attentions), available_attentions

    def find_best_split(self, X: List[int], y: np.array):
        active_train = intersect(self.active, X)
        labels = y[active_train]
        self.value = np.mean(labels)
        self.feature_dimension = self.params.graph.get_number_of_features()
        if len(labels) < self.params.min_leaf_size:
            self.potential_gain_ = 0.0
            return 0.0
        self.set_available_attentions()
        latent_feature_vector = self.params.graph.get_feature_vector(walk_lens=self.params.walks_lens,
                                                                     available_attentions=self.available_attentions,
                                                                     attention_types=self.params.attention_types,
                                                                     )
        latent_train_data = latent_feature_vector[active_train, :]
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(latent_train_data, labels)
        if len(stump.tree_.value) < 3:
            return 0
        self.latent_feature_index = stump.tree_.feature[0]
        self.thresh_ = stump.tree_.threshold[0]
        self.lte_value = stump.tree_.value[1][0][0]
        self.gt_value = stump.tree_.value[2][0][0]
        feature_values = latent_train_data[:, self.latent_feature_index]
        active_lte_local = np.where(feature_values <= self.thresh_)[0].tolist()
        self.active_lte = intersect(self.active, active_lte_local)
        active_gt_local = np.where(feature_values > self.thresh_)[0].tolist()
        self.active_gt = intersect(self.active, active_gt_local)
        self.potential_gain_ = len(self.active_gt) * self.gt_value * self.gt_value + \
                               len(self.active_lte) * self.lte_value * self.lte_value - \
                               len(self.active) * stump.tree_.value[0][0][0] * stump.tree_.value[0][0][0]

        walk_len_index, active_attention_index, attention_type_index, feature_index = self.params.graph.get_index(
            self.latent_feature_index,
            [len(self.params.walks_lens), len(self.available_attentions), len(self.params.attention_types),
             self.feature_dimension])
        self.to_be_attention_type = self.params.attention_types[attention_type_index]
        self.to_be_depth = self.params.walks_lens[walk_len_index]
        self.to_be_active_attention = self.available_attentions[active_attention_index]
        self.to_be_feature_index = feature_index
        self.to_be_generated_attentions = \
            [intersect(self.to_be_active_attention, active_gt_local),
             intersect(self.to_be_active_attention, active_lte_local)]

        return self.potential_gain_

    def apply_best_split(self):
        lte_node = TreeNodeLearner(params=self.params, active=self.active_lte, parent=self)
        lte_node.value = self.lte_value

        gt_node = TreeNodeLearner(params=self.params, active=self.active_gt, parent=self)
        gt_node.value = self.gt_value

        self.gt = gt_node
        self.lte = lte_node

        self.walk_len = self.to_be_depth
        self.feature_index = self.to_be_feature_index
        self.active_attention = self.to_be_active_attention
        self.generated_attentions = self.to_be_generated_attentions

    def fit(self, X: List[int], y: np.array):
        tiny = np.finfo(float).tiny
        min_gain = self.params.min_gain
        if min_gain <= tiny:
            min_gain = tiny
        leafs = [self]
        total_gain = 0
        potential_gains = [self.find_best_split(X, y)]
        for _ in range(1, self.params.max_number_of_leafs):
            index_max = np.argmax(potential_gains)
            gain = potential_gains[index_max]
            if gain < min_gain:
                break
            leaf_to_split = leafs.pop(index_max)
            potential_gains.pop(index_max)
            leaf_to_split.apply_best_split()
            lte = leaf_to_split.lte
            gt = leaf_to_split.gt
            potential_gains += [lte.find_best_split(X, y), gt.find_best_split(X, y)]
            leafs += [lte, gt]
            total_gain += gain

        l2 = 0
        for leaf in leafs:
            for i in range(0, len(X)):
                if X[i] in leaf.active:
                    l2 += (leaf.value - y[i]) ** 2
        return l2, total_gain

    def predict_all(self, predictions: np.array = None) -> np.array:
        if predictions is None:
            predictions = np.zeros(shape=(self.params.graph.get_number_of_nodes()))
        if self.lte is None:
            for a in self.active:
                predictions[a] = self.value
        else:
            predictions = self.lte.predict_all(predictions)
            predictions = self.gt.predict_all(predictions)
        self.all_predictions = predictions
        return predictions

    def predict(self, x: int):
        if not (hasattr(self, 'all_predictions')):
            self.predict_all()
        return self.all_predictions[x]

    def print(self, indent: str = ""):
        if self.gt is None:
            print(indent, "-->", self.value)
        else:
            print(indent, "f%d _thresh %3f depth %2d" % (self.feature_index, self.thresh_, self.walk_len))
            self.lte.print_tree(indent + "  ")
            self.gt.print_tree(indent + "  ")
