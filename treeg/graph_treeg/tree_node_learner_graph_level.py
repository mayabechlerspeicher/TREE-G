import numpy as np
from typing import List, NamedTuple
from treeg.graph_treeg.trained_tree_node_graph_level import TrainedTreeNode
from treeg.graph_treeg.aggregator_graph_level import Aggregator, graph_level_aggregators
from treeg.graph_treeg.graph_data_graph_level import GraphData
from sklearn.tree import DecisionTreeRegressor


class TreeNodeLearnerParams(NamedTuple):
    walk_lens: List[int] = [0, 1, 2],
    max_attention_depth: int = 2,
    aggregators: List[Aggregator] = graph_level_aggregators,
    max_number_of_leafs: int = 10,
    min_leaf_size: int = 10,
    min_gain: float = 0.0,
    attention_types: List[int] = [1, 2, 3, 4],
    attention_type_sample_probability: float = 0.5


class TreeNodeLearner:
    def __init__(self,
                 params: TreeNodeLearnerParams,
                 active: List[int],
                 parent: "TreeNodeLearner" = None):

        self.params = params
        self.active = active
        self.parent = parent

        self.available_attentions = None
        self.potential_gain = None
        self.lte = None
        self.gt = None
        self.walk_len = None
        self.attention_depth = None
        self.attention_position = None
        self.active_attention_index = None
        self.attention_type = None
        self.aggregator = None
        self.feature_index = None
        self.value_as_leaf = None
        self.features_dimension = None
        self.trained_tree_node = None
        self.generated_attentions = None

        self.active_lte_ = None
        self.active_gt_ = None
        self.latent_feature_index_ = None
        self.thresh_ = None
        self.lte_value_ = None
        self.gt_value_ = None

        self.stats_dict = None
        self.node_count = 0
        self.depth = 0
        self.tree_depth = 0

    def get_available_attentions_indices(self):
        """
        This function translate the available attention sets to their indices in the following order:
        (0,-1) is the set of all vertices
        (i,j) is the attention generated at the node of distance i from the current node,
         where j is 0 for the left (leq) attention generated and 1 for the right attention generated
        :return: the set of indices of the available attention sets.
        """

        attention_indices = [(0, -1)]  # the set of all vertices
        p = self
        for i in range(0, self.params.max_attention_depth):
            if p.parent is None:
                break
            p = p.parent
            att = [(i, j) for j in range(1, len(p.generated_attentions[0]) + 1)]
            attention_indices += att
        return attention_indices

    def set_available_attentions_for_graph(self, graph, graph_index: int):
        p = self
        available_attentions = []
        for _ in range(0, self.params.max_attention_depth):
            if p.parent is None:
                break
            p = p.parent
            available_attentions = p.generated_attentions[graph_index] + available_attentions
        available_attentions = [list(range(0, graph.get_number_of_nodes()))] + available_attentions
        self.available_attentions[graph_index] = available_attentions

    def find_best_split(self, X: List[GraphData], y: np.array):
        labels = y[self.active]
        self.value_as_leaf = np.mean(labels)
        if len(self.active) < self.params.min_leaf_size:
            self.potential_gain = 0.0
            return 0.0
        self.available_attentions = [[] for _ in range(0, len(X))]
        latent_feature_vectors = []
        for graph_index in self.active:
            graph = X[graph_index]
            self.set_available_attentions_for_graph(graph, graph_index)
            available_attentions_for_graph = self.available_attentions[graph_index]
            latent_feature_vector = graph.get_latent_feature_vector(walk_lens=self.params.walk_lens,
                                                                    available_attentions=available_attentions_for_graph,
                                                                    aggregators=self.params.aggregators,
                                                                    attention_types=self.params.attention_types)
            latent_feature_vectors.append(latent_feature_vector)
        data = np.vstack(latent_feature_vectors)
        stump = DecisionTreeRegressor(max_depth=1, random_state=42)
        stump.fit(data, labels)
        if len(stump.tree_.value) < 3:
            return 0
        self.latent_feature_index_ = stump.tree_.feature[0]
        self.thresh_ = stump.tree_.threshold[0]
        self.lte_value_ = stump.tree_.value[1][0][0]
        self.gt_value_ = stump.tree_.value[2][0][0]
        feature_values = data[:, self.latent_feature_index_]
        active_lte_local = np.where(feature_values <= self.thresh_)[0].tolist()
        self.active_lte_ = [self.active[i] for i in active_lte_local]
        active_gt_local = np.where(feature_values > self.thresh_)[0].tolist()
        self.active_gt_ = [self.active[i] for i in active_gt_local]
        self.potential_gain = len(
            active_gt_local) * self.gt_value_ * self.gt_value_ + len(
            active_lte_local) * self.lte_value_ * self.lte_value_ - len(self.active) * \
                              stump.tree_.value[0][0][0] * \
                              stump.tree_.value[0][0][0]
        return self.potential_gain

    def sample_attention_types(self):
        attention_types = []
        for i in range(1, 5):
            s = np.random.binomial(1, self.params.attention_type_sample_probability)
            if s:
                attention_types.append(i)
        return attention_types

    def apply_best_split(self, X: List[GraphData], y: np.array):
        lte_sampled_atttention_types = self.sample_attention_types()
        if lte_sampled_atttention_types == []:
            lte_sampled_atttention_types = [1]
            lte_max_attention_depth = 0
        else:
            lte_max_attention_depth = self.params.max_attention_depth
        lte_new_params = TreeNodeLearnerParams(
            walk_lens=self.params.walk_lens,
            max_attention_depth=lte_max_attention_depth,
            aggregators=self.params.aggregators,
            max_number_of_leafs=self.params.max_number_of_leafs,
            min_gain=self.params.min_gain,
            min_leaf_size=self.params.min_leaf_size,
            attention_types=lte_sampled_atttention_types,
            attention_type_sample_probability=self.params.attention_type_sample_probability
        )
        lte_node = TreeNodeLearner(params=lte_new_params,
                                   active=self.active_lte_,
                                   parent=self)
        lte_node.value_as_leaf = np.mean(y[self.active_lte_])
        lte_node.depth = self.depth + 1

        gt_sampled_atttention_types = self.sample_attention_types()
        if gt_sampled_atttention_types == []:
            gt_sampled_atttention_types = [1]
            gt_max_attention_depth = 0
        else:
            gt_max_attention_depth = self.params.max_attention_depth
        gt_new_params = TreeNodeLearnerParams(
            walk_lens=self.params.walk_lens,
            max_attention_depth=gt_max_attention_depth,
            aggregators=self.params.aggregators,
            max_number_of_leafs=self.params.max_number_of_leafs,
            min_gain=self.params.min_gain,
            min_leaf_size=self.params.min_leaf_size,
            attention_types=gt_sampled_atttention_types,
            attention_type_sample_probability=self.params.attention_type_sample_probability
        )

        gt_node = TreeNodeLearner(params=gt_new_params,
                                  active=self.active_gt_,
                                  parent=self)
        gt_node.value_as_leaf = np.mean(y[self.active_gt_])
        gt_node.depth = self.depth + 1

        self.gt = gt_node
        self.lte = lte_node

        available_attention_indices = self.get_available_attentions_indices()
        sizes = [len(self.params.walk_lens), len(available_attention_indices), len(self.params.attention_types),
                 len(self.params.aggregators), X[0].get_number_of_features()]
        walk_len_index, active_attention_index, attention_type_index, aggregator_index, feature_index = self.get_index(
            self.latent_feature_index_, sizes)
        self.active_attention_index = active_attention_index
        self.walk_len = self.params.walk_lens[walk_len_index]
        selected_attention_indices = available_attention_indices[active_attention_index]
        self.attention_depth = selected_attention_indices[0]
        self.attention_position = selected_attention_indices[1]
        self.attention_type = self.params.attention_types[attention_type_index]
        self.aggregator = self.params.aggregators[aggregator_index]
        self.feature_index = feature_index

        self.generated_attentions = [[] for _ in range(0, len(X))]
        for i in self.active:
            g = X[i]

            _, generated_attentions, _ = g.get_score_and_generated_attentions(walk_len=self.walk_len,
                                                                              attention_set=
                                                                              self.available_attentions[i][
                                                                                  active_attention_index],
                                                                              attention_type=self.attention_type,
                                                                              agg=self.aggregator,
                                                                              thresh=self.thresh_,
                                                                              feature_index=feature_index
                                                                              )

            self.generated_attentions[i] = generated_attentions

        return self.attention_depth, self.walk_len, self.aggregator, self.feature_index, self.attention_type

    def init_stats_dict(self):
        stats_dict = {'attention_depth': {}, 'walk_len': {}, 'aggregator': {}, 'feature_index': {},
                      'attention_type': {}}
        for attention in range(self.params.max_attention_depth + 1):
            stats_dict['attention_depth'][attention] = 0
        for walk_len in self.params.walk_lens:
            stats_dict['walk_len'][walk_len] = 0
        for aggregator in self.params.aggregators:
            stats_dict['aggregator'][aggregator.get_name()] = 0
        for feature in range(self.features_dimension):
            stats_dict['feature_index'][feature] = 0
        for attention_type in [1, 2, 3, 4]:
            stats_dict['attention_type'][attention_type] = 0

        return stats_dict

    def update_stats_dict(self, stats_dict, attention_depth, walk_len, aggregator, feature_index, attention_type):

        stats_dict['attention_depth'][attention_depth] += 1
        stats_dict['walk_len'][walk_len] += 1
        stats_dict['aggregator'][aggregator.get_name()] += 1
        stats_dict['feature_index'][feature_index] += 1
        stats_dict['attention_type'][attention_type] += 1

    def fit(self, X: List[GraphData], y: np.array):
        self.features_dimension = X[0].get_number_of_features()
        tiny = np.finfo(float).tiny
        min_gain = self.params.min_gain
        if min_gain <= tiny:
            min_gain = tiny
        stats_dict = self.init_stats_dict()
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
            attention_depth, walk_len, aggregator, feature_index, attention_type = leaf_to_split.apply_best_split(X, y)
            self.update_stats_dict(stats_dict=stats_dict, attention_depth=attention_depth, walk_len=walk_len,
                                   aggregator=aggregator,
                                   feature_index=feature_index, attention_type=attention_type)
            lte = leaf_to_split.lte
            gt = leaf_to_split.gt
            self.tree_depth = max(self.tree_depth, lte.depth, gt.depth)
            self.node_count += 2
            potential_gains += [lte.find_best_split(X, y), gt.find_best_split(X, y)]
            leafs += [lte, gt]
            total_gain += gain

        self.stats_dict = stats_dict
        L2 = 0
        for leaf in leafs:
            labels = y[leaf.active]
            L2 += sum((leaf.value_as_leaf - labels) ** 2)
        return L2, total_gain, stats_dict

    def predict(self, g: GraphData):
        attentions_cache = [[list(range(0, g.get_number_of_nodes()))]]
        histogram = np.zeros(g.get_number_of_nodes())
        p = self
        while p.lte is not None:
            attentions = []
            for a in attentions_cache:
                attentions += a

            score, new_attentions, selected_attention = g.get_score_and_generated_attentions(walk_len=p.walk_len,
                                                                                             attention_set=
                                                                                             p.available_attentions[
                                                                                                 p.active_attention_index],
                                                                                             attention_type=p.attention_type,
                                                                                             agg=p.aggregator)

            histogram[selected_attention] += 1
            if len(attentions_cache) > p.max_attention_depth:
                if len(attentions_cache) > 1:
                    attentions_cache.pop(1)
            attentions_cache.append(new_attentions)
            if score <= p.thresh_:
                p = p.lte
            else:
                p = p.gt
        return p.value_as_leaf, histogram

    @staticmethod
    def get_index(latent_feature_index: int, sizes: List[int]) -> List[int]:
        """
        Translate the chosen index of the latent feature vector for each example to the chosen walk len, attention_depth, feature and aggregator
        :param latent_feature_index: the chosen "latent feature" index
        :param sizes: #walks_lens, #available_attetnions, # attention_types, #aggregators, #features
        :return: a list with the sizes in the following order:  walk_len, attention index in available attentions, attention_type_index, aggregator index, feature index
        """

        indices = []
        for n in range(0, len(sizes)):
            s = sizes[len(sizes) - 1 - n]
            i = latent_feature_index % s
            latent_feature_index = int((latent_feature_index - i) / s)
            indices.insert(0, i)
        return indices

    def build_trained_tree_and_get_root(self):
        if self.gt is None:
            self.trained_tree_node = TrainedTreeNode(gt=None, lte=None, feature_index=-1, thresh=0,
                                                     value_as_leaf=self.value_as_leaf, walk_len=-1,
                                                     active_attention_index=-1,
                                                     max_attention_depth=self.params.max_attention_depth,
                                                     aggregator=None,
                                                     attention_type=self.attention_type,
                                                     )
        else:
            self.trained_tree_node = TrainedTreeNode(gt=None, lte=None, feature_index=self.feature_index,
                                                     thresh=self.thresh_, value_as_leaf=self.value_as_leaf,
                                                     walk_len=self.walk_len,
                                                     active_attention_index=self.active_attention_index,
                                                     max_attention_depth=self.params.max_attention_depth,
                                                     aggregator=self.aggregator,
                                                     attention_type=self.attention_type,
                                                    )

            self.trained_tree_node.lte = self.lte.build_trained_tree_and_get_root()
            self.trained_tree_node.gt = self.gt.build_trained_tree_and_get_root()

        return self.trained_tree_node
