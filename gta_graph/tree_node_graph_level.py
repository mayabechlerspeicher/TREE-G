import numpy as np
from gta_graph.aggregator_graph_level import Aggregator
from gta_graph.graph_data_graph_level import GraphData


class TrainedTreeNode:
    def __init__(self,
                 gt: "TrainedTreeNode" = None,
                 lte: "TrainedTreeNode" = None,
                 feature_index: int = None,
                 thresh: float = None,
                 value_as_leaf: float = None,
                 walk_len: int = None,
                 active_attention_index: int = None,
                 max_attention_depth: int = None,
                 aggregator: Aggregator = None,
                 attention_type: int = None,
                 ):
        self.gt = gt
        self.lte = lte
        self.aggregator = aggregator
        self.value_as_leaf = value_as_leaf
        self.thresh = thresh
        self.active_attention_index = active_attention_index
        self.attention_type = attention_type
        self.walk_len = walk_len
        self.feature_index = feature_index
        self.max_attention_depth = max_attention_depth

    def print(self, indent=""):
        if self.gt is None:
            print(indent, "-->", self.value_as_leaf)
        else:
            print(indent, "f%d _thresh %3f depth %2d function %5s att_ind %d" % (
                self.feature_index, self.thresh, self.walk_len, self.aggregator.get_name(),
                self.active_attention_index))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")

    def predict(self, g: GraphData):
        attentions_cache = [[list(range(0, g.get_number_of_nodes()))]]
        histogram = np.zeros(g.get_number_of_nodes())
        pnt = self
        while pnt.lte is not None:
            attentions = []
            for a in attentions_cache:
                attentions += a
            attention = attentions[pnt.active_attention_index]
            pa = g.propagate_with_attention(pnt.walk_len, attention, pnt.attention_type)
            col = pa[:, pnt.feature_index]
            score = pnt.aggregator.get_score(col)
            new_attentions = pnt.aggregator.get_generated_attentions(col, pnt.thresh)
            selected_attention = attention
            histogram[selected_attention] += 1
            if len(attentions_cache) > pnt.max_attention_depth:
                if len(attentions_cache) > 1:
                    attentions_cache.pop(1)
            attentions_cache.append(new_attentions)

            if score <= pnt.thresh:
                pnt = pnt.lte
            else:
                pnt = pnt.gt
        return pnt.value_as_leaf, histogram
