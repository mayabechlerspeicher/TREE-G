import numpy as np
from typing import Callable, List


class Aggregator:
    def __init__(self,
                 agg: Callable[[np.array], np.generic],
                 attention_set_generator: Callable[[np.array, np.generic], List[List[int]]],
                 name: str
                 ):

        self.aggregator = agg
        self.attention_set_generator = attention_set_generator
        self.name = name

    def get_score(self, activations: np.array) -> np.generic:
        if activations.size == 0:
            return -1000
        return self.aggregator(activations)

    def get_generated_attentions(self, activations: np.array, threshold: float):
        if activations.size == 0:
            return [[] for _ in self.attention_set_generator(np.zeros(1), threshold)]

        new_attentions = []
        raw_attentions = self.attention_set_generator(activations, threshold)
        for raw in raw_attentions:
            if type(raw) is tuple:
                new_attentions.append(raw[0].tolist())
            else:
                new_attentions.append(raw)
        return new_attentions

    def get_name(self):
        return self.name


def _generate_attention_sets_normalized(x, threshold):
    """
    Splits the examples by the normalized threshold. The normalization is by the size of the attention set.
    First are the (attention set) examles greater or equal to the normalized threshold,
    Seconds are the examples less than the normalized threshold
    :param x: The active attention set.
    :param threshold: The threshold set in the
    :return: An array of size two,
    where the first element is the indices of the examples greater or equal to the normalized threshold
    and the second element is the indices of the examples less than the normalized threshold.
    """
    return [np.where(x >= threshold / x.size),
            np.where(x < threshold / x.size)]  # the notrmalization is on the size of the attention set


def _generate_attention_sets_plain(x, threshold):
    return [np.where(x >= threshold), np.where(x < threshold)]


sum = Aggregator(lambda x: np.sum(x), _generate_attention_sets_normalized, "sum")

avg = Aggregator(lambda x: np.mean(x), _generate_attention_sets_plain, "avg")

max = Aggregator(lambda x: np.max(x), _generate_attention_sets_plain, "max")

min = Aggregator(lambda x: np.min(x), _generate_attention_sets_plain, "min")

graph_level_aggregators = [sum, avg, max, min]