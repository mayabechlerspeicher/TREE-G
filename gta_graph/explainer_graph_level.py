import numpy as np
from gta_graph.graph_data_graph_level import GraphData


def get_hist_over_trees(model, g: GraphData):
    histogram_acc = np.zeros(g.get_number_of_nodes())
    trees = model.estimators_
    for tree in trees:
        pred = tree[0].trained_tree_root_.predict(g)
        hist = pred[1]
        histogram_acc += hist
    return histogram_acc


def explain_nodes(model, g: GraphData):
    histogram_acc = get_hist_over_trees(model, g)
    normalized_hist = histogram_acc / np.linalg.norm(histogram_acc)
    sort_order = np.argsort(normalized_hist)
    return [normalized_hist[sort_order], sort_order]


def get_nodes_importance(model, g: GraphData):
    num_of_nodes = g.get_number_of_nodes()
    nodes_scores_over_trees = np.zeros(num_of_nodes)
    trees = model.estimators_
    trees_nodes_scores = np.zeros((len(trees), num_of_nodes))
    for idx, tree in enumerate(trees):
        nodes_scores = tree[0].nodes_scores(g)
        trees_nodes_scores[idx] = nodes_scores

    for node in range(num_of_nodes):
        scores_sum_over_trees = np.sum(trees_nodes_scores[:, node])
        nodes_scores_over_trees[node] = scores_sum_over_trees

    nodes_scores_sums = np.sum(nodes_scores_over_trees)
    nodes_importance = nodes_scores_over_trees / nodes_scores_sums

    return nodes_importance


def get_attention_nodes(model, g: GraphData):
    acc_hist = get_hist_over_trees(model, g)
    nodes_by_count = np.argsort(acc_hist)
    return nodes_by_count, acc_hist


def init_empty_dict_like(dict_to_copy):
    super_dict = {}
    for k in dict_to_copy.keys():
        super_dict[k] = {}
        for l in dict_to_copy[k].keys():
            super_dict[k][l] = 0
    return super_dict


def sum_stats_from_all_trees(model):
    trees = model.estimators_
    super_dict = init_empty_dict_like(trees[0][0].stats_dict)
    for tree in trees:
        curr_dict = tree[0].stats_dict
        for k in curr_dict.keys():
            for l in curr_dict[k].keys():
                super_dict[k][l] += curr_dict[k][l]
    return super_dict


def print_stats_dicts(model):
    trees = model.estimators_
    for tree in trees:
        print(tree.stats_dict)


def log_cross_val_stats(all_stats_dicts, logger):
    for idx, stat_dict in enumerate(all_stats_dicts):
        logger.info("Fold %d stats:" % idx)
        logger.info(str(stat_dict))
