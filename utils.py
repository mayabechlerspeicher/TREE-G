import numpy as np
import random
from treeg.node_treeg.graph_data_node_level import GraphData
import networkx as nx
import pickle


# import dill as pickle
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def balance_multiclass_data(X, y):
    #TODO: implement
    return

def balance_data(X, y):
    X = np.array(X)
    y = np.array(y)
    positive_ind = np.where(y > 0)[0]
    negative_ind = np.where(y == 0)[0]
    num_of_pos = len(positive_ind)
    num_of_neg = len(negative_ind)
    if num_of_pos < num_of_neg:
        neg_ind_balances = negative_ind[: num_of_pos]
        balanced_X = np.append(X[positive_ind], (X[neg_ind_balances]))
        balanced_y = np.append(y[positive_ind], (y[neg_ind_balances]))
    else:
        pos_ind_balances = positive_ind[: num_of_neg]
        balanced_X = np.append(X[pos_ind_balances], (X[negative_ind]))
        balanced_y = np.append(y[pos_ind_balances], (y[negative_ind]))
    c = list(zip(balanced_X, balanced_y))
    random.shuffle(c)
    X, y = zip(*c)
    return np.array(X), np.array(y)


def get_balanced_data_for_class(class_i, X, y):
    labels_by_curr_class = (y == class_i).astype(int)
    pos_ind = np.where((y == class_i).astype(int))[0]
    num_of_pos = len(pos_ind)
    neg_ind = np.where((y != class_i).astype(int))[0]
    neg_ind_balanced = neg_ind[: num_of_pos]
    balances_examples_ind = np.append(neg_ind_balanced, pos_ind)
    X_balanced = X[balances_examples_ind]
    y_balanced = labels_by_curr_class[balances_examples_ind]
    c = list(zip(np.array(X_balanced), np.array(y_balanced)))
    random.shuffle(c)
    X_balanced, y_balanced = zip(*c)

    return list(X_balanced), list(y_balanced)



def compute_dual_graph(G: GraphData):
    nx.Graph()
    # TODO
    return


def load_pickle_from_path(file_path_from_root):
    # full_path = os.path.join(ROOT_DIR, file_path_from_root)
    with open(file_path_from_root, 'rb') as f:
        return pickle.load(f)


def write_pickle_to_path(to_pickle, file_path):
    # full_path = os.path.join(ROOT_DIR, file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(to_pickle, f)


def powerset(s):
    x = len(s)
    all = []
    for i in range(1 << x):
        all.append([s[j] for j in range(x) if (i & (1 << j))])
    return all


def remove_nans_from_data(X, y):
    indices = np.logical_not(np.isnan(np.array(y)))
    X_no_nans = X[indices]
    y_no_nans = y[indices]
    return X_no_nans, y_no_nans
