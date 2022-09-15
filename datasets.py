from torch_geometric.datasets import TUDataset, Planetoid, GitHub
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np


def get_molhiv_dataset(formater):
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()
    train_graphs_list = dataset[split_idx["train"]]
    test_graphs_list = dataset[split_idx["test"]]
    X_train, y_train = formater.pyg_data_list_to_tree_graph_data_list(train_graphs_list)
    X_test, y_test = formater.pyg_data_list_to_tree_graph_data_list(test_graphs_list)
    return X_train, list(np.array(y_train).flatten()),  X_test, list(np.array(y_test).flatten())


def get_arxiv_dataset(formater):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = formater.pyg_data_to_tree_graph_data(dataset.data)
    return graph, train_idx, valid_idx, test_idx


def TU_MUTAG(formater):
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_MUTAG_SPARSE(formater):
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    X, y = formater.pyg_data_list_to_sparse_graph_data_list(dataset)
    return X, y


def TU_NCI1(formater):
    dataset = TUDataset(root='data/TUDataset', name='NCI1')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y


def TU_NCI109(formater):
    dataset = TUDataset(root='data/TUDataset', name='NCI109')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_PTCMR(formater):
    dataset = TUDataset(root='data/TUDataset', name='PTC_MR')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_IMDBB(formater):
    dataset = TUDataset(root='data/TUDataset', name='IMDB-BINARY')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_IMDBM():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-MULTI')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_DD(formater):
    dataset = TUDataset(root='data/TUDataset', name='DD')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y


def TU_PROTEINS(formater):
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS_full')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y


def TU_MDB_BINARY(formater):
    dataset = TUDataset(root='data/TUDataset', name='MDB-BINARY')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def TU_MUTAGANECY(formater):
    dataset = TUDataset(root='data/TUDataset', name='Mutagenicity')
    X, y = formater.pyg_data_list_to_tree_graph_data_list(dataset)
    return X, y

def Planetoid_CORA(formater):
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    graph, y_nodes = formater.transductive_pyg_graph_to_tree_graph(dataset)
    return graph, np.array(y_nodes)


def Planetoid_CITESEER(formater):
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
    graph, y_nodes = formater.transductive_pyg_graph_to_tree_graph(dataset)
    return graph, np.array(y_nodes)

def Planetoid_PUBMED(formater):
    dataset = Planetoid(root='data/Planetoid', name='PubMed')
    graph, y_nodes = formater.transductive_pyg_graph_to_tree_graph(dataset)
    return graph, np.array(y_nodes)


def GITHUB(formater):
    dataset = GitHub(root='data/GitHub')
    graph, y_nodes = formater.transductive_pyg_graph_to_tree_graph(dataset)
    return graph, np.array(y_nodes)


