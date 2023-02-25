import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from treeg.graph_treeg.graph_data_graph_level import GraphData, SparseGraphData


def adj_to_edges_list(adj_mat):
    G = nx.Graph(adj_mat)
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def graph_tree_list_to_pyg_data_list(graph_list, labels):
    graph_data_list = []
    for idx, graph in enumerate(graph_list):
        edge_index = adj_to_edges_list(graph.adj_mat)
        x = torch.from_numpy(graph.features).type(torch.float32)
        y = torch.from_numpy(np.array([labels[idx]])).type(torch.long)
        data = Data(edge_index=edge_index, x=x, y=y)
        graph_data_list.append(data)

    return graph_data_list


def pyg_data_to_tree_graph_data(pyg_graph):
    G = nx.Graph()
    G.add_nodes_from(np.arange(pyg_graph.num_nodes))
    num_nodes = G.number_of_nodes()
    list_of_edges = pyg_graph.edge_index.T.tolist()
    G.add_edges_from(list_of_edges)
    adj = nx.to_numpy_array(G)
    is_undirected = np.allclose(adj, adj.T, rtol=1e-05, atol=1e-08)
    if not is_undirected:
        adj = adj.T
    if pyg_graph.x is None:
        nodes_features = np.array([G.degree(i) for i in range(num_nodes)]) \
            .reshape(num_nodes, 1)
    else:
        nodes_features = np.array(pyg_graph.x.tolist())
    label = pyg_graph.y.tolist()[0]
    g = GraphData(adj, nodes_features, label)
    return g


def pyg_data_to_sparse_graph_data(pyg_graph):
    G = nx.Graph()
    G.add_nodes_from(np.arange(pyg_graph.num_nodes))
    num_nodes = G.number_of_nodes()
    list_of_edges = pyg_graph.edge_index.T.tolist()
    G.add_edges_from(list_of_edges)
    adj = nx.to_numpy_array(G)
    is_undirected = np.allclose(adj, adj.T, rtol=1e-05, atol=1e-08)
    if not is_undirected:
        adj = adj.T
    if pyg_graph.x is None:
        nodes_features = np.array([G.degree(i) for i in range(num_nodes)]) \
            .reshape(num_nodes, 1)
    else:
        nodes_features = np.array(pyg_graph.x.tolist())
    label = pyg_graph.y.tolist()[0]
    g = SparseGraphData(adj, nodes_features, label)
    return g


def pyg_data_list_to_tree_graph_data_list(pyg_data_list):
    tree_graph_data_list = []
    labels_list = []
    for idx, graph in enumerate(pyg_data_list):
        tree_graph_data = pyg_data_to_tree_graph_data(graph)
        tree_graph_data_list.append(tree_graph_data)
        labels_list.append(tree_graph_data.label)
    return tree_graph_data_list, labels_list


def transductive_pyg_graph_to_tree_graph(pyg_graph):
    tree_graph_data = pyg_data_to_tree_graph_data(pyg_graph.data)
    tree_graph_data.label = np.array(pyg_graph.data.y)
    return tree_graph_data, tree_graph_data.label


def pyg_data_list_to_sparse_graph_data_list(pyg_data_list):
    sparse_graph_data_list = []
    labels_list = []
    for idx, graph in enumerate(pyg_data_list):
        sparse_graph = pyg_data_to_sparse_graph_data(graph)
        graph_data = pyg_data_to_tree_graph_data(graph)
        if not (sparse_graph.sparse_adj.toarray() == graph_data.adj_mat).all():
            print("diff")
        sparse_graph_data_list.append(sparse_graph)
        labels_list.append(sparse_graph.label)
    return sparse_graph_data_list, labels_list
