import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from scipy import sparse

import torch_geometric

class DataFormatter:
    def __init__(self, graph_data_object):
        self.graph_data_object = graph_data_object

    @staticmethod
    def adj_to_edges_list(adj_mat):
        G = nx.Graph(adj_mat)
        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def graph_tree_list_to_pyg_data_list(self, graph_list, labels):
        graph_data_list = []
        for idx, graph in enumerate(graph_list):
            edge_index = self.adj_to_edges_list(graph.adj_mat)
            x = torch.from_numpy(graph.features).type(torch.float32)
            y = torch.from_numpy(np.array([labels[idx]])).type(torch.long)
            data = Data(edge_index=edge_index, x=x, y=y)
            graph_data_list.append(data)

        return graph_data_list

    def transductive_pyg_graph_to_tree_graph(self, pyg_graph):
        tree_graph_data = self.pyg_data_to_tree_graph_data(pyg_graph.data)
        tree_graph_data.label = np.array(pyg_graph.data.y)
        return tree_graph_data, tree_graph_data.label

    def pyg_data_list_to_tree_graph_data_list(self, pyg_data_list):
        tree_graph_data_list = []
        labels_list = []
        for idx, graph in enumerate(pyg_data_list):
            tree_graph_data = self.pyg_data_to_tree_graph_data(graph)
            tree_graph_data_list.append(tree_graph_data)
            labels_list.append(tree_graph_data.label)
        return tree_graph_data_list, labels_list

    def pyg_data_list_to_sparse_graph_data_list(self, pyg_data_list):
        sparse_graph_data_list = []
        labels_list = []
        for idx, graph in enumerate(pyg_data_list):
            sparse_graph = self.fast_pyg_data_to_sparse_graph_data(graph)
            sparse_graph_data_list.append(sparse_graph)
            labels_list.append(sparse_graph.label)
        return sparse_graph_data_list, labels_list

    def pyg_data_to_tree_graph_data(self, pyg_graph):
        G = torch_geometric.utils.to_networkx(pyg_graph)
        num_nodes = G.number_of_nodes()
        adj = nx.adjacency_matrix(G).toarray()
        is_undirected = np.allclose(adj, adj.T, rtol=1e-05, atol=1e-08)
        if not is_undirected:
            adj = adj.T
        if pyg_graph.x is None:
            nodes_features = np.array([G.degree(i) for i in range(num_nodes)]) \
                .reshape(num_nodes, 1)
        if pyg_graph.x is not None:
            nodes_features = np.array(pyg_graph.x.tolist())
        label = pyg_graph.y.tolist()[0]
        g = self.graph_data_object(adj, nodes_features, label)
        return g

    def fast_pyg_data_to_sparse_graph_data(self, pyg_graph):
        num_nodes = pyg_graph.num_nodes
        row = pyg_graph.edge_index[0].numpy()
        col = pyg_graph.edge_index[1].numpy()
        data = np.ones(shape=(pyg_graph.num_edges,))
        edge_index = sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        is_undirrected = (np.abs(edge_index - edge_index.T) > 1e-10).nnz == 0
        if not is_undirrected:
            edge_index = edge_index.T
        if pyg_graph.x is None:
            nodes_features = np.ones(shape=(num_nodes, 1))
        if pyg_graph.x is not None:
            nodes_features = pyg_graph.x.numpy()
        label = pyg_graph.y.numpy()
        g = self.graph_data_object(edge_index, nodes_features, label)
        if not(g.sparse_adj.shape[0] == g.sparse_features.shape[0]):
            print("problem")
        return g

    def pyg_data_to_sparse_graph_data(self, pyg_graph):
        G = nx.Graph()
        G.add_nodes_from(np.arange(pyg_graph.num_nodes))
        num_nodes = G.number_of_nodes()
        list_of_edges = pyg_graph.edge_index.T.tolist()
        G.add_edges_from(list_of_edges)
        adj = nx.to_numpy_array(G)
        if pyg_graph.x is None:
            nodes_features = np.array([G.degree(i) for i in range(num_nodes)]) \
                .reshape(num_nodes, 1)
        if pyg_graph.x is not None:
            nodes_features = np.array(pyg_graph.x.tolist())
        label = pyg_graph.y.tolist()[0]
        g = self.graph_data_object(adj, nodes_features, label)
        return g
