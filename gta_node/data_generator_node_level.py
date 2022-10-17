from gta_node.graph_data_node_level import GraphData
import numpy as np
from networkx.generators.random_graphs import fast_gnp_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix


class Gnp_sign_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n, 2])
        features[:, 1] = np.random.normal(size=(n))
        y = (0.5 * np.sign(adj @ features[:, 1]) + 0.5).astype(int)
        y = np.array(y.reshape((n, -1)))
        g = GraphData(adj, features)
        return g, y


class Gnp_sign_red_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n, 3])
        features[:, 2] = np.random.choice([0, 1], size=(n))
        features[:, 1] = np.random.normal(size=(n))
        t = np.multiply(features[:, 1], features[:, 2])
        y = (0.5 * np.sign(adj @ t) + 0.5).astype(int)
        y = y.reshape((n, -1))
        g = GraphData(adj, features)
        return g, y


class Gnp_sign_red_blue_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n, 3])
        features[:, 2] = np.random.choice([-1, 1], size=(n))
        features[:, 1] = np.random.normal(size=(n))
        t = np.multiply(features[:, 1], features[:, 2])
        y = (0.5 * np.sign(adj @ t) + 0.5).astype(int)
        y = y.reshape((n, -1))
        g = GraphData(adj, features)
        return g, y
