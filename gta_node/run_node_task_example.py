import numpy as np
import datasets
from gta_node_level import GTANode
from starboost import BoostingClassifier
from gta_graph.data_formetter import DataFormatter
from gta_node.graph_data_node_level import GraphData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score


def print_trees(model):
    for idx, estimator in enumerate(model.estimators_):
        print('Tree ' + str(idx) + ':')
        estimator[0].print_tree()


def test(model, X, y):
    y_preds = model.predict(X).flatten()
    l2 = mean_squared_error(y, y_preds)
    auc = roc_auc_score(y, y_preds)
    acc = accuracy_score(y, y_preds)
    return l2, auc, acc


def run(attention_types=[1, 4]):
    n_estimators = 50
    learning_rate = 0.1
    max_number_of_leafs = 10
    max_attention_depth = 1
    max_graph_depth = 1
    walk_lens = list(range(0, max_graph_depth + 1))

    formatter = DataFormatter(GraphData)
    graph, y_nodes = datasets.Planetoid_CORA(formatter)
    n_nodes = graph.get_number_of_nodes()
    X = np.arange(n_nodes)
    graph.compute_walks(max_graph_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y_nodes, test_size=0.2, random_state=42)

    gbgta = BoostingClassifier(
        init_estimator=GTANode(
            max_number_of_leafs=max_number_of_leafs,
            max_attention_depth=max_attention_depth,
            walk_lens=walk_lens,
            attention_types=attention_types),
        base_estimator=GTANode(
            max_number_of_leafs=max_number_of_leafs,
            max_attention_depth=max_attention_depth,
            walk_lens=walk_lens,
            attention_types=attention_types),
        n_estimators=n_estimators,
        learning_rate=learning_rate)

    gbgta.fit(X_train, y_train)
    l2_test, auc_test, acc_test = test(gbgta, X_test, y_test)
    print("Test: l2 %5f accuracy %5f auc %5f" % (l2_test, acc_test, auc_test))

run()