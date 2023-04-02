import numpy as np
from treeg.node_treeg.node_level_treeg import NodeTreeG
from sklearn.metrics import mean_squared_error, accuracy_score
from starboost import BoostingClassifier, BoostingRegressor
from treeg.node_treeg.data_formetter_node_level import DataFormatter
from treeg.node_treeg.graph_data_node_level import SparseGraphData
import wandb
import threading
from data_utils import add_constant_one_feature

wandb_flag = False
n_estimators = 50
learning_rate = 0.1
attention_type_sample_probability = 0.5


def calc_metrics(model, X, y):
    y_preds = model.predict(X).flatten()
    l2 = mean_squared_error(y, y_preds)
    acc = accuracy_score(y, y_preds)
    return l2, acc


def train_valid_test_multiclass_paralel(graph, X_train, y_train, X_valid, X_test, max_attention_depth, max_graph_depth,
                                        attention_types, class_idx,
                                        train_results_arr, test_results_arr, valid_results_arr, classification=True):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=NodeTreeG(graph=graph,
                                 max_attention_depth=max_attention_depth,
                                 walk_lens=list(range(0, max_graph_depth + 1)),
                                 attention_types=attention_types,
                                 attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=NodeTreeG(graph=graph,
                                 max_attention_depth=max_attention_depth,
                                 walk_lens=list(range(0, max_graph_depth + 1)),
                                 attention_types=attention_types,
                                 attention_type_sample_probability=attention_type_sample_probability
                                 ),
        n_estimators=n_estimators,
        learning_rate=learning_rate)
    y = np.array(y_train)
    y = y.flatten()
    gbgta.fit(X_train, y)

    train_preds = gbgta.predict_proba(X_train)[:, 1]
    train_results_arr[:, class_idx] = train_preds

    valid_preds = gbgta.predict_proba(X_valid)[:, 1]
    valid_results_arr[:, class_idx] = valid_preds

    test_preds = gbgta.predict_proba(X_test)[:, 1]
    test_results_arr[:, class_idx] = test_preds


def parallel_multiclass_train_val_test(dataset, with_constant_one_feature=True,
                                       with_topological_features=False,
                                       balance_datasets=False):
    max_attention_depths = [1]
    max_walk_lengths = [0]
    attention_types = [1, 4]

    formatter = DataFormatter(SparseGraphData)
    graph, y_nodes = formatter.transductive_pyg_graph_to_tree_graph(dataset)
    X = np.arange(dataset.data.num_nodes)
    X_train, X_valid, X_test = X[dataset.data.train_mask], X[dataset.data.val_mask], X[dataset.data.test_mask]
    y_train, y_valid, y_test = y_nodes[dataset.data.train_mask], y_nodes[dataset.data.val_mask], y_nodes[
        dataset.data.test_mask]
    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_walk_lengths:
            print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_walk_length: ' + str(max_graph_depth))

            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset.name,
                               "attention_types": str(attention_types),
                               'attention_type_sample_probability': attention_type_sample_probability,
                               'num_of_features': graph.get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature,
                               'with_topological_features': with_topological_features,
                               'balance_datasets': balance_datasets,
                           })
                run_name = '%s_%d_%d' % (dataset.name, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name

            print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_walk_length: ' + str(max_graph_depth))

            num_of_classes = np.max([2, np.max(y_test) + 1])
            threads = []
            if with_constant_one_feature:
                add_constant_one_feature([graph])

            valid_preds_all_classes = np.zeros(shape=(len(y_valid), num_of_classes))
            test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
            train_preds_all_classes = np.zeros(shape=(len(y_train), num_of_classes))
            for class_i in range(num_of_classes):
                X_train_class = X_train
                X_valid_class = X_valid
                X_test_class = X_test
                y_train_class = (y_train == class_i).astype(int)

                thread = threading.Thread(target=train_valid_test_multiclass_paralel, args=(graph,
                                                                                            X_train_class,
                                                                                            y_train_class,
                                                                                            X_valid_class,
                                                                                            X_test_class,
                                                                                            max_attention_depth,
                                                                                            max_graph_depth,
                                                                                            attention_types,
                                                                                            class_i,
                                                                                            train_preds_all_classes,
                                                                                            test_preds_all_classes,
                                                                                            valid_preds_all_classes,
                                                                                            True))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
            acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)

            valid_one_vs_all_preds = np.argmax(valid_preds_all_classes, axis=1)
            acc_valid = (y_valid == valid_one_vs_all_preds).sum() / len(y_valid)

            train_one_vs_all_preds = np.argmax(train_preds_all_classes, axis=1)
            acc_train = (y_train == train_one_vs_all_preds).sum() / len(y_train)

            print('acc_test: ' + str(acc_test))
            print('acc_valid: ' + str(acc_valid))
            print('acc_train: ' + str(acc_train))

            if wandb_flag:
                wandb.log({'acc test': acc_test})
                wandb.log({'acc valid': acc_valid})
                wandb.log({'acc train': acc_train})
                wandb.finish()


def parallel_multiclass_train_val_test_ogb(dataset, with_constant_one_feature=True,
                                           with_topological_features=False,
                                           balance_datasets=False):
    max_attention_depths = [0, 1, 2]
    max_walk_lengths = [0, 1, 2]
    attention_types = [1, 4]

    formatter = DataFormatter(SparseGraphData)
    split_idx = dataset.get_idx_split()
    graph = formatter.fast_pyg_data_to_sparse_graph_data(dataset.data)
    y_nodes = dataset.data.y
    X = np.arange(dataset.data.num_nodes)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
    y_train, y_valid, y_test = y_nodes[train_idx], y_nodes[valid_idx], y_nodes[test_idx]

    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_walk_lengths:
            print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_walk_length: ' + str(max_graph_depth))

            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset.name,
                               "attention_types": str(attention_types),
                               'attention_type_sample_probability': attention_type_sample_probability,
                               'num_of_features': graph.get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature,
                               'with_topological_features': with_topological_features,
                               'balance_datasets': balance_datasets,
                           })
                run_name = '%s_%d_%d' % (dataset.name, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name

            print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_walk_length: ' + str(max_graph_depth))

            num_of_classes = np.max([2, np.max(y_test) + 1])
            threads = []
            if with_constant_one_feature:
                add_constant_one_feature([graph])

            valid_preds_all_classes = np.zeros(shape=(len(y_valid), num_of_classes))
            test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
            train_preds_all_classes = np.zeros(shape=(len(y_train), num_of_classes))
            for class_i in range(num_of_classes):
                X_train_class = X_train
                X_valid_class = X_valid
                X_test_class = X_test
                y_train_class = (y_train == class_i).astype(int)

                thread = threading.Thread(target=train_valid_test_multiclass_paralel, args=(graph,
                                                                                            X_train_class,
                                                                                            y_train_class,
                                                                                            X_valid_class,
                                                                                            X_test_class,
                                                                                            max_attention_depth,
                                                                                            max_graph_depth,
                                                                                            attention_types,
                                                                                            class_i,
                                                                                            train_preds_all_classes,
                                                                                            test_preds_all_classes,
                                                                                            valid_preds_all_classes,
                                                                                            True))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
            acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)

            valid_one_vs_all_preds = np.argmax(valid_preds_all_classes, axis=1)
            acc_valid = (y_valid == valid_one_vs_all_preds).sum() / len(y_valid)

            train_one_vs_all_preds = np.argmax(train_preds_all_classes, axis=1)
            acc_train = (y_train == train_one_vs_all_preds).sum() / len(y_train)

            print('acc_test: ' + str(acc_test))
            print('acc_valid: ' + str(acc_valid))
            print('acc_train: ' + str(acc_train))

            if wandb_flag:
                wandb.log({'acc test': acc_test})
                wandb.log({'acc valid': acc_valid})
                wandb.log({'acc train': acc_train})
                wandb.finish()
