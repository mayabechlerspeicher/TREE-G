import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from treeg.graph_treeg.graph_level_treeg import GraphTreeG
from starboost import BoostingClassifier, BoostingRegressor
from sklearn.metrics import roc_auc_score
import wandb
from data_utils import add_constant_one_feature
from treeg.graph_treeg.data_formetter_graph_level import DataFormatter
from treeg import graph_treeg as explainer_graph_level
from treeg.graph_treeg.aggregator_graph_level import graph_level_aggregators
from treeg.graph_treeg.graph_data_graph_level import GraphData, SparseGraphData
from sklearn.model_selection import KFold
import threading


def calc_metrics(model, X, y):
    y_preds = model.predict(X).flatten()
    l2 = mean_squared_error(y, y_preds)
    num_of_classes = len(np.unique(y))
    if num_of_classes > 1:
        auc = roc_auc_score(y, y_preds)
    else:
        auc = -1
    acc = accuracy_score(y, y_preds)
    return l2, auc, acc


def train(X_train, y_train, max_attention_depth, max_graph_depth, attention_types, classification=True, n_estimators=50,
          learning_rate=0.1, attention_type_sample_probability=0.25):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GraphTreeG(
            max_attention_depth=max_attention_depth,
            walk_lens=list(range(0, max_graph_depth + 1)),
            attention_types=attention_types,
            attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GraphTreeG(
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
    return gbgta


def train_test(X_train, y_train, X_test, y_test, max_attention_depth, max_graph_depth, attention_types,
               classification=True, use_wandb=False, n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GraphTreeG(max_attention_depth=max_attention_depth,
                                  walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                  attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GraphTreeG(max_attention_depth=max_attention_depth,
                                  walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                  attention_type_sample_probability=attention_type_sample_probability),
        n_estimators=n_estimators,
        learning_rate=learning_rate)

    y = np.array(y_train)
    y = y.flatten()
    gbgta.fit(X_train, y)

    L2_train, auc_train, acc_train = calc_metrics(gbgta, X_train, y_train)
    print("Train: l2 %5f accuracy %5f auc %5f" % (L2_train, acc_train, auc_train))
    L2_test, auc_test, acc_test = calc_metrics(gbgta, X_test, y_test)
    print("Test: l2 %5f accuracy %5f auc %5f" % (L2_test, acc_test, auc_test))

    if use_wandb:
        wandb.log({"L2_train": L2_train})
        wandb.log({"auc_train": auc_train})
        wandb.log({"acc_train": acc_train})

        wandb.log({"L2_test": L2_test})
        wandb.log({"auc_test": auc_test})
        wandb.log({"acc_test": acc_test})

    stats_dict = explainer_graph_level.sum_stats_from_all_trees(gbgta)

    return gbgta, stats_dict, L2_train, auc_train, acc_train, L2_test, auc_test, acc_test


def train_val_test(X_train, y_train, X_val, y_val, X_test, y_test, max_attention_depth, max_graph_depth,
                   attention_types,
                   classification=True, use_wandb=False, n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GraphTreeG(max_attention_depth=max_attention_depth,
                                  walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                  attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GraphTreeG(max_attention_depth=max_attention_depth,
                                  walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                  attention_type_sample_probability=attention_type_sample_probability),
        n_estimators=n_estimators,
        learning_rate=learning_rate)

    y = np.array(y_train)
    y = y.flatten()

    X_train = X_train.flatten().tolist()
    X_val = X_val.flatten().tolist()
    X_test = X_test.flatten().tolist()
    gbgta.fit(X_train, y)

    L2_train, auc_train, acc_train = calc_metrics(gbgta, X_train, y_train)
    print("Train: l2 %5f accuracy %5f auc %5f" % (L2_train, acc_train, auc_train))
    L2_val, auc_val, acc_val = calc_metrics(gbgta, X_val, y_val)
    print("Val: l2 %5f accuracy %5f auc %5f" % (L2_val, acc_val, auc_val))
    L2_test, auc_test, acc_test = calc_metrics(gbgta, X_test, y_test)
    print("Test: l2 %5f accuracy %5f auc %5f" % (L2_test, acc_test, auc_test))

    if use_wandb:
        wandb.log({"L2_train": L2_train})
        wandb.log({"auc_train": auc_train})
        wandb.log({"acc_train": acc_train})

        wandb.log({"L2_test": L2_test})
        wandb.log({"auc_test": auc_test})
        wandb.log({"acc_test": acc_test})

        wandb.log({"L2_val": L2_val})
        wandb.log({"auc_val": auc_val})
        wandb.log({"acc_val": acc_val})

    return gbgta


def train_test_multiclass_parallel(X_train, y_train, X_test, max_attention_depth, max_graph_depth, attention_types,
                                   class_idx,
                                   results_arr, classification=True,
                                   n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GraphTreeG(
            max_attention_depth=max_attention_depth,
            walk_lens=list(range(0, max_graph_depth + 1)),
            attention_types=attention_types,
            attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GraphTreeG(
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

    test_preds = gbgta.predict_proba(X_test)[:, 1]
    results_arr[:, class_idx] = test_preds


def cross_validation(dataset, with_constant_one_feature=True, max_attention_depth=3,
                     max_walk_length=3, use_wandb=False,
                     n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    num_folds = 10
    attention_types = [1, 2, 3, 4]

    formatter = DataFormatter(GraphData)
    X, y = formatter.pyg_data_list_to_tree_graph_data_list(dataset)
    X, y = np.array(X), np.array(y)

    kf = KFold(n_splits=num_folds)
    print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
        max_attention_depth) + ' max_walk_length: ' + str(max_walk_length))
    all_auc_train = []
    all_auc_test = []
    all_acc_train = []
    all_acc_test = []

    if use_wandb:
        wandb.init(project='TREE-G', reinit=True, entity='your entity',
                   config={
                       "max_attention": max_attention_depth,
                       "max_graph_depth": max_walk_length,
                       "num_folds": num_folds,
                       "n_estimators": n_estimators,
                       "learning_rate": learning_rate,
                       "dataset": dataset.name,
                       "attention_types": str(attention_types),
                       'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                       'attention_type_sample_probability': attention_type_sample_probability,
                   })
        run_name = '%s_%d_%d' % (dataset.name, max_attention_depth, max_walk_length)
        wandb.run.name = run_name
    inferences_mean_time = []
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
            max_attention_depth) + ' max_walk_length: ' + str(max_walk_length) + ' fold: ' + str(idx))
        X_train, y_train = list(X[train_index]), list(y[train_index])
        X_test, y_test = list(X[test_index]), list(y[test_index])
        if with_constant_one_feature:
            add_constant_one_feature(X_train)
            add_constant_one_feature(X_test)

        if use_wandb:
            wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                       'num_train_samples': len(X_train),
                       'with_constant_one_feature': with_constant_one_feature})

        if X_train[0].adj_powers is not list:
            for graph in X_train:
                graph.compute_walks(max_walk_length)
            for graph in X_test:
                graph.compute_walks(max_walk_length)

        gbgta, stats_dict, L2_train, auc_train, acc_train, L2_test, auc_test, acc_test, total_inference_time = \
            train_test(X_train, y_train, X_test, y_test, max_attention_depth, max_walk_length, attention_types,
                       classification=True)
        total_inference_time_minutes = total_inference_time.total_seconds() / 60
        inferences_mean_time.append(total_inference_time_minutes)
        all_auc_train.append(auc_train)
        all_auc_test.append(auc_test)
        all_acc_train.append(acc_train)
        all_acc_test.append(acc_test)

        print('fold: ' + str(idx) + ' auc_train: ' + str(auc_train) + ' auc_test: ' + str(
            auc_test) + ' acc_train: ' + str(acc_train) + ' acc_test: ' + str(acc_test))
        print('inference time ' + str(total_inference_time_minutes))

        if use_wandb:
            wandb.log(stats_dict)
        if use_wandb:
            wandb.log({'fold %d auc train' % idx: auc_train, 'fold %d auc test' % idx: auc_test,
                       'fold %d acc train' % idx: acc_train, 'fold %d acc test' % idx: acc_test,
                       'fold %d inference time' % idx: total_inference_time_minutes})

    print('auc-std-train: ' + str(np.std(all_auc_train)))
    print('auc-std-test: ' + str(np.std(all_auc_test)))
    print('acc-std-train: ' + str(np.std(all_acc_train)))
    print('acc-std-test: ' + str(np.std(all_acc_test)))
    print('auc-mean-train: ' + str(np.mean(all_auc_train)))
    print('auc-mean-test: ' + str(np.mean(all_auc_test)))
    print('acc-mean-train: ' + str(np.mean(all_acc_train)))
    print('acc-mean-test: ' + str(np.mean(all_acc_test)))

    if use_wandb:
        wandb.log({"auc-std train": np.std(all_auc_train)})
        wandb.log({"avg auc train": np.mean(all_auc_train)})
        wandb.log({"auc-std test": np.std(all_auc_test)})
        wandb.log({"avg auc test": np.mean(all_auc_test)})
        wandb.log({"acc-std train": np.std(all_acc_train)})
        wandb.log({"avg acc train": np.mean(all_acc_train)})
        wandb.log({"acc-std test": np.std(all_acc_test)})
        wandb.log({"avg acc test": np.mean(all_acc_test)})
        wandb.log({"avg inference time minutes": np.mean(inferences_mean_time)})
        wandb.finish()


def parallel_multiclass_cross_validation(dataset, with_constant_one_feature=True, max_attention_depth=3,
                                         max_walk_length=3, use_wandb=False,
                                         n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    num_folds = 2
    attention_types = [1, 2, 3, 4]

    formatter = DataFormatter(GraphData)
    X, y = formatter.pyg_data_list_to_tree_graph_data_list(dataset)
    X, y = np.array(X), np.array(y)

    kf = KFold(n_splits=num_folds)
    print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
        max_attention_depth) + ' max_walk_length: ' + str(max_walk_length))
    all_acc_test = []
    if use_wandb:
        wandb.init(project='TREE-G', reinit=True, entity='your entity',
                   config={
                       "max_attention": max_attention_depth,
                       "max_graph_depth": max_walk_length,
                       "num_folds": num_folds,
                       "n_estimators": n_estimators,
                       "learning_rate": learning_rate,
                       "dataset": dataset.__name__,
                       "attention_types": str(attention_types),
                       'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                       'attention_type_sample_probability': attention_type_sample_probability,
                   })
        run_name = '%s_%d_%d' % (dataset.name, max_attention_depth, max_walk_length)
        wandb.run.name = run_name

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
            max_attention_depth) + ' max_walk_len: ' + str(max_walk_length) + ' fold: ' + str(idx))
        X_train, y_train = list(X[train_index]), list(y[train_index])
        X_test, y_test = list(X[test_index]), list(y[test_index])
        num_of_classes = np.max([2, np.max(y_test) + 1])
        threads = []
        if with_constant_one_feature:
            add_constant_one_feature(X_train)
            add_constant_one_feature(X_test)

        if use_wandb:
            wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                       'num_train_samples': len(X_train),
                       'with_constant_one_feature': with_constant_one_feature})

        if X_train[0].adj_powers is not list:
            for graph in X_train:
                graph.compute_walks(max_walk_length)
            for graph in X_test:
                graph.compute_walks(max_walk_length)
        test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
        for class_i in range(num_of_classes):
            X_train_class = X_train
            X_test_class = X_test
            y_train_class = (np.array(y_train) == class_i).astype(int)

            thread = threading.Thread(target=train_test_multiclass_parallel, args=(
                X_train_class, y_train_class, X_test_class, max_attention_depth, max_walk_length,
                attention_types, class_i, test_preds_all_classes, True))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
        acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)
        all_acc_test.append(acc_test)

        print('fold: ' + str(idx) + ' acc_test: ' + str(acc_test))

        if use_wandb:
            wandb.log({'fold %d acc test' % idx: acc_test})

    print('Test accuracy std: ' + str(np.std(all_acc_test)))
    print('Mean test accuracy: ' + str(np.mean(all_acc_test)))
    if use_wandb:
        wandb.log({"Test accuracy std": np.std(all_acc_test)})
        wandb.log({"Mean test accuracy": np.mean(all_acc_test)})
        wandb.finish()


def multiclass_cross_validation(dataset, with_constant_one_feature=True, max_attention_depth=3,
                                max_walk_length=3, use_wandb=False,
                                n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    num_folds = 10
    attention_types = [1, 2, 3, 4]

    formatter = DataFormatter(GraphData)
    X, y = formatter.pyg_data_list_to_tree_graph_data_list(dataset)
    X, y = np.array(X), np.array(y)

    kf = KFold(n_splits=num_folds)

    print('Running dataset: ' + dataset.anme + ' max_attention_depth: ' + str(
        max_attention_depth) + ' max_walk_len: ' + str(max_walk_length))
    all_acc_test = []
    if use_wandb:
        wandb.init(project='TREE-G', reinit=True, entity='your entity',
                   config={
                       "max_attention": max_attention_depth,
                       "max_graph_depth": max_walk_length,
                       "num_folds": num_folds,
                       "n_estimators": n_estimators,
                       "learning_rate": learning_rate,
                       "dataset": dataset.anme,
                       "attention_types": str(attention_types),
                       'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                       'attention_type_sample_probability': attention_type_sample_probability,
                   })
        run_name = '%s_%d_%d' % (dataset.anme, max_attention_depth, max_walk_length)
        wandb.run.name = run_name

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        print('Running dataset: ' + dataset.anme + ' max_attention_depth: ' + str(
            max_attention_depth) + ' max_walk_len: ' + str(max_walk_length) + ' fold: ' + str(idx))
        X_train, y_train = list(X[train_index]), list(y[train_index])
        X_test, y_test = list(X[test_index]), list(y[test_index])
        num_of_classes = np.max([np.max(y_test), np.max(y_train)]) + 1
        if with_constant_one_feature:
            add_constant_one_feature(X_train)
            add_constant_one_feature(X_test)

        if use_wandb:
            wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                       'num_train_samples': len(X_train),
                       'with_constant_one_feature': with_constant_one_feature})

        if X_train[0].adj_powers is not list:
            for graph in X_train:
                graph.compute_walks(max_walk_length)
            for graph in X_test:
                graph.compute_walks(max_walk_length)
        test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
        for class_i in range(num_of_classes):
            X_train_class = X_train
            y_train_class = (y_train == class_i).astype(int)
            gbgta = train(X_train=X_train_class, y_train=y_train_class, max_attention_depth=max_attention_depth,
                          max_graph_depth=max_walk_length, attention_types=attention_types, classification=True)
            test_preds = gbgta.predict_proba(X_test)[:, 1]
            test_preds_all_classes[:, class_i] = test_preds

        test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
        acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)
        all_acc_test.append(acc_test)

        print('fold: ' + str(idx) + ' acc_test: ' + str(acc_test))

        if use_wandb:
            wandb.log({'fold %d acc test' % idx: acc_test})

    print('acc-std-test: ' + str(np.std(all_acc_test)))
    print('acc-mean-test: ' + str(np.mean(all_acc_test)))
    if use_wandb:
        wandb.log({"acc-std test": np.std(all_acc_test)})
        wandb.log({"avg acc test": np.mean(all_acc_test)})
        wandb.finish()


def multiclass_train_val_test_ogb_splits(dataset, with_constant_one_feature=True, max_attention_depth=3,
                                         max_walk_length=3, use_wandb=False,
                                         n_estimators=50, learning_rate=0.1, attention_type_sample_probability=0.25):
    attention_types = [1, 2, 3, 4]

    formatter = DataFormatter(SparseGraphData)
    split_idx = dataset.get_idx_split()
    train_graphs_list = dataset[split_idx["train"]]
    valid_graphs_list = dataset[split_idx["valid"]]
    test_graphs_list = dataset[split_idx["test"]]
    X_train, y_train = formatter.pyg_data_list_to_tree_graph_data_list(train_graphs_list)
    X_val, y_val = formatter.pyg_data_list_to_tree_graph_data_list(valid_graphs_list)
    X_test, y_test = formatter.pyg_data_list_to_tree_graph_data_list(test_graphs_list)

    print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
        max_attention_depth) + ' max_walk_len: ' + str(max_walk_length))
    all_acc_test = []
    all_acc_val = []
    if use_wandb:
        wandb.init(project='TREE-G', reinit=True, entity='your entity',
                   config={
                       "max_attention": max_attention_depth,
                       "max_graph_depth": max_walk_length,
                       "n_estimators": n_estimators,
                       "learning_rate": learning_rate,
                       "dataset": dataset.name,
                       "attention_types": str(attention_types),
                       'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                       'attention_type_sample_probability': attention_type_sample_probability,
                   })
        run_name = '%s_%d_%d' % (dataset.name, max_attention_depth, max_walk_length)
        wandb.run.name = run_name

        print('Running dataset: ' + dataset.name + ' max_attention_depth: ' + str(
            max_attention_depth) + ' max_walk_length: ' + str(max_walk_length))
        num_of_classes = np.max([np.max(y_test), np.max(y_train)]) + 1
        if with_constant_one_feature:
            add_constant_one_feature(X_train)
            add_constant_one_feature(X_test)

        if use_wandb:
            wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                       'num_train_samples': len(X_train),
                       'with_constant_one_feature': with_constant_one_feature})

        if X_train[0].adj_powers is not list:
            for graph in X_train:
                graph.compute_walks(max_walk_length)
            for graph in X_test:
                graph.compute_walks(max_walk_length)
        test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
        val_preds_all_classes = np.zeros(shape=(len(y_val), num_of_classes))
        for class_i in range(num_of_classes):
            X_train_class = X_train
            y_train_class = (y_train == class_i).astype(int)
            gbgta = train(X_train=X_train_class, y_train=y_train_class,
                          max_attention_depth=max_attention_depth,
                          max_graph_depth=max_walk_length, attention_types=attention_types,
                          classification=True)
            val_preds = gbgta.predict_proba(X_val)[:, 1]
            val_preds_all_classes[:, class_i] = val_preds
            test_preds = gbgta.predict_proba(X_test)[:, 1]
            test_preds_all_classes[:, class_i] = test_preds

        val_one_vs_all_preds = np.argmax(val_preds_all_classes, axis=1)
        test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
        acc_val = (y_val == val_one_vs_all_preds).sum() / len(y_test)
        acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)
        all_acc_val.append(acc_val)
        all_acc_test.append(acc_test)

        print('acc_test: ' + str(acc_test) + 'acc_val: ' + str(acc_val))

        if wandb:
            wandb.log({'acc test': acc_test,
                       'val test': acc_val})

    print('acc-std-test: ' + str(np.std(all_acc_test)))
    print('acc-mean-test: ' + str(np.mean(all_acc_test)))
    print('acc-std-val: ' + str(np.std(all_acc_val)))
    print('acc-mean-val: ' + str(np.mean(all_acc_val)))
    if wandb:
        wandb.log({"acc-std test": np.std(all_acc_test)})
        wandb.log({"avg acc test": np.mean(all_acc_test)})
        wandb.log({"acc-std val": np.std(all_acc_val)})
        wandb.log({"avg acc val": np.mean(all_acc_val)})
        wandb.finish()
