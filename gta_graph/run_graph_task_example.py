import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from gta_graph.gta_graph_level import GTAGraph
from starboost import BoostingClassifier, BoostingRegressor
from sklearn.metrics import roc_auc_score
import wandb
from data_formatter import DataFormatter
import gta_graph.explainer_graph_level as explainer_graph_level
from gta_graph.aggregator_graph_level import graph_level_aggregators
import data_utils
from graph_data_graph_level import GraphData
import threading

wandb_flag = False
n_estimators = 50
learning_rate = 0.1
attention_type_sample_probability = 0.25


def print_trees(model):
    for idx, estimator in enumerate(model.estimators_):
        print('Tree ' + str(idx) + ':')
        estimator[0].print_tree()


def add_constant_one_feature(Xs):
    for x in Xs:
        ones_vec = np.ones(shape=(x.get_number_of_nodes(), 1))
        x.features = np.append(x.features, ones_vec, axis=1)


def test(model, X, y):
    y_preds = model.predict(X).flatten()
    l2 = mean_squared_error(y, y_preds)
    num_of_classes = len(np.unique(y))
    if num_of_classes > 1:
        auc = roc_auc_score(y, y_preds)
    else:
        auc = -1
    acc = accuracy_score(y, y_preds)
    return l2, auc, acc


def train_test(X_train, y_train, X_test, y_test, max_attention_depth, max_graph_depth, attention_types,
               classification=True):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GTAGraph(max_attention_depth=max_attention_depth,
                                walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GTAGraph(max_attention_depth=max_attention_depth,
                                walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                attention_type_sample_probability=attention_type_sample_probability),
        n_estimators=n_estimators,
        learning_rate=learning_rate)

    y = np.array(y_train)
    y = y.flatten()
    gbgta.fit(X_train, y)

    L2_train, auc_train, acc_train = test(gbgta, X_train, y_train)
    print("Train: l2 %5f accuracy %5f auc %5f" % (L2_train, acc_train, auc_train))
    L2_test, auc_test, acc_test = test(gbgta, X_test, y_test)
    print("Test: l2 %5f accuracy %5f auc %5f" % (L2_test, acc_test, auc_test))

    if wandb_flag:
        wandb.log({"L2_train": L2_train})
        wandb.log({"auc_train": auc_train})
        wandb.log({"acc_train": acc_train})

        wandb.log({"L2_test": L2_test})
        wandb.log({"auc_test": auc_test})
        wandb.log({"acc_test": acc_test})

    stats_dict = explainer_graph_level.sum_stats_from_all_trees(gbgta)

    return gbgta, stats_dict, L2_train, auc_train, acc_train, L2_test, auc_test, acc_test


def train(X_train, y_train, max_attention_depth, max_graph_depth, attention_types, classification=True):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GTAGraph(
            max_attention_depth=max_attention_depth,
            walk_lens=list(range(0, max_graph_depth + 1)),
            attention_types=attention_types,
            attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GTAGraph(
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


def train_val_test(X_train, y_train, X_val, y_val, X_test, y_test, max_attention_depth, max_graph_depth,
                   attention_types,
                   classification=True):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GTAGraph(max_attention_depth=max_attention_depth,
                                walk_lens=list(range(0, max_graph_depth + 1)), attention_types=attention_types,
                                attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GTAGraph(max_attention_depth=max_attention_depth,
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

    L2_train, auc_train, acc_train = test(gbgta, X_train, y_train)
    print("Train: l2 %5f accuracy %5f auc %5f" % (L2_train, acc_train, auc_train))
    L2_val, auc_val, acc_val = test(gbgta, X_val, y_val)
    print("Val: l2 %5f accuracy %5f auc %5f" % (L2_val, acc_val, auc_val))
    L2_test, auc_test, acc_test = test(gbgta, X_test, y_test)
    print("Test: l2 %5f accuracy %5f auc %5f" % (L2_test, acc_test, auc_test))

    if wandb_flag:
        wandb.log({"L2_train": L2_train})
        wandb.log({"auc_train": auc_train})
        wandb.log({"acc_train": acc_train})

        wandb.log({"L2_test": L2_test})
        wandb.log({"auc_test": auc_test})
        wandb.log({"acc_test": acc_test})

        wandb.log({"L2_val": L2_val})
        wandb.log({"auc_val": auc_val})
        wandb.log({"acc_val": acc_val})

    stats_dict = explainer_graph_level.sum_stats_from_all_trees(gbgta)

    return gbgta, stats_dict


def train_multiclass_paralel(X_train, y_train, X_test, max_attention_depth, max_graph_depth, attention_types, class_idx,
                             results_arr, classification=True):
    boosting_model = BoostingClassifier if classification else BoostingRegressor
    gbgta = boosting_model(
        init_estimator=GTAGraph(
            max_attention_depth=max_attention_depth,
            walk_lens=list(range(0, max_graph_depth + 1)),
            attention_types=attention_types,
            attention_type_sample_probability=attention_type_sample_probability),
        base_estimator=GTAGraph(
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


def run_cross_val_fixed_splits(dataset_name, with_constant_one_feature=True):
    num_folds = 10
    max_attention_depths = [2]
    max_graph_depths = [2]
    attention_types = [1, 2, 3, 4]

    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_graph_depths:
            print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth))
            all_auc_train = []
            all_auc_test = []
            all_acc_train = []
            all_acc_test = []

            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           # settings=wandb.Settings(start_method='thread'),
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "num_folds": num_folds,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset_name,
                               "attention_types": str(attention_types),
                               'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                               'attention_type_sample_probability': attention_type_sample_probability,
                           })
                run_name = '%s_%d_%d' % (dataset_name, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name
            inferences_mean_time = []
            for idx in range(1, num_folds + 1):
                print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                    max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth) + ' fold: ' + str(idx))
                X_train, y_train = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=True)
                X_test, y_test = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=False)
                if with_constant_one_feature:
                    add_constant_one_feature(X_train)
                    add_constant_one_feature(X_test)

                if wandb_flag:
                    wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature})

                if X_train[0].adj_powers is not list:
                    for graph in X_train:
                        graph.compute_walks(max_graph_depth)
                    for graph in X_test:
                        graph.compute_walks(max_graph_depth)

                gbgta, stats_dict, L2_train, auc_train, acc_train, L2_test, auc_test, acc_test, total_inference_time = \
                    train_test(X_train, y_train, X_test, y_test, max_attention_depth, max_graph_depth, attention_types,
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

                if wandb_flag:
                    wandb.log(stats_dict)
                if wandb_flag:
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

            if wandb_flag:
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


def run_paralel_multiclass_cross_val_fixed_splits(dataset_name, with_constant_one_feature=True):
    num_folds = 10
    max_attention_depths = [0, 1, 2]
    max_graph_depths = [0, 1, 2]
    attention_types = [1, 2, 3, 4]

    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_graph_depths:
            print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth))
            all_acc_test = []
            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "num_folds": num_folds,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset_name,
                               "attention_types": str(attention_types),
                               'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                               'attention_type_sample_probability': attention_type_sample_probability,
                           })
                run_name = '%s_%d_%d' % (dataset_name, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name

            for idx in range(1, num_folds + 1):
                print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                    max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth) + ' fold: ' + str(idx))
                X_train, y_train = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=True)
                X_test, y_test = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=False)
                num_of_classes = np.max([2, np.max(y_test) + 1])
                threads = []
                if with_constant_one_feature:
                    add_constant_one_feature(X_train)
                    add_constant_one_feature(X_test)

                if wandb_flag:
                    wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature})

                if X_train[0].adj_powers is not list:
                    for graph in X_train:
                        graph.compute_walks(max_graph_depth)
                    for graph in X_test:
                        graph.compute_walks(max_graph_depth)
                test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
                for class_i in range(num_of_classes):
                    X_train_class = X_train
                    X_test_class = X_test
                    y_train_class = (np.array(y_train) == class_i).astype(int)

                    thread = threading.Thread(target=train_multiclass_paralel, args=(
                        X_train_class, y_train_class, X_test_class, max_attention_depth, max_graph_depth,
                        attention_types, class_i, test_preds_all_classes, True))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()
                test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
                acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)
                all_acc_test.append(acc_test)

                print('fold: ' + str(idx) + ' acc_test: ' + str(acc_test))

                if wandb_flag:
                    wandb.log({'fold %d acc test' % idx: acc_test})

            print('acc-std-test: ' + str(np.std(all_acc_test)))
            print('acc-mean-test: ' + str(np.mean(all_acc_test)))
            if wandb_flag:
                wandb.log({"acc-std test": np.std(all_acc_test)})
                wandb.log({"avg acc test": np.mean(all_acc_test)})
                wandb.finish()


def run_multiclass_cross_val_fixed_splits(dataset_name, with_constant_one_feature=True):
    num_folds = 10
    max_attention_depths = [0, 1, 2]
    max_graph_depths = [0, 1, 2]
    attention_types = [1, 2, 3, 4]

    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_graph_depths:
            print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth))
            all_acc_test = []
            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "num_folds": num_folds,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset_name,
                               "attention_types": str(attention_types),
                               'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                               'attention_type_sample_probability': attention_type_sample_probability,
                           })
                run_name = '%s_%d_%d' % (dataset_name, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name

            for idx in range(1, num_folds + 1):
                print('running dataset: ' + dataset_name + ' max_attention_depth: ' + str(
                    max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth) + ' fold: ' + str(idx))
                X_train, y_train = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=True)
                X_test, y_test = data_utils.load_split_processed_datasets(dataset_name, fold_idx=idx, is_train=False)
                num_of_classes = np.max([np.max(y_test), np.max(y_train)]) + 1
                if with_constant_one_feature:
                    add_constant_one_feature(X_train)
                    add_constant_one_feature(X_test)

                if wandb_flag:
                    wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature})

                if X_train[0].adj_powers is not list:
                    for graph in X_train:
                        graph.compute_walks(max_graph_depth)
                    for graph in X_test:
                        graph.compute_walks(max_graph_depth)
                test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
                for class_i in range(num_of_classes):
                    X_train_class = X_train
                    y_train_class = (y_train == class_i).astype(int)
                    gbgta = train(X_train=X_train_class, y_train=y_train_class, max_attention_depth=max_attention_depth,
                                  max_graph_depth=max_graph_depth, attention_types=attention_types, classification=True)
                    test_preds = gbgta.predict_proba(X_test)[:, 1]
                    test_preds_all_classes[:, class_i] = test_preds

                test_one_vs_all_preds = np.argmax(test_preds_all_classes, axis=1)
                acc_test = (y_test == test_one_vs_all_preds).sum() / len(y_test)
                all_acc_test.append(acc_test)

                print('fold: ' + str(idx) + ' acc_test: ' + str(acc_test))

                if wandb_flag:
                    wandb.log({'fold %d acc test' % idx: acc_test})

            print('acc-std-test: ' + str(np.std(all_acc_test)))
            print('acc-mean-test: ' + str(np.mean(all_acc_test)))
            if wandb_flag:
                wandb.log({"acc-std test": np.std(all_acc_test)})
                wandb.log({"avg acc test": np.mean(all_acc_test)})
                wandb.finish()


def run_multiclass_ogb_splits(dataset, with_constant_one_feature=True):
    max_attention_depths = [0, 1, 2]
    max_graph_depths = [0, 1, 2]
    attention_types = [1, 2, 3, 4]
    formatter = DataFormatter(GraphData)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset(formatter)
    for max_attention_depth in max_attention_depths:
        for max_graph_depth in max_graph_depths:
            print('running dataset: ' + dataset.__name__ + ' max_attention_depth: ' + str(
                max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth))
            all_acc_test = []
            all_acc_val = []
            if wandb_flag:
                wandb.init(project='GTA_experiments', reinit=True, entity='your entity',
                           config={
                               "max_attention": max_attention_depth,
                               "max_graph_depth": max_graph_depth,
                               "n_estimators": n_estimators,
                               "learning_rate": learning_rate,
                               "dataset": dataset.__name__,
                               "attention_types": str(attention_types),
                               'graph_level_aggregators': str([agg.name for agg in graph_level_aggregators]),
                               'attention_type_sample_probability': attention_type_sample_probability,
                           })
                run_name = '%s_%d_%d' % (dataset.__name__, max_attention_depth, max_graph_depth)
                wandb.run.name = run_name

                print('running dataset: ' + dataset.__name__ + ' max_attention_depth: ' + str(
                    max_attention_depth) + ' max_graph_depth: ' + str(max_graph_depth))
                num_of_classes = np.max([np.max(y_test), np.max(y_train)]) + 1
                if with_constant_one_feature:
                    add_constant_one_feature(X_train)
                    add_constant_one_feature(X_test)

                if wandb_flag:
                    wandb.log({'num_of_features': X_train[0].get_number_of_features(),
                               'num_train_samples': len(X_train),
                               'with_constant_one_feature': with_constant_one_feature})

                if X_train[0].adj_powers is not list:
                    for graph in X_train:
                        graph.compute_walks(max_graph_depth)
                    for graph in X_test:
                        graph.compute_walks(max_graph_depth)
                test_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
                val_preds_all_classes = np.zeros(shape=(len(y_test), num_of_classes))
                for class_i in range(num_of_classes):
                    X_train_class = X_train
                    y_train_class = (y_train == class_i).astype(int)
                    gbgta = train(X_train=X_train_class, y_train=y_train_class,
                                  max_attention_depth=max_attention_depth,
                                  max_graph_depth=max_graph_depth, attention_types=attention_types,
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

                if wandb_flag:
                    wandb.log({'acc test': acc_test,
                               'val test': acc_val})

            print('acc-std-test: ' + str(np.std(all_acc_test)))
            print('acc-mean-test: ' + str(np.mean(all_acc_test)))
            print('acc-std-val: ' + str(np.std(all_acc_val)))
            print('acc-mean-val: ' + str(np.mean(all_acc_val)))
            if wandb_flag:
                wandb.log({"acc-std test": np.std(all_acc_test)})
                wandb.log({"avg acc test": np.mean(all_acc_test)})
                wandb.log({"acc-std val": np.std(all_acc_val)})
                wandb.log({"avg acc val": np.mean(all_acc_val)})
                wandb.finish()
