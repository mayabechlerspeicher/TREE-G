import numpy as np
import logging
from pprint import pformat
from sklearn.metrics import roc_auc_score
from treeg_gbdt import GradientBoostedGraphTreeGRegressor
from eval_utils.metrics import round_acc
from eval_utils.general import load_pickle, Timer
from experiments import datasets
from treeg.graph_treeg.data_formetter_graph_level import DataFormatter
from treeg.graph_treeg.graph_data_graph_level import GraphData
from sklearn.model_selection import train_test_split
from treeg.graph_treeg.aggregator_graph_level import graph_level_aggregators


def train_test_gbdt_treeg(params, ds_train, train_y, ds_test, test_y,
                          mode='', out_metrics=False, resume=None, eval_train=True, verbose=True):
    if verbose:
        logging.info('############ GBDT TREEG ############ ')
        logging.info('Params:\n' + pformat(params))

    gbdt = GradientBoostedGraphTreeGRegressor(**params)
    eval_met = round_acc
    eval_met_name = round_acc.__name__

    timer = Timer()

    if resume != None:
        gbdt = load_pickle(resume)

        none_estimators_inds = np.where(gbdt.estimators_[:, 0] == None)[0]
        if hasattr(gbdt, 'n_estimators_'):
            n_stages = gbdt.n_estimators_

        elif len(none_estimators_inds):
            n_stages = min(none_estimators_inds)

        else:
            n_stages = gbdt.n_estimators

        if n_stages < gbdt.n_estimators:
            gbdt.estimators_ = gbdt.estimators_[:n_stages]
            gbdt.train_score_ = gbdt.train_score_[:n_stages]
            if hasattr(gbdt, 'oob_improvement_'):
                gbdt.oob_improvement_ = gbdt.oob_improvement_[:n_stages]

        logging.info('Loaded model from {}, with {} trees, resume training'.format(resume, n_stages))

        gbdt.treeg_params(**{'n_estimators': n_stages + params['n_estimators']})
        logging.info('Continue training for {} estimators'.format(params['n_estimators']))

        logging.info('Warning: continue training with the previous parameters')
        logging.info('Original model parameters:')
        logging.info(pformat(params))

    gbdt.fit(ds_train, train_y)

    if verbose:
        logging.info('Train took: {}'.format(timer.end()))

    if mode == 'bin_cls':
        timer = Timer()
        if eval_train:
            train_raw_predictions = gbdt.decision_function(ds_train)
            if verbose:
                logging.info('Eval train took: {}'.format(timer.end()))
        else:
            logging.info('Skipped train evaluation - train metrics are irrelevant')
            train_raw_predictions = np.zeros((len(ds_train),))  # tmp solution

        test_raw_predictions = gbdt.decision_function(ds_test)
        train_encoded_labels = gbdt.loss_._raw_prediction_to_decision(train_raw_predictions)
        train_preds = gbdt.classes_.take(train_encoded_labels, axis=0)
        test_encoded_labels = gbdt.loss_._raw_prediction_to_decision(test_raw_predictions)
        test_preds = gbdt.classes_.take(test_encoded_labels, axis=0)

        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)

        train_probs = gbdt.loss_._raw_prediction_to_proba(train_raw_predictions)
        test_probs = gbdt.loss_._raw_prediction_to_proba(test_raw_predictions)

        train_auc = roc_auc_score(train_y, train_probs[:, 1])
        test_auc = roc_auc_score(test_y, test_probs[:, 1])
        if verbose:
            logging.info(
                'Results : train {} {:.6f} auc: {:.6f} | test {} : {:.4f} auc: {:.4f}'.format(eval_met_name, train_met,
                                                                                              train_auc, eval_met_name,
                                                                                              test_met, test_auc))
    else:
        timer = Timer()
        if eval_train:
            train_preds = gbdt.predict(ds_train)
            if verbose:
                logging.info('Eval train took: {}'.format(timer.end()))
        else:
            logging.info('Skipped train evaluation - train metrics are irrelevant')
            train_preds = np.zeros((len(ds_train),))  # tmp solution

        test_preds = gbdt.predict(ds_test)
        train_met = eval_met(train_y, train_preds)
        test_met = eval_met(test_y, test_preds)
        if verbose:
            logging.info('Results : train {} {:.6f} | test {} : {:.6f}'.format(eval_met_name, train_met,
                                                                               eval_met_name, test_met))

    depths = []
    n_leafs = []
    n_stages, K = gbdt.estimators_.shape
    for i in range(n_stages):
        for k in range(K):
            # depths.append(gbdt.estimators_[i, k].depth) #TODO: add depth to the tree
            depths.append(0)
            n_leafs.append(0)
            # n_leafs.append(gbdt.estimators_[i, k].n_leafs)  #TODO: add n_leafs to the tree

    depths = np.array(depths)
    n_leafs = np.array(n_leafs)
    if verbose:
        logging.info(
            'Trees sizes stats: depth: {:.1f}+-{:.3f} | n_leafs: {:.1f}+-{:.3f}'.format(depths.mean(), depths.std(),
                                                                                        n_leafs.mean(), n_leafs.std()))
    if out_metrics:
        return gbdt, train_met, test_met
    else:
        return gbdt


params = {'exp_name': 'DD',
          'seed': 0,
          'n_train': 100000,
          'n_test': 10000,
          'dim': 100,
          'train_set_size': 20,
          'test_sizes': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300],
          'n_exp': 5,
          'n_estimators': 50,
          'learning_rate': 0.1,
          'max_depth': 2,
          'max_features': None,
          'subsample': 1,
          'random_state': 0
          }

seed = 1
logging.info('Start exp {}'.format(seed))
params['seed'] = seed
np.random.seed(seed)

dataset = datasets.TU_DD()

formatter = DataFormatter(GraphData)
X, y = formatter.pyg_data_list_to_tree_graph_data_list(dataset)
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

treeg_params = {'n_estimators': params['n_estimators'],
              'aggregators': graph_level_aggregators,
              'splitter': 'sklearn',
              'use_attention_set': True,
              'attention_set_limit': 2,
              'max_depth': params['max_depth'],
              'max_features': params['max_features'],
              'subsample': params['subsample'],
              'random_state': params['random_state'],
              'validation_fraction': 0.1,
              'tol': 1e-3,
                'n_iter_no_change': 3,
                'verbose': 3}

xgboost_params = {'n_estimators': params['n_estimators'],
                  'criterion': 'mse',
                  'learning_rate': params['learning_rate'],
                  'max_depth': params['max_depth'],
                  'max_features': params['max_features'],
                  'subsample': params['subsample'],
                  'validation_fraction': 0.1,
                  'tol': 1e-3,
                  'n_iter_no_change': 5,
                  'verbose': 0,
                  'random_state': params['random_state']}

model_treeg, train_acc, test_acc = train_test_gbdt_treeg(treeg_params,
                                                         X_train, y_train,
                                                         X_test, y_test,
                                                         mode='',
                                                         out_metrics=True,
                                                         verbose=False)

print('Train accuracy: {:.4f}'.format(train_acc))
print('Test accuracy: {:.4f}'.format(test_acc))
