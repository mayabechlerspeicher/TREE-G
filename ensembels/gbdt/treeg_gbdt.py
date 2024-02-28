import logging
import numpy as np
from time import time
from treeg.graph_treeg.graph_level_treeg import GraphTreeG
from treeg.node_treeg.node_level_treeg import NodeTreeG
from treeg.graph_treeg.aggregator_graph_level import graph_level_aggregators
from fixed_star_boost import loss as starboost_loss
from fixed_star_boost import boosting as starboosting


class BaseGradientBoostedTreeG():
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=50,
                 subsample=1.0, criterion='mse',
                 attention_set_limit=1,
                 min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, random_state=None,
                 max_features=None, max_leaf_nodes=None, tol=1e-4):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.aggregators = graph_level_aggregators
        self.attention_set_limit = attention_set_limit
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.tol = tol

        self.gbtreeg = None

    def fit(self, X, y):
        self.n_features_ = X[0].get_number_of_features
        self.gbtreeg.fit(X, y)
        return self

    def predict(self, X):
        """Predict regression target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        return self.gbtreeg.predict(X)

    def apply(self, X):
        self._check_initialized()
        # X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.gbtreeg.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    def score(self, X, y):
        return self.gbtreeg.score(X, y)

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total gain  brought by that feature.
       """

        relevant_trees = [tree
                          for stage in self.gbtreeg.estimators_ for tree in stage
                          if tree.trained_tree_root_.node_count > 1]
        if not relevant_trees:
            # degenerate case where all trees have only one node
            return np.zeros(shape=self.n_features_, dtype=np.float64)

        relevant_feature_importances = [
            tree.feature_importances_
            for tree in relevant_trees
        ]
        avg_feature_importances = np.mean(relevant_feature_importances,
                                          axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)


class GradientBoostedGraphTreeGRegressor(BaseGradientBoostedTreeG):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=50, criterion='mse',
                 attention_set_limit=1, min_samples_split=2, min_samples_leaf=1,
                 max_walk_len=3, random_state=None, max_leaf_nodes=None, tol=1e-4):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                         criterion=criterion, attention_set_limit=attention_set_limit,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         max_depth=max_walk_len, random_state=random_state, max_leaf_nodes=max_leaf_nodes, tol=tol)

        self.gbtreeg = starboosting.BoostingRegressor(
            init_estimator=GraphTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
            ),
            base_estimator=GraphTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate)


class GradientBoostedGraphTreeGClassifier(BaseGradientBoostedTreeG):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=50, criterion='mse',
                 attention_set_limit=1, min_samples_split=2, min_samples_leaf=1,
                 max_walk_len=3, random_state=None, max_leaf_nodes=None, tol=1e-4):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                         criterion=criterion, attention_set_limit=attention_set_limit,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         max_depth=max_walk_len, random_state=random_state, max_leaf_nodes=max_leaf_nodes, tol=tol)

        self.gbtreeg = starboosting.BoostingClassifier(
            init_estimator=GraphTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
            ),
            base_estimator=GraphTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate)

    def predict_proba(self, X):
        return self.gbtreeg.predict_proba(X)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)


class GradientBoostedNodeTreeGRegressor(BaseGradientBoostedTreeG):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=50, criterion='mse',
                 attention_set_limit=1, min_samples_split=2, min_samples_leaf=1,
                 max_walk_len=3, random_state=None, max_leaf_nodes=None, tol=1e-4):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                         criterion=criterion, attention_set_limit=attention_set_limit,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         max_depth=max_walk_len, init=init, random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                         tol=tol)

        self.gbtreeg = starboosting.BoostingRegressor(
            init_estimator=NodeTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
                random_state=random_state
            ),
            base_estimator=NodeTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
                random_state=random_state
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate, loss=starboost_loss.LogLoss)


class GradientBoostedNodeTreeGClassifier(BaseGradientBoostedTreeG):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=50, criterion='mse',
                 attention_set_limit=1, min_samples_split=2, min_samples_leaf=1,
                 max_walk_len=3, random_state=None, max_leaf_nodes=None, tol=1e-4):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                         criterion=criterion, attention_set_limit=attention_set_limit,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         max_depth=max_walk_len, init=init, random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                         tol=tol)

        self.gbtreeg = starboosting.BoostingClassifier(
            init_estimator=NodeTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
                random_state=random_state
            ),
            base_estimator=NodeTreeG(
                max_attention_depth=self.attention_set_limit,
                walk_lens=list(range(0, self.max_depth + 1)),
                min_leaf_size=self.min_samples_leaf,
                random_state=random_state
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate, loss=starboost_loss.LogLoss)

    def predict_proba(self, X):
        return self.gbtreeg.predict_proba(X)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)
