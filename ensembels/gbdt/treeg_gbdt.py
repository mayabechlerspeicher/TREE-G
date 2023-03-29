import warnings
import logging
import pickle
import numbers
from sklearn.ensemble._base import BaseEnsemble
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.utils import check_random_state
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.metrics import mean_squared_error, accuracy_score
from time import time
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import DTYPE, DOUBLE

from sklearn.utils import check_array
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _deprecate_positional_args

from treeg.graph_treeg.graph_level_treeg import GraphTreeG
from treeg.graph_treeg.aggregator_graph_level import graph_level_aggregators
from gbdt_losses import *
from scipy.special import softmax
from scipy.special import expit as sigmoid


def predict_stages(estimators, X, scale, out):
    n_estimators, K = estimators.shape
    for i in range(n_estimators):
        for k in range(K):
            tree = estimators[i, k]
            out[:, k] += scale * tree.predict(X).flatten()


def predict_stage(estimators, stage, X, scale, out):
    return predict_stages(estimators[stage:stage + 1], X, scale, out)


class VerboseReporter:
    """Reports verbose output to stdout.
    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter
        Parameters
        ----------
        est : Estimator
            The estimator
        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        logging.info(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration.
        Parameters
        ----------
        j : int
            The new iteration
        est : Estimator
            The estimator
        """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            logging.info(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10


class BaseGradientBoostedTreeG(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting. """

    @abstractmethod
    def __init__(self, *, loss, learning_rate, n_estimators, criterion, splitter='sklearn',
                 aggregators=graph_level_aggregators, use_attention_set=True, use_attention_set_comp=True, attention_set_limit=1, save_path=None,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features, ccp_alpha,
                 random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion

        self.splitter = splitter
        self.aggregators = aggregators
        self.use_attention_set = use_attention_set
        self.use_attention_set_comp = use_attention_set_comp
        self.attention_set_limit = attention_set_limit

        self.save_path = save_path
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol


    # def get_masked_subset(self, Xs, mask):
    #     inds = np.where(mask == True)[0]
    #     records = Xs[inds]
    #     return records

    def _fit_stage(self, i, X_graph, y, raw_predictions, sample_weight, sample_mask, random_state):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == np.bool
        loss = self.loss_
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = GraphTreeG(
                # classifier=False,
                #            criterion=self.criterion,
                #            max_features=self.max_features,
                #            min_samples_split = self.min_samples_split,
                #            splitter=self.splitter,
                           aggregators=self.aggregators,
                           # use_attention_set=self.use_attention_set,
                           # use_attention_set_comp=self.use_attention_set_comp,
                           max_attention_depth=self.attention_set_limit,
                           # max_walk_len=self.max_depth,
                           min_leaf_size=self.min_samples_leaf,
                           # random_state=random_state
                           )

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)



            # get relevant sebset of records if needed
            X_subset = X_graph[np.where(sample_mask == True)[0]]
            # X_subset = X_graph.get_masked_subset(sample_mask)
            tree.fit(X_subset, residual[sample_mask], sample_weight=None)
            del X_subset
            X_subset = None

            # update tree leaves, only for the subset of data
            loss.update_terminal_regions(
                tree, X_graph, y, residual, raw_predictions, sample_weight, sample_mask,
                learning_rate=self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        if self._save_checkpoint:
            with open(self.save_path, 'wb') as file:
                pickle.dump(self, file)

        return raw_predictions

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if len(self.classes_) > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """
        #np.random.seed(self.random_state)
        self._rng = check_random_state(self.random_state)

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)
        if self.save_path != None:
            self._save_checkpoint = True
        else:
            self._save_checkpoint = False

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_ = np.resize(self.estimators_,
                                     (total_n_estimators, self.loss_.K))
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = np.resize(self.oob_improvement_,
                                                  total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    def fit(self, X, y, sample_weight=None, monitor=None):

        y = check_array(y, dtype=DTYPE, ensure_2d=False)
        n_samples = X.size
        self.n_features_ = X[0].get_number_of_features

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)

        y = column_or_1d(y, warn=True)
        y = self._validate_y(y, sample_weight)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            inds, inds_val = (train_test_split(range(len(X)),
                                               random_state=self.random_state,
                                               test_size=self.validation_fraction,
                                               stratify=stratify))
            X_val = X[inds_val]
            X = X[inds]

            y_val = y.take(inds_val)
            y = y.take(inds)
            sample_weight_val = sample_weight.take(inds_val)
            sample_weight = sample_weight.take(inds)

            if is_classifier(self):
                if self.n_classes_ != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.'
                    )
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                           dtype=np.float64)
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError:  # regular estimator without SW support
                        raise ValueError(msg)
                    except ValueError as e:
                        if "pass parameters to specific steps of " \
                           "your pipeline using the " \
                           "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = \
                    self.loss_.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

        else:
            # TODO currently not supported
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            #X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
            sample_weight_val, begin_at_stage, monitor)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, raw_predictions, sample_weight,
                    random_state, X_val, y_val, sample_weight_val, begin_at_stage=0, monitor=None):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """

        n_samples = X.shape[0]

        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                # create binary sample mask
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)

                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask,
                random_state)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             raw_predictions[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                        old_oob_score - loss_(y[~sample_mask],
                                              raw_predictions[~sample_mask],
                                              sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    logging.info('Early stopping at stage {}'.format(i))
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    logging.info('Early stopping at stage {}'.format(i))
                    break

        return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        #X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        # if X.shape[1] != self.n_features_: #TODO check this
        #     raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
        #         self.n_features_, X.shape[1]))
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(
                X, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        # raw_predictions = np.zeros((len(X), self.loss_.K))
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        #X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate,
                          raw_predictions)
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total gain  brought by that feature.
       """
        self._check_initialized()

        relevant_trees = [tree
                          for stage in self.estimators_ for tree in stage
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

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.
        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features,)
            The set of target features for which the partial dependence
            should be evaluated.
        Returns
        -------
        averaged_predictions : ndarray of shape \
                (n_trees_per_iteration, n_samples)
            The value of the partial dependence function on each grid point.
        """
        if self.init is not None:
            warnings.warn(
                'Using recursion method with a non-constant init predictor '
                'will lead to incorrect partial dependence values. '
                'Got init=%s.' % self.init,
                UserWarning
            )
        grid = np.asarray(grid, dtype=DTYPE, order='C')
        n_estimators, n_trees_per_stage = self.estimators_.shape
        averaged_predictions = np.zeros((n_trees_per_stage, grid.shape[0]),
                                        dtype=np.float64, order='C')
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(grid, target_features,
                                                averaged_predictions[k])
        averaged_predictions *= self.learning_rate

        return averaged_predictions

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(DOUBLE)
        # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        #X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves

class GradientBoostedTreeGRegressor(RegressorMixin, BaseGradientBoostedTreeG):
    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    @_deprecate_positional_args
    def __init__(self, *, loss='ls', learning_rate=0.1, n_estimators=50,
                 subsample=1.0, criterion='mse', splitter='sklearn',
                 aggregators=graph_level_aggregators, use_attention_set=True, use_attention_set_comp=True,
                 attention_set_limit=1, save_path=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, splitter=splitter,
            aggregators=aggregators, use_attention_set=use_attention_set, use_attention_set_comp=use_attention_set_comp,
            attention_set_limit=attention_set_limit, save_path=save_path,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

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
        # X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    def score(self, X, y, sample_weight=None):
        """
        R^2 (coefficient of determination) regression score function.
        """
        raw_predictions = self._raw_predict(X).ravel()
        u = mean_squared_error(y, raw_predictions, sample_weight=sample_weight)
        v = ((y - y.mean()) ** 2).sum()
        R_2 = 1 - (u / v)
        return R_2

class GradientBoostedTreeGClassifier(ClassifierMixin, BaseGradientBoostedTreeG):
    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    def __init__(self, *, loss='ls', learning_rate=0.1, n_estimators=50,
                 subsample=1.0, criterion='mse', splitter='sklearn',
                 aggregators=graph_level_aggregators, use_attention_set=True, use_attention_set_comp=True,
                 attention_set_limit=1, save_path=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, splitter=splitter,
            aggregators=aggregators, use_attention_set=use_attention_set, use_attention_set_comp=use_attention_set_comp,
            attention_set_limit=attention_set_limit, save_path=save_path,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, splitter=splitter,
            aggregators=aggregators, use_attention_set=use_attention_set, use_attention_set_comp=use_attention_set_comp,
            attention_set_limit=attention_set_limit, save_path=save_path,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)



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
        # X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        probas = self.predict_proba(X)
        if self.n_classes_ > 2:
            pred_labels = np.argmax(probas, axis=1)
        else:
            pred_labels = np.where(probas > 0.5, 1, 0)
        return accuracy_score(y, pred_labels, sample_weight=sample_weight)

    def predict_proba(self, X):
        """Predict class probabilities for X.
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
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        raw_predictions = self._raw_predict(X).ravel()
        if self.n_classes_ > 2:
            return softmax(raw_predictions)
        return sigmoid(raw_predictions)

        # try:
        #     return self.loss_._raw_prediction_to_proba(raw_predictions)
        # except NotFittedError:
        #     raise
        # except AttributeError:
        #     raise AttributeError('loss=%r does not support predict_proba' %
        #                          self.loss)

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
