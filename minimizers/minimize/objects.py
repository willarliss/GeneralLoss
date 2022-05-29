"""Custom loss minimization objects"""

import warnings
from typing import Union
from functools import partial

import numpy as np
from scipy import optimize
from sklearn.exceptions import NotFittedError
from sklearn.base import RegressorMixin, ClassifierMixin

from .base import BaseEstimatorABC
from ..penalties import penalty_functions
from ..losses import mse_loss, bce_loss, cce_loss, multi_mse_loss
from ..typing import (
    Array_NxP,
    Array_Nx1,
    Array_NxK,
    Array_1xP,
    Array_KxP,
    LossFunction,
    LinkFunction,
    PenaltyFunction,
)
from ..links import (
    linear_link,
    sigmoid_link,
    softmax_link,
    link_fn_multioutput_reshape,
)
from ..utils import (
    EPS,
    METHODS,
    check_weights,
    OneHotLabelEncoder,
)


class GeneralLossMinimizer(BaseEstimatorABC):
    """This class is a general minimizer for custom-defined loss functions. Users can specify a
    custom loss function, custom link function, and custom penalty function. Minimization is
    performed with Scipy and the API is compatible with Scikit-Learn. The class is designed to
    support both classification and regression.

    Parameters:
        loss_fn: [Callable] Loss function to optimize during training. Should accept an array of
            targets and an array of predictions. Should return an array of losses. Defaults to
            binary cross-entropy for _estimator_type='classifier' and _multi_output=False.
            Defaults to categorical cross-entropy for _estimator_type='classifier' and
            _multi_output=True. Defaults to mean squared error for _estimator_type='regressor' and
            _multi_output=False. Defaults to multiple mean squared error for
            _estimator_type='regressor' and _multi_output=True.
        link_fn: [Callable] Link function to make predictions. Should accept an array of
            observation data and an array of parameters. Should return an array of predictions.
            Defaults to the sigmoid function for _estimator_type='classifier' and
            _multi_output=False. Defaults to softmax function for _estimator_type='classifier' and
             _multi_output=True. Defaults to linear/identity link function for
            _estimator_type='regressor'.
        penalty: [Callable, str] Penalty function to regularize parameters during training. Can be
            'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for
            an elastic-net penalty, or a callable function. If a function, it should accept an
            array of coefficients and return a single float. Defaults to no/zero penalty.
        alpha: [float] Penalty strength parameter. Constant by which to multiply the penalty
            function. Should be greater than 0.
        l1_ratio: [float] Elastic-net mixing parameter. Must be between 0 and 1 (inclusive).
            l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
        solver: [str] Solver method used by `scipy.optimize.minimze` to minimize the loss
            function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
        tol: [float] Stopping criterion for minimizer. Should be greater than or equal to 0.
        max_iter: [int] Maximum number of passes over the data during training/minimization.
            Should be greater than 0.
        verbose: [int] Verbosity level. Value other than zero will print convergence messages
            from minimizer.
        fit_intercept: [bool] Whether an intercept term should be fit in training. It True, a
            column of ones is concatenated to input data matrix.
        random_state: [int] Seed for randomly initializing coefficients.
        options: [dict] A dictionary of options to pass to solver. 'maxiter' and 'disp' are
            already included.

    Attributes:
        coef_: Coefficient vector fitted to input data features.
        n_features_in_: Number of features/columns in the input data.
        n_inputs_: Number of input columns (one more than n_features_in_ if fit_intercept=True)
        n_outputs_: Number of targets. In classification, number of classes. In regression, number
             of output variables.
        _estimator_type: Type of estimator the instance is ('classifier' or 'regressor').
        _multi_output: Whether the estimator supports multi-output prediction.
        _check_params: Dictionary of parameters to use for validating input data.
    """

    def __init__(self, *,
                 loss_fn: LossFunction = None,
                 link_fn: LinkFunction = None,
                 penalty: Union[PenaltyFunction, str] = 'none',
                 alpha: float = 0.1,
                 l1_ratio: float = 0.15,
                 solver: str = 'bfgs',
                 tol: float = 1e-4,
                 max_iter: int = 1000,
                 verbose: int = 0,
                 fit_intercept: bool = True,
                 random_state: int = None,
                 options: dict = None):

        super().__init__()

        self.loss_fn = loss_fn
        self.link_fn = link_fn
        self.alpha = alpha
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.options = options

        self.set_multi_output(False)

        if self.alpha <= 0:
            raise ValueError(f'`alpha` must be greater than 0. Not {self.alpha}')
        if not 0.<=self.l1_ratio<=1.:
            raise ValueError(f'`l1_ratio` must be in [0,1]. Not {self.l1_ratio}')
        if tol < 0.:
            raise ValueError(f'`tol` must be greater than or equal to 0. Not {self.tol}')
        if self.max_iter <= 0:
            raise ValueError(f'`max_iter` must be greater than 0. Not {self.max_iter}')

    def _define_loss_fn(self, X, y, weights):

        link_function = self.get_link_fn()
        loss_function = self.get_loss_fn()
        regularization = self.get_reg_fn()
        weights = check_weights(weights, y)

        def loss(params):
            y_hat = link_function(X, params)
            loss = loss_function(y, y_hat)
            reg = regularization(params)
            return loss.dot(weights) + self.alpha*reg

        return loss

    def _partial_fit(self, X, y, coef_0, sample_weight, n_iter):

        loss_function = self._define_loss_fn(X, y, sample_weight)

        if self.solver.upper() not in METHODS:
            raise ValueError(f'Unsuported solver: {self.solver}')
        if self.options is None:
            options = {}
        else:
            options = self.options.copy()
        options.update({
            'eps': EPS,
            'maxiter': n_iter,
            'disp': self.verbose,
        })

        result = optimize.minimize(
            fun=loss_function,
            x0=coef_0,
            options=options,
            method=self.solver,
            tol=self.tol,
        )

        coef_1 = result.x
        if self._multi_output:
            coef_1 = coef_1.reshape(self.n_outputs_, self.n_inputs_)

        return coef_1

    def initialize_coef(self, coef: Union[Array_1xP, Array_KxP] = None):
        """Initialize coefficient array. If nothing is passed, coefficients are initialized
        normally according to number of inputs and number of outputs.

        Parameters:
            coef: [ndarray] Initial coefficients for estimator.

        Returns:
            [self] Instance of GeneralLossMinimizer.

        Raises:
            None.
        """

        if coef is not None:
            self.coef_ = coef

        else:
            rng = np.random.default_rng(self.random_state)
            if self._multi_output:
                self.coef_ = rng.normal(size=(self.n_outputs_, self.n_inputs_))
            else:
                self.coef_ = rng.normal(size=self.n_inputs_)

        return self

    def get_loss_fn(self) -> LossFunction:
        """Returns loss function used by the estimator. In training, the loss function is combined
        with the penalty function and is aggregated by sample weights.

        Parameters:
            None

        Returns:
            [Callable] Loss function.

        Raises:
            ValueError if loss_fn argument on init is not callable.
        """

        if callable(self.loss_fn):
            return self.loss_fn

        if self.loss_fn is None:
            if self._estimator_type=='classifier' and self._multi_output:
                return cce_loss
            if self._estimator_type=='classifier' and not self._multi_output:
                return bce_loss
            if self._estimator_type=='regressor' and self._multi_output:
                return multi_mse_loss
            if self._estimator_type=='regressor' and not self._multi_output:
                return mse_loss

        raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

    def get_link_fn(self, wrap: bool = True) -> LinkFunction:
        """Returns link function used by the estimator.

        Parameters:
            wrap: [boo] Whether to return the function wrapped in `link_fn_multioutput_reshape`.

        Returns:
            [Callable] Link function.

        Raises:
            ValueError if link_fn argument on init is not callable.
        """

        if callable(self.link_fn):
            func = self.link_fn
        elif self.link_fn is None:
            if self._estimator_type=='classifier' and self._multi_output:
                func = softmax_link
            if self._estimator_type=='classifier' and not self._multi_output:
                func = sigmoid_link
            if self._estimator_type=='regressor':
                func = linear_link
        else:
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')

        if wrap and self._multi_output:
            return link_fn_multioutput_reshape(self.n_outputs_)(func)
        return func

    def get_reg_fn(self) -> PenaltyFunction:
        """Returns penalty/regularization function used by the estimator. In training, the
        penalty function is combined with the loss function and is multiplied by the strength
        parameter alpha.

        Parameters:
            None.

        Returns:
            [Callable] Penalty function.

        Raises:
            None
        """

        penalty = penalty_functions(self.penalty)

        if isinstance(self.penalty, str) and self.penalty == 'elasticnet':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Union[Array_NxK, Array_Nx1],
                    sample_weight: Array_Nx1 = None, **kwargs):
        """Perform one training pass on the data.

        Parameters:
            X: [ndarray] A (N,P) array of observation data.
            y: [ndarray] A (N,1) or (N,) array of training targets.
            sample_weight: [ndarray] A (N,1) or (N,) array of weights to be applied to individual
                samples/observations.

        Returns:
            [self] Instance of GeneralLossMinimizer.

        Raises:
            None.
        """

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            self.initialize_coef()
        else:
            X, y = self._validate_data(X, y, reset=False)

        self.coef_ = self._partial_fit(
            X=X,
            y=y,
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=1,
        )

        return self

    def fit(self, X: Array_NxP, y: Union[Array_NxK, Array_Nx1], sample_weight: Array_Nx1 = None):
        """Perform training on the data until convergence (or max_iter is reached).

        Parameters:
            X: [ndarray] A (N,P) array of observation data.
            y: [ndarray] A (N,1) or (N,) array of training targets.
            sample_weight: [ndarray] A (N,1) or (N,) array of weights to be applied to individual
                samples/observations.

        Returns:
            [self] Instance of GeneralLossMinimizer.

        Raises:
            None.
        """

        X, y = self._validate_data(X, y, reset=True)
        self.initialize_coef()

        self.coef_ = self._partial_fit(
            X=X,
            y=y,
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=self.max_iter,
        )

        return self

    def decision_function(self, X: Array_NxP) -> Union[Array_NxK, Array_Nx1]:
        """Use the link function to make predictions on the input data.

        Parameters:
            X: [ndarray] A (N,P) array of input/observation data.

        Returns:
            [ndarray] A (N,) array if _multi_output=False. Returns a (N,K) array
                if _multi_output=True.

        Raises:
            None.
        """

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)

    def predict(self, X: Array_NxP) -> Union[Array_NxK, Array_Nx1]:
        """Wrapper around decision function.

        Parameters:
            X: [ndarray] A (N,P) array of input/observation data.

        Returns:
            [ndarray] A (N,) array if _multi_output=False. Returns a (N,K) array
                if _multi_output=True.

        Raises:
            None.
        """

        return self.decision_function(X)


class CustomLossRegressor(RegressorMixin, GeneralLossMinimizer):
    """This class is a minimizer for custom-defined regression loss functions. Users can specify a
    custom loss function, custom link function, and custom penalty function. Minimization is
    performed with Scipy and the API is compatible with Scikit-Learn. Outputs/targets will always
    be assumed to be multi-dimmensional (i.e. a two axis NumPy array).

    Parameters:
        loss_fn: [Callable] Loss function to optimize during training. Should accept a (N,K) array
            of targets and a (N,K) array of predictions. Should return a (N,) array of losses.
            Defaults to (multiple) mean squared error.
        link_fn: [Callable] Link function to make predictions. Should accept a (N,P) array of
            observation data and a (K,P) array of parameters. Should return a (N,K) array of
            predictions. Defaults to linear/identity link function.
        penalty: [Callable, str] Penalty function to regularize parameters during training. Can be
            'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for
            an elastic-net penalty, or a callable function. If a function, it should accept a
            (K,P) array of coefficients and return a single float. Defaults to no/zero penalty.
        alpha: [float] Penalty strength parameter. Constant by which to multiply the penalty
            function. Should be greater than 0.
        l1_ratio: [float] Elastic-net mixing parameter. Must be between 0 and 1 (inclusive).
            l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
        solver: [str] Solver method used by `scipy.optimize.minimze` to minimize the loss
            function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
        tol: [float] Stopping criterion for minimizer. Should be greater than or equal to 0.
        max_iter: [int] Maximum number of passes over the data during training/minimization.
            Should be greater than 0.
        verbose: [int] Verbosity level. Value other than zero will print convergence messages
            from minimizer.
        fit_intercept: [bool] Whether an intercept term should be fit in training. It True, a
            column of ones is concatenated to input data matrix.
        random_state: [int] Seed for randomly initializing coefficients.
        options: [dict] A dictionary of options to pass to solver. 'maxiter' and 'disp' are
            already included.

    Attributes:
        coef_: Coefficient vector fitted to input data features.
        n_features_in_: Number of features/columns in the input data.
        n_inputs_: Number of input columns (one more than n_features_in_ if fit_intercept=True)
        n_outputs_: Number of output variables/targets.
        _estimator_type: Type of estimator the instance is. Automatically set to `classifier'.
        _multi_output: Whether the estimator supports multi-output prediction. Automatically set
             to True.
        _check_params: Dictionary of parameters to use for validating input data.
    """

    def __init__(self, *,
                 loss_fn: LossFunction = None,
                 link_fn: LinkFunction = None,
                 penalty: Union[PenaltyFunction, str] = 'none',
                 alpha: float = 0.1,
                 l1_ratio: float = 0.15,
                 solver: str = 'bfgs',
                 tol: float = 1e-4,
                 max_iter: int = 1000,
                 verbose: int = 0,
                 fit_intercept: bool = True,
                 random_state: int = None,
                 options: dict = None):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            super().__init__(
                loss_fn=loss_fn,
                link_fn=link_fn,
                alpha=alpha,
                penalty=penalty,
                l1_ratio=l1_ratio,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                fit_intercept=fit_intercept,
                random_state=random_state,
                options=options,
            )

    def set_multi_output(self, multi: bool = True):
        """Set type of estimator the instance should be. Automatically set to 'regressor'.

        Parameters:
            multi: [bool] Multi-output status (True).

        Returns:
            [self] Instance of CustomLossRegressor.

        Raises:
            RuntimeWarning if method is called.
        """

        warnings.warn(
            '`_multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str = 'regressor'):
        """Set type of estimator the instance should be. Automatically set to 'regressor'.

        Parameters:
            etype: [str] Estimator type (regressor).

        Returns:
            [self] Instance of CustomLossRegressor.

        Raises:
            RuntimeWarning if method is called.
        """

        warnings.warn(
            "`_estimator_type` cannot be set directly. Automatically set to 'regressor'",
            category=RuntimeWarning,
        )

        return super().set_estimator_type('regressor')

    def predict(self, X: Array_NxP) -> Array_NxK:
        """Wrapper around decision function. Use the link function to make predictions on the
        input data. lattens array to (N,) if single output is (N,1).

        Parameters:
            X: [ndarray] A (N,P) array of input/observation data.

        Returns:
            [ndarray] (N,K) or (N,) array.

        Raises:
            None.
        """

        y_hat = super().predict(X)

        if y_hat.ndim==2 and y_hat.shape[1]==1:
            return y_hat.flatten()
        return y_hat


class CustomLossClassifier(ClassifierMixin, GeneralLossMinimizer):
    """This class is a minimizer for custom-defined classification loss functions. Users can
    specify a custom loss function, custom link function, and custom penalty function.
    Minimization is performed with Scipy and the API is compatible with Scikit-Learn.
    Outputs/targets are assumed to always be multi-dimmensional. For binary tasks, the outputs
    will be of shape Nx2. For K-class tasks, the outputs will be of shape NxK.

    Parameters:
        loss_fn: [Callable] Loss function to optimize during training. Should accept a (N,K)
            array of targets and a (N,K) array of predictions. Should return a (N,) array of
            losses. Defaults to categorical cross-entropy.
        link_fn: [Callable] Link function to make predictions. Should accept a (N,P) array of
            observation data and a (K,P) array of parameters. Should return a (N,K) array of
            predictions. Defaults to softmax function.
        penalty: [Callable, str] Penalty function to regularize parameters during training. Can
            be 'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet'
            for an elastic-net penalty, or a callable function. If a function, it should accept a
            (K,P) array of coefficients and return a single float. Defaults to no/zero penalty.
        alpha: [float] Penalty strength parameter. Constant by which to multiply the penalty
            function. Should be greater than 0.
        l1_ratio: [float] Elastic-net mixing parameter. Must be between 0 and 1 (inclusive).
            l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
        solver: [str] Solver method used by `scipy.optimize.minimze` to minimize the loss
            function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
        tol: [float] Stopping criterion for minimizer. Should be greater than or equal to 0.
        max_iter: [int] Maximum number of passes over the data during training/minimization.
            Should be greater than 0.
        verbose: [int] Verbosity level. Value other than zero will print convergence messages
            from minimizer.
        fit_intercept: [bool] Whether an intercept term should be fit in training. It True, a
            column of ones is concatenated to input data matrix.
        random_state: [int] Seed for randomly initializing coefficients.
        options: [dict] A dictionary of options to pass to solver. 'maxiter' and 'disp' are
            already included.

    Attributes:
        coef_: Coefficient vector fitted to input data features.
        n_features_in_: Number of features/columns in the input data.
        n_inputs_: Number of input columns (one more than n_features_in_ if fit_intercept=True)
        n_outputs_; Number of classes/targets.
        le_: One-hot label encoder transforming 1d array of classes into a 2d array of one-hot
            encodings.
        _estimator_type: Type of estimator the instance is. Automatically set to `classifier'.
        _multi_output: Whether the estimator supports multi-output prediction. Automatically set
            to True.
        _check_params: Dictionary of parameters to use for validating input data.
    """

    def __init__(self, *,
                 loss_fn: LossFunction = None,
                 link_fn: LinkFunction = None,
                 penalty: Union[PenaltyFunction, str] = 'none',
                 alpha: float = 0.1,
                 l1_ratio: float = 0.15,
                 solver: str = 'bfgs',
                 tol: float = 1e-4,
                 max_iter: int = 1000,
                 verbose: int = 0,
                 fit_intercept: bool = True,
                 random_state: int = None,
                 options: dict = None):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            super().__init__(
                loss_fn=loss_fn,
                link_fn=link_fn,
                alpha=alpha,
                penalty=penalty,
                l1_ratio=l1_ratio,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                fit_intercept=fit_intercept,
                random_state=random_state,
                options=options,
            )

    def set_multi_output(self, multi: bool = True):
        """Set type of estimator the instance should be. Automatically set to 'regressor'.

        Parameters:
            multi: [bool] Multi-output status (True).

        Returns:
            [self] Instance of CustomLossClassifier.

        Raises:
            RuntimeWarning if method is called.
        """

        warnings.warn(
            '`_multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str = 'classifier'):
        """Set type of estimator the instance should be. Automatically set to 'classifier'.

        Parameters:
            etype: [str] Estimator type (classifier).

        Returns:
            [self] Instance of CustomLossClassifier.

        Raises:
            RuntimeWarning if method is called.
        """

        warnings.warn(
            "`_estimator_type` cannot be set directly. Automatically set to 'classifier'",
            category=RuntimeWarning,
        )

        return super().set_estimator_type('classifier')

    def partial_fit(self, X: Array_NxP, y: Array_Nx1,
                    sample_weight: Array_Nx1 = None, classes: tuple = None):
        """Perform one training pass on the data.

        Parameters:
            X: [ndarray] A (N,P) array of observation data.
            y: [ndarray] A (N,1) or (N,) array of training targets.
            sample_weight: [ndarray] A (N,1) or (N,) array of weights to be applied to individual
                samples/observations.
            classes: [tuple] Known classes/targets in the training data.

        Returns:
            [self] Instance of CustomLossClassifier.

        Raises:
            None.
        """

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            if classes is None:
                raise ValueError('classes must be passed on the first call to `partial_fit`')
            self.le_ = OneHotLabelEncoder(classes)
            self.n_outputs_ = self.le_.n_classes_
            self.initialize_coef()
        else:
            X, y = self._validate_data(X, y, reset=False)

        self.coef_ = self._partial_fit(
            X=X,
            y=self.le_.transform(y),
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=1,
        )

        return self

    def fit(self, X: Array_NxP, y: Array_Nx1, sample_weight: Array_Nx1 = None):
        """Perform training on the data until convergence (or max_iter is reached).

        Parameters:
            X: [ndarray] A (N,P) array of observation data.
            y: [ndarray] A (N,1) or (N,) array of training targets.
            sample_weight: [ndarray] A (N,1) or (N,) array of weights to be applied to individual
                samples/observations.

        Returns:
            [self] Instance of CustomLossClassifier.

        Raises:
            None.
        """

        X, y = self._validate_data(X, y, reset=True)
        self.le_ = OneHotLabelEncoder(np.unique(y))
        self.n_outputs_ = self.le_.n_classes_
        self.initialize_coef()

        self.coef_ = self._partial_fit(
            X=X,
            y=self.le_.transform(y),
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=self.max_iter,
        )

        return self

    def predict(self, X: Array_NxP) -> Array_Nx1:
        """Wrapper around decision function. Use the link function to make predictions on the
        input data. Recodes 2d vector into labels and returns a (N,) array.

        Parameters:
            X: [ndarray] A (N,P) array of input/observation data.

        Returns:
            [ndarray] A (N,) array.

        Raises:
            None.
        """

        y_hat = super().predict(X)
        y_hat = self.le_.inverse_transform(
            np.where(y_hat==y_hat.max(1).reshape(-1,1), 1, 0)
        )

        if y_hat.ndim==2 and y_hat.shape[1]==1:
            return y_hat.flatten()
        return y_hat
