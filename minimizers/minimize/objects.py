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
    LossFunction,
    LinkFunction,
    PenaltyFunction,
)
from ..links import (
    linear_link,
    sigmoid_link,
    softmax_link,
    multi_linear_link,
    link_fn_multioutput_reshape,
)
from ..utils import (
    EPS,
    METHODS,
    check_weights,
    OneHotLabelEncoder,
)


class GeneralLossMinimizer(BaseEstimatorABC):

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

    def _init_params(self):

        rng = np.random.default_rng(self.random_state)

        if self._multi_output:
            shape = (self.n_outputs_, self.n_inputs_)
        else:
            shape = self.n_inputs_

        return rng.normal(size=shape)

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

    def get_loss_fn(self) -> LossFunction:

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

        if callable(self.link_fn):
            func = self.link_fn
        elif self.link_fn is None:
            if self._estimator_type=='classifier' and self._multi_output:
                func = softmax_link
            if self._estimator_type=='classifier' and not self._multi_output:
                func = sigmoid_link
            if self._estimator_type=='regressor' and self._multi_output:
                func = multi_linear_link
            if self._estimator_type=='regressor' and not self._multi_output:
                func = linear_link
        else:
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')

        if wrap and self._multi_output:
            return link_fn_multioutput_reshape(self.n_outputs_)(func)
        return func

    def get_reg_fn(self) -> PenaltyFunction:

        penalty = penalty_functions(self.penalty)

        if isinstance(self.penalty, str) and self.penalty == 'elasticnet':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Union[Array_NxK, Array_Nx1],
                    sample_weight: Array_Nx1 = None, **kwargs):

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            self.coef_ = self._init_params()
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

        X, y = self._validate_data(X, y, reset=True)
        self.coef_ = self._init_params()

        self.coef_ = self._partial_fit(
            X=X,
            y=y,
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=self.max_iter,
        )

        return self

    def decision_function(self, X: Array_NxP) -> Union[Array_NxK, Array_Nx1]:

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)

    def predict(self, X: Array_NxP) -> Union[Array_NxK, Array_Nx1]:

        return self.decision_function(X)


class CustomLossRegressor(RegressorMixin, GeneralLossMinimizer):

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

    def set_multi_output(self, multi: bool):

        warnings.warn(
            '`_multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str):

        warnings.warn(
            "`_estimator_type` cannot be set directly. Automatically set to 'regressor'",
            category=RuntimeWarning,
        )

        return super().set_estimator_type('regressor')

    def predict(self, X: Array_NxP) -> Array_NxK:

        y_hat = super().predict(X)

        if y_hat.ndim==2 and y_hat.shape[1]==1:
            return y_hat.flatten()
        return y_hat


class CustomLossClassifier(ClassifierMixin, GeneralLossMinimizer):

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

    def set_multi_output(self, multi: bool):

        warnings.warn(
            '`_multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str):

        warnings.warn(
            "`_estimator_type` cannot be set directly. Automatically set to 'classifier'",
            category=RuntimeWarning,
        )

        return super().set_estimator_type('classifier')

    def partial_fit(self, X: Array_NxP, y: Array_Nx1,
                    sample_weight: Array_Nx1 = None, classes: tuple = None):

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            if classes is None:
                raise ValueError('classes must be passed on the first call to `partial_fit`')
            self.le_ = OneHotLabelEncoder(classes)
            self.n_outputs_ = self.le_.n_classes_
            self.coef_ = self._init_params()
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

        X, y = self._validate_data(X, y, reset=True)
        self.le_ = OneHotLabelEncoder(np.unique(y))
        self.n_outputs_ = self.le_.n_classes_
        self.coef_ = self._init_params()

        self.coef_ = self._partial_fit(
            X=X,
            y=self.le_.transform(y),
            coef_0=self.coef_.copy(),
            sample_weight=sample_weight,
            n_iter=self.max_iter,
        )

        return self

    def predict(self, X: Array_NxP) -> Array_Nx1:

        y_hat = super().predict(X)
        y_hat = self.le_.inverse_transform(
            np.where(y_hat==y_hat.max(1).reshape(-1,1), 1, 0)
        )

        if y_hat.ndim==2 and y_hat.shape[1]==1:
            return y_hat.flatten()
        return y_hat
