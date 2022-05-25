"""Custom loss minimization objects"""

import warnings
from typing import Union
from functools import partial

import numpy as np
from sklearn.exceptions import NotFittedError

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
    Minimize,
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
            shape = (self.n_features_in_+int(self.fit_intercept), self.n_outputs_)
        else:
            shape = self.n_features_in_ + int(self.fit_intercept)

        return rng.normal(size=shape)

    def _init_minimizer(self, n_iter=None):

        if self.solver.upper() not in METHODS:
            raise ValueError(f'Unsuported solver: {self.solver}')
        if n_iter is None:
            n_iter = self.max_iter
        if self.options is None:
            options = {}
        else:
            options = self.options.copy()
        options['eps'] = EPS

        return Minimize(
            method=self.solver,
            tol=self.tol,
            maxiter=n_iter,
            disp=self.verbose,
            **options,
        )

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

    def get_loss_fn(self) -> LossFunction:

        if (self.loss_fn is None) and (self._estimator_type == 'classifier'):
            return bce_loss

        if (self.loss_fn is None) and (self._estimator_type == 'regressor'):
            return mse_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self) -> LinkFunction:

        if (self.link_fn is None) and (self._estimator_type == 'classifier'):
            return sigmoid_link

        if (self.link_fn is None) and (self._estimator_type == 'regressor'):
            return linear_link

        if not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')

        return self.link_fn

    def get_reg_fn(self) -> PenaltyFunction:

        penalty = penalty_functions(self.penalty)

        if isinstance(self.penalty, str) and self.penalty == 'elasticnet':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Array_NxK, sample_weight: Array_Nx1 = None, **kwargs):

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            self.coef_ = self._init_params()
        else:
            X, y = self._validate_data(X, y, reset=False)

        loss_function = self._define_loss_fn(X, y, sample_weight)
        minimizer = self._init_minimizer(1)

        result = minimizer(
            fun=loss_function,
            x0=self.coef_.copy(),
        )

        self.coef_ = result.x

        return self

    def fit(self, X: Array_NxP, y: Array_NxK, sample_weight: Array_Nx1 = None):

        X, y = self._validate_data(X, y, reset=True)
        self.coef_ = self._init_params()

        loss_function = self._define_loss_fn(X, y, sample_weight)
        minimizer = self._init_minimizer(self.max_iter)

        result = minimizer(
            fun=loss_function,
            x0=self.coef_.copy(),
        )

        self.coef_ = result.x

        return self

    def predict(self, X: Array_NxP) -> Array_NxK:

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)


class CustomLossRegressor(GeneralLossMinimizer):

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
            '`multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str):

        warnings.warn(
            'estimator type cannot be set directly. Automatically set to `regressor`',
            category=RuntimeWarning,
        )

        return super().set_estimator_type('regressor')

    def get_loss_fn(self) -> LossFunction:

        if self.loss_fn is None:
            return multi_mse_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self, wrap: bool = True) -> LinkFunction:

        if self.link_fn is None:
            link = multi_linear_link
        elif not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')
        else:
            link = self.link_fn

        if wrap:
            return link_fn_multioutput_reshape(self.n_outputs_)(link)

        return link


class CustomLossClassifier(GeneralLossMinimizer):

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
            '`multi_output` cannot be set directly. Automatically set to True',
            category=RuntimeWarning,
        )

        return super().set_multi_output(True)

    def set_estimator_type(self, etype: str):

        warnings.warn(
            'estimator type cannot be set directly. Automatically set to `classifier`',
            category=RuntimeWarning,
        )

        return super().set_estimator_type('classifier')

    def get_loss_fn(self) -> LossFunction:

        if self.loss_fn is None:
            return cce_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self, wrap: bool = True) -> LinkFunction:

        if self.link_fn is None:
            link = softmax_link
        elif not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')
        else:
            link = self.link_fn

        if wrap:
            return link_fn_multioutput_reshape(self.le_.n_classes_)(link)

        return link

    def partial_fit(self, X: Array_NxP, y: Array_NxK,
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

        y = self.le_.transform(y)

        loss_function = self._define_loss_fn(X, y, sample_weight)
        minimizer = self._init_minimizer(1)

        result = minimizer(
            fun=loss_function,
            x0=self.coef_.copy(),
        )

        self.coef_ = result.x

        return self

    def fit(self, X: Array_NxP, y: Array_NxK, sample_weight: Array_Nx1 = None):

        X, y = self._validate_data(X, y, reset=True)
        self.le_ = OneHotLabelEncoder(np.unique(y))
        self.n_outputs_ = self.le_.n_classes_
        self.coef_ = self._init_params()

        y = self.le_.transform(y)

        loss_function = self._define_loss_fn(X, y, sample_weight)
        minimizer = self._init_minimizer(self.max_iter)

        result = minimizer(
            fun=loss_function,
            x0=self.coef_.copy(),
        )

        self.coef_ = result.x

        return self
