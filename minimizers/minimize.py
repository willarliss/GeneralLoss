import warnings
from functools import partial

import numpy as np
from sklearn.exceptions import NotFittedError

from .penalties import penalty_functions
from .losses import mse_loss, bce_loss, cce_loss, multi_mse_loss
from .types import Array_NxP, Array_Nx1, Array_NxK
from .links import (linear_link, sigmoid_link, softmax_link, multi_linear_link,
                    link_fn_multioutput_reshape)
from .utils import (
    EPS,
    METHODS,
    check_weights,
    Minimize,
    BaseEstimatorABC,
    OneHotLabelEncoder,
)


class CustomLossMinimizer(BaseEstimatorABC):

    def __init__(self, *,
                 loss_fn=None,
                 link_fn=None,
                 alpha=0.1,
                 penalty='none',
                 l1_ratio=0.15,
                 solver='bfgs',
                 tol=1e-4,
                 max_iter=1000,
                 verbose=0,
                 fit_intercept=True,
                 random_state=None,
                 options=None):

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

    def _init_params(self):

        rng = np.random.default_rng(self.random_state)
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

    def get_loss_fn(self):

        if (self.loss_fn is None) and (self._estimator_type == 'classifier'):
            return bce_loss

        if (self.loss_fn is None) and (self._estimator_type == 'regressor'):
            return mse_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self):

        if (self.link_fn is None) and (self._estimator_type == 'classifier'):
            return sigmoid_link

        if (self.link_fn is None) and (self._estimator_type == 'regressor'):
            return linear_link

        if not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')

        return self.link_fn

    def get_reg_fn(self):

        penalty = penalty_functions(self.penalty)

        if penalty.__name__ == 'elasticnet_penalty':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Array_NxK, sample_weight: Array_Nx1 = None):

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

    def predict(self, X: Array_NxP):

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)


class CustomLossClassifier(BaseEstimatorABC):

    def __init__(self, *,
                 loss_fn=None,
                 link_fn=None,
                 alpha=0.1,
                 penalty='none',
                 l1_ratio=0.15,
                 solver='bfgs',
                 tol=1e-4,
                 max_iter=1000,
                 verbose=0,
                 fit_intercept=True,
                 random_state=None,
                 options=None):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
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

    def _validate_data(self,
                       X='no_validation',
                       y='no_validation',
                       reset=True,
                       validate_separately=False,
                       **check_params):

        check_params.update(self._check_params)
        if not (isinstance(y, str) and y == 'no_validation'):
            check_params.update({'multi_output': True})

        return super()._validate_data(
            X=X,
            y=y,
            reset=reset,
            validate_separately=validate_separately,
            **check_params,
        )

    def set_estimator_type(self, etype):

        warnings.warn(
            'estimator type cannot be set directly. Automatically set to `classifier`',
            category=RuntimeWarning,
        )

        return super().set_estimator_type('classifier')

    def _init_params(self):

        rng = np.random.default_rng(self.random_state)
        shape = (self.n_features_in_+int(self.fit_intercept), self.le_.n_classes_)

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

    def get_loss_fn(self):

        if self.loss_fn is None:
            return cce_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self, wrap=True):

        if self.link_fn is None:
            link = softmax_link
        elif not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')
        else:
            link = self.link_fn

        if wrap:
            return link_fn_multioutput_reshape(self.le_.n_classes_)(link)

        return link

    def get_reg_fn(self):

        penalty = penalty_functions(self.penalty)

        if penalty.__name__ == 'elasticnet_penalty':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Array_Nx1,
                    sample_weight: Array_Nx1 = None, classes: tuple = None):

        if not hasattr(self, 'coef_'):
            X, y = self._validate_data(X, y, reset=True)
            if classes is None:
                raise ValueError('classes must be passed on the first call to `partial_fit`')
            self.le_ = OneHotLabelEncoder(classes)
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

    def fit(self, X: Array_NxP, y: Array_Nx1, sample_weight: Array_Nx1 = None):

        X, y = self._validate_data(X, y, reset=True)
        self.le_ = OneHotLabelEncoder(np.unique(y))
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

    def predict(self, X: Array_NxP):

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)


class CustomLossRegressor(BaseEstimatorABC):

    def __init__(self, *,
                 loss_fn=None,
                 link_fn=None,
                 alpha=0.1,
                 penalty='none',
                 l1_ratio=0.15,
                 solver='bfgs',
                 tol=1e-4,
                 max_iter=1000,
                 verbose=0,
                 fit_intercept=True,
                 random_state=None,
                 options=None):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
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

    def _validate_data(self,
                       X='no_validation',
                       y='no_validation',
                       reset=True,
                       validate_separately=False,
                       **check_params):

        check_params.update(self._check_params)
        if not (isinstance(y, str) and y == 'no_validation'):
            check_params.update({'multi_output': True})

        return super()._validate_data(
            X=X,
            y=y,
            reset=reset,
            validate_separately=validate_separately,
            **check_params,
        )

    def set_estimator_type(self, etype):

        warnings.warn(
            'estimator type cannot be set directly. Automatically set to `regressor`',
            category=RuntimeWarning,
        )

        return super().set_estimator_type('regressor')

    def _init_params(self):

        rng = np.random.default_rng(self.random_state)
        shape = (self.n_features_in_+int(self.fit_intercept), self.n_outputs_)

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

    def get_loss_fn(self):

        if self.loss_fn is None:
            return multi_mse_loss

        if not callable(self.loss_fn):
            raise ValueError(f'Loss function must be a callable object. Not {self.loss_fn}')

        return self.loss_fn

    def get_link_fn(self, wrap=True):

        if self.link_fn is None:
            link = multi_linear_link
        elif not callable(self.link_fn):
            raise ValueError(f'Link function must be a callable object. Not {self.link_fn}')
        else:
            link = self.link_fn

        if wrap:
            return link_fn_multioutput_reshape(self.n_outputs_)(link)

        return link

    def get_reg_fn(self):

        penalty = penalty_functions(self.penalty)

        if penalty.__name__ == 'elasticnet_penalty':
            return partial(penalty, gamma=self.l1_ratio)

        return penalty

    def partial_fit(self, X: Array_NxP, y: Array_Nx1, sample_weight: Array_Nx1 = None):

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

    def fit(self, X: Array_NxP, y: Array_Nx1, sample_weight: Array_Nx1 = None):

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

    def predict(self, X: Array_NxP):

        if not hasattr(self, 'coef_'):
            raise NotFittedError(
                'Estimator not fitted. '
                'Call `fit` with appropriate arguments before calling `predict`'
            )

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)
