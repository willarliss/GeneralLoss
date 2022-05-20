from functools import partial

import numpy as np
from sklearn.exceptions import NotFittedError

from .utils import (
    EPS,
    Array_NxP,
    Array_Nx1,
    linear_link,
    sigmoid_link,
    mse_loss,
    bce_loss,
    zero_penalty,
    l1_penalty,
    l2_penalty,
    elasticnet_penalty,
    check_weights,
    Minimize,
    BaseEstimatorABC,
)

supported_methods = ('BFGS', 'L-BFGS-B', 'SLSQP')


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

        if self.solver.upper() not in supported_methods:
            raise ValueError
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
            raise ValueError

        return self.loss_fn

    def get_link_fn(self):

        if (self.link_fn is None) and (self._estimator_type == 'classifier'):
            return sigmoid_link

        if (self.link_fn is None) and (self._estimator_type == 'regressor'):
            return linear_link

        if not callable(self.link_fn):
            raise ValueError

        return self.link_fn

    def get_reg_fn(self):

        if self.penalty == 'none':
            return zero_penalty
        if self.penalty == 'l1':
            return partial(l1_penalty, alpha=self.alpha)
        if self.penalty == 'l2':
            return partial(l2_penalty, alpha=self.alpha)
        if self.penalty == 'elasticnet':
            return partial(elasticnet_penalty, alpha=self.alpha, gamma=self.l1_ratio)
        raise ValueError

    def partial_fit(self, X: Array_NxP, y: Array_Nx1,
                    sample_weight: Array_Nx1 = None, classes: tuple = None):

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
            raise NotFittedError

        X = self._validate_data(X, reset=False)
        link_function = self.get_link_fn()

        return link_function(X, self.coef_)
