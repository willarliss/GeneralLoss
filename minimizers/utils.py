import warnings
from abc import ABC

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

EPS = np.finfo(float).eps ** 0.5


class Array_NxP(np.ndarray):
    """Numpy ndarray of shape (N,P). Train/test data."""
class Array_1xP(np.ndarray):
    """Numpy ndarray of shape (1,P) or (P,). Model parameters."""
class Array_Nx1(np.ndarray):
    """Numpy ndarray of shape (N,1) or (N,). Train/test labels or instance weights."""


def linear_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:
    return X.dot(b)

def sigmoid_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = 1 / (1+np.exp(-linear_link(X, b)))
    return np.clip(y_hat, EPS, 1-EPS)

def mse_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    return 0.5 * (y-y_hat)**2

def bce_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    entropy = (y)*np.log(y_hat) + (1-y)*np.log(1-y_hat)
    return -entropy

def zero_penalty(b: Array_Nx1) -> float:
    return 0.

def l1_penalty(b: Array_Nx1, alpha: float = 0.1) -> float:
    return np.linalg.norm(b, 1) * alpha

def l2_penalty(b: Array_Nx1, alpha: float = 0.1) -> float:
    return np.linalg.norm(b, 2) * alpha

def elasticnet_penalty(b: Array_Nx1, gamma: float = 0.5, alpha: float = 0.1) -> float:
    return ((gamma)*l1_penalty(b) + (1-gamma)*l2_penalty(b)) * alpha

def check_weights(w: Array_Nx1, y: Array_Nx1) -> Array_Nx1:
    if w is None:
        w = np.full(y.shape[0], 1/y.shape[0])
    else:
        w = np.asarray(w)
    if not (np.all(w>=0) and w.shape[0]==y.shape[0]):
        raise ValueError
    return w


class Minimize:

    def __init__(self, *,
                 args=(),
                 method=None,
                 jac=None,
                 hess=None,
                 hessp=None,
                 bounds=None,
                 constraints=(),
                 tol=None,
                 callback=None,
                 maxiter=None,
                 disp=False,
                 **options):

        self.args = args
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
        self.options['maxiter'] = maxiter
        self.options['disp'] = disp

    def __call__(self, fun, x0, *, args=None):

        if args is None:
            args = self.args

        result = minimize(
            fun=fun,
            x0=x0,
            args=args,
            method=self.method,
            jac=self.jac,
            hess=self.hess,
            hessp=self.hessp,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )

        return result


class BaseEstimatorABC(BaseEstimator, ABC):

    def __init__(self):

        self.set_check_params()
        self.set_estimator_type('classifier')

    def _validate_data(self,
                       X='no_validation',
                       y='no_validation',
                       reset=True,
                       validate_separately=False,
                       **check_params):

        check_params.update(self._check_params)

        out = super()._validate_data(
            X=X,
            y=y,
            reset=reset,
            validate_separately=validate_separately,
            **check_params,
        )

        val_X = not (isinstance(X, str) and X=='no_validation')
        val_y = not (y is None or isinstance(y, str) and y=='no_validation')

        if val_X and val_y:
            if self.fit_intercept:
                out = np.c_[np.ones(out[0].shape[0]), out[0]], out[1]

        elif val_X and (not val_y):
            if self.fit_intercept:
                out = np.c_[np.ones(out.shape[0]), out]

        elif (not val_X) and val_y:
            pass

        return out

    def set_estimator_type(self, etype):

        if etype not in ('classifier', 'regressor'):
            raise ValueError

        self._estimator_type = etype

        return self

    def get_params(self, deep=True):

        if deep:
            return super().get_params(deep=False).copy()

        return super().get_params(deep=False)

    def set_params(self, **params):

        super().set_params(**params)

        return self

    def set_check_params(self, **check_params):

        if 'multi_output' in check_params:
            raise ValueError

        self._check_params = check_params.copy()

        return self

    def get_check_params(self, deep=True):

        if deep:
            return self._check_params().copy()

        return self._check_params()
