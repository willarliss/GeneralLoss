import warnings
from abc import ABC

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils import DataConversionWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .types import Array_Nx1

EPS = np.finfo(float).eps ** 0.5
METHODS = ('BFGS', 'L-BFGS-B', 'SLSQP')


def check_weights(w: Array_Nx1, y: Array_Nx1) -> Array_Nx1:

    if w is None:
        w = np.full(y.shape[0], 1/y.shape[0])
    else:
        w = np.asarray(w)
    if not (np.all(w>=0) and w.shape[0]==y.shape[0]):
        raise ValueError('Weights must be positive and have the same length as the input data')

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
        multi_output = check_params.get('multi_output', False)

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
            if multi_output and (out[1].ndim == 1):
                out = out[0], out[1].reshape(-1,1)
            if reset:
                self.n_outputs_ = 1 if not multi_output else out[1].shape[1]

        elif val_X and (not val_y):
            if self.fit_intercept:
                out = np.c_[np.ones(out.shape[0]), out]

        elif (not val_X) and val_y:
            if multi_output and (out[1].ndim == 1):
                out = out.reshape(-1,1)
            if reset:
                self.n_outputs_ = 1 if not multi_output else out.shape[1]

        return out

    def set_estimator_type(self, etype):

        if etype not in ('classifier', 'regressor'):
            raise ValueError(
                'Only estimator types `classifier` and `regressor` are supported.'
                f' Not {etype}'
            )

        self._estimator_type = etype

        return self

    def get_params(self, deep=True):

        return super().get_params(deep=False)

    def set_params(self, **params):

        return super().set_params(**params)

    def set_check_params(self, **check_params):

        if 'multi_output' in check_params:
            raise ValueError('`multi_output` cannot be set directly')

        self._check_params = check_params.copy()

        return self

    def get_check_params(self, deep=True):

        if deep:
            return self._check_params().copy()

        return self._check_params()


class OneHotLabelEncoder(BaseEstimator):

    def __init__(self, classes):

        self.classes = classes
        self.n_classes_  = len(self.classes)
        self.le_ = LabelEncoder()
        self.ohe_ = OneHotEncoder(sparse=False)

        self.le_.classes_ = self.classes
        self.ohe_.categories_ = [np.arange(len(self.classes))]
        self.ohe_.drop_idx_ = None

    def fit(self, *args, **kwargs):

        return self

    def partial_fit(self, *args, **kwargs):

        return self

    def transform(self, y, *args, **kwargs):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DataConversionWarning)
            yt = self.le_.transform(y).reshape(-1,1)

        yt = self.ohe_.transform(yt)

        return yt

    def inverse_transform(self, yt, *args, **kwargs):

        y = self.ohe_.inverse_transform(yt).squeeze()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DataConversionWarning)
            y = self.le_.inverse_transform(y)

        return y
