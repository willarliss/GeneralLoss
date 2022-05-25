import warnings
from typing import Union, Callable
from inspect import getfullargspec

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from sklearn.base import BaseEstimator
from sklearn.utils import DataConversionWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, _check_y

from .typing import Array_Nx1, Array_NxK, Array_1xP


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
                 args: tuple = (),
                 method: Union[str, Callable] = None,
                 jac: Union[Callable, str, bool] = None,
                 hess: Union[Callable, str, bool] = None,
                 hessp: Callable = None,
                 bounds: list = None,
                 constraints: Union[dict, list] = (),
                 tol: float = None,
                 callback: callable = None,
                 maxiter: int = None,
                 disp: int = False,
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

    def __call__(self, fun: Callable, x0: Array_1xP, *, args: tuple = None) -> OptimizeResult:

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


class OneHotLabelEncoder(BaseEstimator):

    def __init__(self, classes: tuple):

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

    def transform(self, y: Array_Nx1, *args, **kwargs) -> Array_NxK:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DataConversionWarning)
            yt = self.le_.transform(y).reshape(-1,1)

        yt = self.ohe_.transform(yt)

        return yt

    def inverse_transform(self, yt: Array_NxK, *args, **kwargs) -> Array_Nx1:

        y = self.ohe_.inverse_transform(yt).squeeze()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DataConversionWarning)
            y = self.le_.inverse_transform(y)

        return y


class FilterCheckArgs:

    def __init__(self):

        ignore = ['X', 'y', 'array']

        spec = getfullargspec(check_X_y)
        self.xy_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

        spec = getfullargspec(check_array)
        self.x_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

        spec = getfullargspec(_check_y)
        self.y_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

    def __call__(self, val_X, val_y, args):

        for arg in args:
            if arg not in self.xy_params|self.x_params|self.y_params:
                warnings.warn(f'Unknown argument: {arg}', category=RuntimeWarning)

        if val_X and val_y:
            return {k:v for k, v in args.items() if k in self.xy_params}
        if val_X and not val_y:
            return {k:v for k, v in args.items() if k in self.x_params}
        if val_X and not val_y:
            return {k:v for k, v in args.items() if k in self.y_params}
        return {}
