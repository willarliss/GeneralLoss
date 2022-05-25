import warnings
from typing import Union, Callable
from inspect import getfullargspec

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from sklearn.utils import DataConversionWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, _check_y

from .typing import Array_Nx1, Array_NxK, Array_1xP


EPS = np.finfo(float).eps ** 0.5
METHODS = ('BFGS', 'L-BFGS-B', 'SLSQP')


class Minimize:
    """Wrapper for `scipy.optimize.minimize` function. From documentation:
    Minimization of scalar function of one or more variables.

    Parameters:
        args: (tuple) Extra arguments passed to the objective function and its derivatives.
        method: (str, Callable) Type of solver.
        jac: (Callable, str, bool) Method for computing the gradient vector.
        hess: (Callable, str, bool) Method for computing the Hessian matrix.
        hessp: (Callable) Hessian of objective function times an arbitrary vector p.
        bounds: (list) List of bounds on variables for certain methods.
        constraints: (dict, list) Constraints to the optimization problem.
        tol: (float) Tolerance for termination.
        callback: (Callable) Called after each iteration.
        maxiter: (int) Maximum number of iterations to perform.
        disp: (bool) Whether to print convergence messages.
        options: (**) A dictionary of solver options.

    Attributes:
        args, method, jac, hess, hessp, bounds, constraints, tol, callback, options.
    """

    def __init__(self, *,
                 args: tuple = (),
                 method: Union[str, Callable] = None,
                 jac: Union[Callable, str, bool] = None,
                 hess: Union[Callable, str, bool] = None,
                 hessp: Callable = None,
                 bounds: list = None,
                 constraints: Union[dict, list] = (),
                 tol: float = None,
                 callback: Callable = None,
                 maxiter: int = None,
                 disp: bool = False,
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
        """Call functionality.

        Parameters:
            fun: (Callable) The objective function to be minimized.
            x0: (ndarray) Array for initial guess.
            args: (tuple) Arguments to override those passed on __init__.

        Returns:
            (OptimizeResult) The optimization result.

        Raises:
            None.
        """

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


class OneHotLabelEncoder(TransformerMixin, BaseEstimator):
    """Wrapper around sklearn.preprocessing.LabelEncoder and sklearn.preprocessing.OneHotEncoder.

    Parameters:
        classes: (tuple) The known classes/targets in the training data.

    Attributes:
        classes
        n_classes_: Number of classes derived from classes argumenet.
        le_: (LabelEncoder) LabelEncoder object.
        ohe_: (OneHotEncoder) OneHotEncoder object.
    """

    def __init__(self, classes: tuple):

        self.classes = classes
        self.n_classes_  = len(self.classes)
        self.le_ = LabelEncoder()
        self.ohe_ = OneHotEncoder(sparse=False)

        self.le_.classes_ = self.classes
        self.ohe_.categories_ = [np.arange(len(self.classes))]
        self.ohe_.drop_idx_ = None

    def fit(self, *args, **kwargs):
        """Does nothing.
        """

        return self

    def partial_fit(self, *args, **kwargs):
        """Does nothing.
        """

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
    """Functionality for filtering out appropriate `check_params` passed to
    BaseEstimatorABC._validate_data depending on whether `X`, `y`, or both are being validated.

    Parameters:
        None.

    Attributes:
        xy_params: (set) Parameters accepted by sklearn.utils.validation.check_X_y.
        x_params: (set) Parameters accepted by sklearn.utils.validation.check_array.
        y_params: (set) Parameters accepted by sklearn.utils.validation._check_y.
    """

    def __init__(self):

        ignore = ['X', 'y', 'array']

        spec = getfullargspec(check_X_y)
        self.xy_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

        spec = getfullargspec(check_array)
        self.x_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

        spec = getfullargspec(_check_y)
        self.y_params = set(arg for arg in spec.args+spec.kwonlyargs if arg not in ignore)

    def __call__(self, val_X, val_y, args):
        """Call functionality.

        Parameters:
            val_X: (bool) Whether or not `X` is being validated.
            val_y: (bool) Whether or not `y` is being validated.
            args: (dict) Keyword arguments passed to `_validate_data`.

        Returns:
            (dict) Filtered keyword arguments passed to `_validate_data`.

        Raises:
            RuntimeWarning if unrecognized keyword arguments are passed.
        """

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

# Initialize FilterCheckArgs as function
filter_check_args: Callable = FilterCheckArgs()


def check_weights(w: Array_Nx1, y: Array_Nx1) -> Array_Nx1:
    """Validate that the observations weights are all positive and match the input data length.
    If no weights are passed, weights are set to 1/N for arithmetic averaging.

    Parameters:
        w: (ndarray) Array of obvservation weights.
        y: (ndarray) Array of training targets.

    Returns:
        (ndarray) Array of obvservation weights.

    Raises:
        ValueError if weights are not all positive and do not match input data length.
    """

    if w is None:
        w = np.full(y.shape[0], 1/y.shape[0])
    else:
        w = np.asarray(w)
    if not (np.all(w>=0) and w.shape[0]==y.shape[0]):
        raise ValueError('Weights must be positive and have the same length as the input data')

    return w
