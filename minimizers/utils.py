"""Utility functions for minimizers"""

import warnings
from typing import Callable
from inspect import getfullargspec

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, _check_y

from .typing import Array_Nx1


EPS = np.finfo(float).eps ** 0.5
METHODS = ('BFGS', 'L-BFGS-B', 'SLSQP')


class OneHotLabelEncoder(OneHotEncoder):
    """Wrapper around sklearn.preprocessing.OneHotEncoder.

    Parameters:
        classes: [tuple] The known classes/targets in the training data.

    Attributes:
        n_classes_: Number of classes derived from classes argumenet.
        classes, OneHotEncoder attributes
    """

    def __init__(self, classes):

        super().__init__(
            categories=[classes],
            drop=None,
            sparse=False,
        )

        self.classes = classes
        self.n_classes_ = len(classes)
        self.categories_ = [classes]
        self.n_features_in_ = 1
        self.drop_idx_ = None
        self.infrequent_categories_ = None
        self.feature_names_in_ = None

    def _validate_data(self, *args, **kwargs):

        kwargs.update(dict(ensure_2d=False))
        out = super().validate_data(*args, **kwargs)

        if out.ndim == 1:
            return out.reshape(-1,1)
        return out

    def fit(self, *args, **kwargs):
        """Does nothing, categories are already known.
        """

        return NotImplemented


class FilterCheckArgs:
    """Functionality for filtering out appropriate `check_params` passed to
    BaseEstimatorABC._validate_data depending on whether `X`, `y`, or both are being validated.

    Parameters:
        None.

    Attributes:
        xy_params: [set] Parameters accepted by sklearn.utils.validation.check_X_y.
        x_params: [set] Parameters accepted by sklearn.utils.validation.check_array.
        y_params: [set] Parameters accepted by sklearn.utils.validation._check_y.
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
            val_X: [bool] Whether or not `X` is being validated.
            val_y: [bool] Whether or not `y` is being validated.
            args: [dict] Keyword arguments passed to `_validate_data`.

        Returns:
            [dict] Filtered keyword arguments passed to `_validate_data`.

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
        w: [ndarray] Array of obvservation weights.
        y: [ndarray] Array of training targets.

    Returns:
        [ndarray] Array of obvservation weights.

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
