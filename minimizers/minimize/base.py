"""Base class for minimizer objects"""

from abc import ABC

import numpy as np
from sklearn.base import BaseEstimator

from ..typing import Array
from ..utils import filter_check_args


class BaseEstimatorABC(BaseEstimator, ABC):
    """Abstract base class for custom loss estimators.
    """

    def __init__(self):

        self.set_check_params()
        self.set_estimator_type('classifier')
        self.set_multi_output(False)

    def _validate_data(self,
                       X='no_validation',
                       y='no_validation',
                       reset=True,
                       validate_separately=False,
                       **check_params):


        val_X = not (isinstance(X, str) and X=='no_validation')
        val_y = not (y is None or isinstance(y, str) and y=='no_validation')

        multi_output = self._multi_output
        check_params = filter_check_args(
            val_X=val_X,
            val_y=val_y,
            args={**check_params, **self._check_params, **{'multi_output': multi_output}},
        )

        out = super()._validate_data(
            X=X,
            y=y,
            reset=reset,
            validate_separately=validate_separately,
            **check_params,
        )

        if val_X and val_y:
            if self.fit_intercept:
                out = np.c_[np.ones(out[0].shape[0]), out[0]], out[1]
            if multi_output and (out[1].ndim == 1):
                out = out[0], out[1].reshape(-1,1)
            if reset:
                self.n_outputs_ = 1 if not multi_output else out[1].shape[1]
                self.n_inputs_ = out[0].shape[1]

        elif val_X and (not val_y):
            if self.fit_intercept:
                out = np.c_[np.ones(out.shape[0]), out]
            if reset:
                self.n_inputs_ = out.shape[1]

        elif (not val_X) and val_y:
            if multi_output and (out[1].ndim == 1):
                out = out.reshape(-1,1)
            if reset:
                self.n_outputs_ = 1 if not multi_output else out.shape[1]

        return out

    def set_estimator_type(self, etype: str):
        """Set the `_estimator_type` attribute manually. Should be `regressor` or `classifier`.
        """

        if etype not in ('classifier', 'regressor'):
            raise ValueError(
                'Only estimator types `classifier` and `regressor` are supported.'
                f' Not {etype}'
            )

        self._estimator_type = str(etype)

        return self

    def set_multi_output(self, multi: bool):
        """Set the `_multi_output` attribute manually. Should be boolean.
        """

        self._multi_output = bool(multi)

        return self

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters passed to `__init__`.
        """

        return super().get_params(deep=False)

    def set_params(self, **params):
        """Set `__init__` parameters .
        """

        return super().set_params(**params)

    def set_check_params(self, **check_params):
        """Set the `check_parameters` used in the `_validate_data` call.
        """

        if 'multi_output' in check_params:
            raise ValueError('`multi_output` parameter must be set with `set_multi_output`')

        self._check_params = check_params.copy()

        return self

    def get_check_params(self, deep: bool = True) -> dict:
        """Get the `check_parameters` used in the `_validate_data` call.
        """

        if deep:
            return self._check_params().copy()

        return self._check_params()


    def fit(self, X: Array, y: Array, sample_weight: Array = None, **kwargs) -> Array:
        """Perform one training pass on input data `X` and targets `y` applying observation
        weights `sample_weight`.
        """

        raise NotImplementedError

    def partial_fit(self, X: Array, y: Array, sample_weight: Array = None, **kwargs) -> Array:
        """Perform training on input data `X` and targets `y` applying observation
        weights `sample_weight` until convergence.
        """

        raise NotImplementedError

    def predict(self, X: Array, **kwargs) -> Array:
        """Use link function to make predictions on the input data `X`.
        """

        raise NotImplementedError
