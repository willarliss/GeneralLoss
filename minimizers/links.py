"""Predefined link functions supported by minimizers"""

import warnings
from typing import Union
from functools import wraps

import numpy as np

from .utils import clip_probability, EPS
from .typing import Array_NxP, Array_1xP, Array_KxP, Array_Nx1, Array_NxK, LinkFunction


def link_fn_multioutput_reshape(outputs: int) -> LinkFunction:
    """Wrapper for link functions to reshape a flat coefficient array into proper 2d-array
    for multioutput prediction. During minimization, a data array of shape (N,P) and a
    coefficient array of shape (K*P,) is passed. This wrapper reshapes the coefficient inputs
    into shape (K,P).

    Parameters:
        outputs: [int] Number of outputs K passed to an array.

    Returns:
        [Callable] Link function wrapper.

    Raises:
        None.
    """

    outputs = int(outputs)

    def wrapper(link_fn):

        @wraps(link_fn)
        def link_fn_wrapped(X, B):
            return link_fn(X, B.reshape(outputs, X.shape[1]))

        return link_fn_wrapped

    return wrapper


def linear_link(X: Array_NxP, b: Union[Array_1xP, Array_KxP]) -> Union[Array_Nx1, Array_NxK]:
    """Linear combination of input matrix and coefficient vector/matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [ndarray] A (N,) or (N,K) array of continuous valued predictions.

    Raises:
        None.
    """

    return X.dot(b.T)


def sigmoid_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:
    """Sigmoid function applied to linear combination of input matrix and coefficient vector.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (1,P) or (P,) array of coefficients.

    Returns:
        [ndarray] A (N,) array of probabilisitc predictions.

    Raises:
        None.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = 1 / (1+np.exp(-linear_link(X, b)))

    return clip_probability(y_hat)


def softmax_link(X: Array_NxP, b: Array_KxP) -> Array_NxK:
    """Softmax function applied to linear combination of input matrix and coefficient matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (K,P) array of coefficients.

    Returns:
        [ndarray] A (N,K) array of probabilisitc predictions.

    Raises:
        None.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        Y_hat = np.exp(linear_link(X, b))

    Y_hat = Y_hat / Y_hat.sum(1).reshape(-1,1)

    return clip_probability(Y_hat)


def log_link(X: Array_NxP, b: Union[Array_1xP, Array_KxP]) -> Union[Array_Nx1, Array_NxK]:
    """Log-link function applied to linear combination of input matrix and
    coefficient vector/matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [ndarray] A (N,) or (N,K) array of continuous valued predictions.

    Raises:
        None.
    """

    return np.exp(linear_link(X, b))


def inverse_link(X: Array_NxP, b: Union[Array_1xP, Array_KxP]) -> Union[Array_Nx1, Array_NxK]:
    """Inverse-link function applied to linear combination of input matrix and
    coefficient vector/matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [ndarray] A (N,) or (N,K) array of continuous valued predictions.

    Raises:
        None.
    """

    pred = linear_link(X, b)
    pred[pred==0] += EPS

    return 1 / pred
