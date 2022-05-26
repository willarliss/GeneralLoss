"""Predefined link functions supported by minimizers"""

import warnings
from functools import wraps

import numpy as np

from .utils import EPS
from .typing import Array_NxP, Array_1xP, Array_PxK, Array_Nx1, Array_NxK, LinkFunction


def link_fn_multioutput_reshape(outputs: int) -> LinkFunction:
    """Wrapper for link functions to reshape a flat coefficient array into proper 2d-array
    for multioutput prediction. During minimization, a data array of shape (N,P) and a
    coefficient array of shape (P*K,) is passed. This wrapper reshapes the coefficient inputs
    into shape (P, K).

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
            return link_fn(X, B.reshape(X.shape[1], outputs))
        return link_fn_wrapped

    return wrapper


def linear_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:
    """Linear combination of input matrix and coefficient vector.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        b: [ndarray] A (1,P) or (P,) array of coefficients.

    Returns:
        [ndarray] A (N,) array of continuous valued predictions.

    Raises:
        None.
    """

    return X.dot(b.T)


def multi_linear_link(X: Array_NxP, B: Array_PxK) -> Array_NxK:
    """Linear combination of input matrix and coefficient matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        B: [ndarray] A (P,N) array of coefficients.

    Returns:
        [ndarray] A (N,K) array of continuous valued predictions.

    Raises:
        None.
    """

    return X.dot(B)


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

    return np.clip(y_hat, EPS, 1-EPS)


def softmax_link(X: Array_NxP, B: Array_PxK) -> Array_NxK:
    """Softmax function applied to linear combination of input matrix and coefficient matrix.

    Parameters:
        X: [ndarray] A (N,P) array of input data.
        B: [ndarray] A (P,N) array of coefficients.

    Returns:
        [ndarray] A (N,K) array of probabilisitc predictions.

    Raises:
        None.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = np.exp(multi_linear_link(X, B))

    return y_hat / y_hat.sum(1).reshape(-1,1)
