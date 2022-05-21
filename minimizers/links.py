import warnings
from typing import Callable
from functools import wraps

import numpy as np

from .utils import EPS
from .types import Array_NxP, Array_1xP, Array_PxK, Array_Nx1, Array_NxK


def link_fn_multioutput_reshape(outputs: int) -> Callable:

    outputs = int(outputs)

    def wrapper(link_fn):
        @wraps(link_fn)
        def link_fn_wrapped(X, B):
            return link_fn(X, B.reshape(X.shape[1], outputs))
        return link_fn_wrapped

    return wrapper


def linear_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:

    return X.dot(b.T)


def multi_linear_link(X: Array_NxP, B: Array_PxK) -> Array_NxK:

    return X.dot(B)


def sigmoid_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = 1 / (1+np.exp(-linear_link(X, b)))

    return np.clip(y_hat, EPS, 1-EPS)


def softmax_link(X: Array_NxP, B: Array_PxK) -> Array_NxK:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = np.exp(multi_linear_link(X, B))

    return y_hat / y_hat.sum(1).reshape(-1,1)
