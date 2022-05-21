import warnings

import numpy as np

from .utils import EPS
from .types import Array_NxP, Array_1xP, Array_PxK, Array_Nx1


def linear_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:

    return X.dot(b.T)

def sigmoid_link(X: Array_NxP, b: Array_1xP) -> Array_Nx1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = 1 / (1+np.exp(-linear_link(X, b)))

    return np.clip(y_hat, EPS, 1-EPS)

def softmax_link(X: Array_NxP, B: Array_PxK) -> Array_Nx1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_hat = np.exp(X.dot(B))

    return y_hat / y_hat.sum(1).reshape(-1,1)
