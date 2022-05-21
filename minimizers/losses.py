import numpy as np

from .utils import EPS
from .types import Array_Nx1, Array_NxK


def mse_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:

    return 0.5 * (y-y_hat)**2


def multi_mse_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:

    return ((Y-Y_hat)**2).sum(1)


def bce_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:

    entropy = (y)*np.log(y_hat) + (1-y)*np.log(1-y_hat)

    return -entropy


def cce_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:

    Y_hat = np.clip(Y_hat, EPS, 1-EPS)

    return -np.log((Y*Y_hat).max(1))


def hinge_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:

    zeros = np.zeros_like(y)
    margin = 1 - (y*y_hat)

    return np.c_[zeros, margin].max(1)
