import numpy as np

from .utils import EPS
from .types import Array_Nx1, Array_NxK


def mse_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Mean squared error loss for single output regression.
    """

    return 0.5 * (y-y_hat)**2


def multi_mse_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Mean squared error loss for multi-output regression.
    """

    return ((Y-Y_hat)**2).sum(1)


def bce_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Cross entropy loss for binary classification.
    """

    y_hat = np.clip(y_hat, EPS, 1-EPS)

    entropy = (y)*np.log(y_hat) + (1-y)*np.log(1-y_hat)

    return -entropy


def cce_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical cross entropy loss for multiple classification.
    """

    Y_hat = np.clip(Y_hat, EPS, 1-EPS)

    return -np.log((Y*Y_hat).max(1))


def hinge_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Hinge loss for binary classification.
    """

    zeros = np.zeros_like(y)
    margin = 1 - (y*y_hat)

    return np.c_[zeros, margin].max(1)


def mae_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Mean absolute error loss for binary classification.
    """

    return 1 - y*y_hat - (1-y)*(1-y_hat)


def categorical_mae_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical mean absolute error loss for multiple classification.
    """

    return 1 - (Y*Y_hat).sum(1)
