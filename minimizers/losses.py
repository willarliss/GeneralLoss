"""Predefined loss functions supported by minimizers"""

import numpy as np

from .utils import EPS
from .typing import Array_Nx1, Array_NxK


def mse_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Mean squared error loss for single output regression.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of continuous valued targets.
        y_hat: [ndarray] A (N,1) or (N,) array of continuous valued predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    return 0.5 * (y-y_hat)**2


def multi_mse_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Mean squared error loss for multi-output regression.

    Parameters:
        y: [ndarray] A (N,K) array of continuous valued targets.
        y_hat: [ndarray] A (N,K) array of continuous valued predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    return ((Y-Y_hat)**2).sum(1)


def bce_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Binary cross entropy loss for binary classification.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of binary targets.
        y_hat: [ndarray] A (N,1) or (N,) array of probabilisitc predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y_hat = np.clip(y_hat, EPS, 1-EPS)

    entropy = (y)*np.log(y_hat) + (1-y)*np.log(1-y_hat)

    return -entropy


def cce_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical cross entropy loss for multiple classification.

    Parameters:
        y: [ndarray] A (N,K) array of categorical targets.
        y_hat: [ndarray] A (N,K) array of probabilisitc predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y_hat = np.clip(Y_hat, EPS, 1-EPS)

    return -np.log((Y*Y_hat).max(1))


def hinge_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Hinge loss for binary classification.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of binary targets.
        y_hat: [ndarray] A (N,1) or (N,) array of margin predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    zeros = np.zeros_like(y)
    margin = 1 - (y*y_hat)

    return np.c_[zeros, margin].max(1)


def mae_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Mean absolute error loss for binary classification.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of binary targets.
        y_hat: [ndarray] A (N,1) or (N,) array of probabilistic predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    return 1 - y*y_hat - (1-y)*(1-y_hat)


def cmae_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical mean absolute error loss for multiple classification.

    Parameters:
        y: [ndarray] A (N,K) array of categorical targets.
        y_hat: [ndarray] A (N,K) array of probabilistic targets.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    return 1 - (Y*Y_hat).sum(1)
