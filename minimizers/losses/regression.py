"""Predefined regression loss functions supported by minimizers"""

import numpy as np
from scipy import stats

from ..typing import Array_Nx1, Array_NxK
from ..utils import check_loss_inputs


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

    y, y_hat = check_loss_inputs(y, y_hat)

    return 0.5 * (y-y_hat)**2


def multi_mse_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Mean squared error loss for multi-output regression.

    Parameters:
        Y: [ndarray] A (N,K) array of continuous valued targets.
        Y_hat: [ndarray] A (N,K) array of continuous valued predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat, multi_output=True)

    return ((Y-Y_hat)**2).sum(1)


def pseudo_huber_loss(y: Array_Nx1, y_hat: Array_Nx1, delta: float = 1.) -> Array_Nx1:
    """Pseudo-Huber loss for single output regression.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of continuous valued targets.
        y_hat: [ndarray] A (N,1) or (N,) array of continuous valued predictions.
        delta: [float] Smoothing pararmeter.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat)

    error = y - y_hat

    return delta**2 * (np.sqrt(1 + (error/delta)**2) - 1)


def multi_pseudo_huber_loss(Y: Array_NxK, Y_hat: Array_NxK, delta: float = 1.) -> Array_Nx1:
    """Pseudo-Huber loss for multi-output regression.

    Parameters:
        Y: [ndarray] A (N,K) array of continuous valued targets.
        Y_hat: [ndarray] A (N,K) array of continuous valued predictions.
        delta: [float] Smoothing pararmeter.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat, multi_output=True)

    error = Y - Y_hat
    loss = delta**2 * (np.sqrt(1 + (error/delta)**2) - 1)

    return loss.sum(1)


def gaussian_mle(y: Array_Nx1, y_hat: Array_Nx1, sigma: float = 1.) -> Array_Nx1:
    """Maximum Likelihood Estimation with the Gaussian distribution.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of continuous valued targets.
        y_hat: [ndarray] A (N,1) or (N,) array of continuous valued predictions.
        sigma: [float] Scaling parameter.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat)

    return -stats.norm(loc=y_hat, scale=sigma).logpdf(y)


def multivariate_gaussian_mle(Y: Array_NxK, Y_hat: Array_NxK, sigma: float = 1.) -> Array_Nx1:
    """Maximum Likelihood Estimation with the Multivariate-Gaussian distribution.

    Parameters:
        Y: [ndarray] A (N,K) array of continuous valued targets.
        Y_hat: [ndarray] A (N,K) array of continuous valued predictions.
        sigma: [float] Scaling parameter.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat, multi_output=True)

    length = Y.shape[0]
    log_likelihood = np.empty(length)
    cov = np.eye(Y.shape[1]) * sigma

    for idx in range(length):
        log_likelihood[idx] = -stats.multivariate_normal(mean=Y_hat[idx], cov=cov).logpdf(Y[idx])

    return log_likelihood


def poisson_mle(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Maximum Likelihood Estimation with the Poisson distribution.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of discrete valued targets.
        y_hat: [ndarray] A (N,1) or (N,) array of continuous valued predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat)

    return -stats.poisson(mu=y_hat).logpmf(y)
