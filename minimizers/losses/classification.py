"""Predefined classification loss functions supported by minimizers"""

import numpy as np
from scipy import stats

from ..utils import check_loss_inputs
from ..typing import Array_Nx1, Array_NxK


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

    y, y_hat = check_loss_inputs(y, y_hat,
                                 clip_probas=True,
                                 expected_targets={0,1})

    entropy = (y)*np.log(y_hat) + (1-y)*np.log(1-y_hat)

    return -entropy


def cce_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical cross entropy loss for multiple classification.

    Parameters:
        Y: [ndarray] A (N,K) array of categorical targets.
        Y_hat: [ndarray] A (N,K) array of probabilisitc predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat,
                                 clip_probas=True,
                                 multi_output=True,
                                 expected_targets={0,1})

    return -np.log((Y*Y_hat).max(1))


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

    y, y_hat = check_loss_inputs(y, y_hat,
                                 expected_targets={0,1})

    return 1 - y*y_hat - (1-y)*(1-y_hat)


def cmae_loss(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Categorical mean absolute error loss for multiple classification.

    Parameters:
        Y: [ndarray] A (N,K) array of categorical targets.
        Y_hat: [ndarray] A (N,K) array of probabilistic targets.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat,
                                 multi_output=True,
                                 expected_targets={0,1})

    return 1 - (Y*Y_hat).sum(1)


def neg_box_cox_loss(y: Array_Nx1, y_hat: Array_Nx1, lam: float = 1.) -> Array_Nx1:
    """Negative Box-Cox transformation function for binary classification.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of binary targets.
        y_hat: [ndarray] A (N,1) or (N,) array of probabilistic predictions.
        lam: [float] Power term in Box-Cox transform.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat,
                                 expected_targets={0,1})

    proba = y*y_hat + (1-y)*(1-y_hat)

    return (1 - proba**lam) / lam


def multi_neg_box_cox_loss(Y: Array_Nx1, Y_hat: Array_Nx1, lam: float = 1.) -> Array_Nx1:
    """Negative Box-Cox transformation function for categorical classification.

    Parameters:
        Y: [ndarray] A (N,K) array of categorical targets.
        Y_hat: [ndarray] A (N,K) array of probabilistic predictions.
        lam: [float] Power term in Box-Cox transform.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat,
                                 multi_output=True,
                                 expected_targets={0,1})

    proba = (Y*Y_hat).max(1)

    return (1 - proba**lam) / lam


def binomial_mle(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Maximum Likelihood Estimation with the Binomial distribution.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of binary targets.
        y_hat: [ndarray] A (N,1) or (N,) array of probabilistic predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat,
                                 clip_probas=True,
                                 expected_targets={0,1})

    return -stats.binom(n=1, p=y_hat).logpmf(y)


def multinomial_mle(Y: Array_NxK, Y_hat: Array_NxK) -> Array_Nx1:
    """Maximum Likelihood Estimation with the Multinomial distribution.

    Parameters:
        Y: [ndarray] A (N,K) array of binary targets.
        Y_hat: [ndarray] A (N,K) array of probabilistic predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    Y, Y_hat = check_loss_inputs(Y, Y_hat,
                                 multi_output=True,
                                 expected_targets={0,1})

    length = Y.shape[0]
    log_likelihood = np.empty(length)

    for idx in range(length):
        log_likelihood[idx]= -stats.multinomial(n=1, p=Y_hat[idx]).logpmf(Y[idx])

    return log_likelihood


def perceptron_loss(y: Array_Nx1, y_hat: Array_Nx1) -> Array_Nx1:
    """Perceptron loss for binary classification.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of +/- targets.
        y_hat: [ndarray] A (N,1) or (N,) array of probabilistic predictions.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat,
                                 expected_targets={-1,1})

    zeros = np.zeros_like(y)
    margin = -(y*y_hat)

    return np.c_[zeros, margin].max(1)


def hinge_loss(y: Array_Nx1, y_hat: Array_Nx1, power: float = 1.) -> Array_Nx1:
    """Hinge loss for binary classification. Optional squared or p-degree.

    Parameters:
        y: [ndarray] A (N,1) or (N,) array of +/- targets.
        y_hat: [ndarray] A (N,1) or (N,) array of margin predictions.
        power: [float] Power to raise hinge loss by.

    Returns:
        [ndarray] A (N,) array of computed losses.

    Raises:
        None.
    """

    y, y_hat = check_loss_inputs(y, y_hat,
                                 expected_targets={-1,1})

    zeros = np.zeros_like(y)
    margin = 1 - (y*y_hat)

    return np.c_[zeros, margin].max(1) ** power
