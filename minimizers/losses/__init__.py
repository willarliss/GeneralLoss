"""Predefined loss functions supported by minimizers"""

from .regression import (
    mse_loss,
    multi_mse_loss,
    pseudo_huber_loss,
    multi_pseudo_huber_loss,
    gaussian_mle,
    multivariate_gaussian_mle,
    poisson_mle,
)
from .classification import (
    bce_loss,
    cce_loss,
    mae_loss,
    cmae_loss,
    neg_box_cox_loss,
    multi_neg_box_cox_loss,
    binomial_mle,
    multinomial_mle,
    perceptron_loss,
    hinge_loss,
)
