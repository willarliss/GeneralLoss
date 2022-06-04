"""Experimental autoencoder anomaly detection objects and functions"""
# pylint: disable=invalid-name

import warnings
from typing import Union
from functools import partial

import numpy as np

from minimizers.utils import check_loss_inputs
from minimizers.minimize.objects import CustomLossRegressor
from minimizers.typing import (
    Array_NxP,
    Array_Nx1,
    Array_NxK,
    Array_1xP,
    LinkFunction,
    PenaltyFunction,
)


def reconstruction_loss(X: Array_NxP, X_hat: Array_NxP) -> Array_Nx1:

    X, X_hat = check_loss_inputs(X, X_hat, multi_output=True)

    return np.linalg.norm(X-X_hat, 2, axis=1)


def encode_(X: Array_NxP, b: Array_1xP, *, latent: int, eps: float = np.e) -> Array_NxK:

    b = b.reshape(X.shape[1], latent)

    X_enc = np.exp(X.dot(b) - eps)

    return X_enc / X_enc.sum(1).reshape(-1,1)


def decode_(X: Array_NxK, b: Array_1xP, *, size: int, eps: float = np.e) -> Array_NxP:

    b = b.reshape(X.shape[1], size)

    X_dec = X.dot(b) + eps

    return X_dec


def encode_decode_(X: Array_NxP, b: Array_1xP, latent: int = 1, eps: float = np.e) -> Array_NxP:

    size, split = X.shape[1], b.shape[0]//2
    b_enc, b_dec = b[:split], b[split:]

    return decode_(
        X=encode_(
            X=X,
            b=b_enc,
            latent=latent,
            eps=eps,
        ),
        b=b_dec,
        size=size,
        eps=eps,
    )


class EpsAutoEncoder(CustomLossRegressor):

    def __init__(self, latent_dim: int, *,
                 penalty: Union[PenaltyFunction, str] = 'none',
                 alpha: float = 0.1,
                 l1_ratio: float = 0.15,
                 solver: str = 'bfgs',
                 tol: float = 1e-4,
                 max_iter: int = 1000,
                 verbose: int = 0,
                 random_state: int = None,
                 warm_start: bool = False,
                 options: dict = None):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            super().__init__(
                loss_fn=reconstruction_loss,
                link_fn=None,
                fit_intercept=False,
                alpha=alpha,
                penalty=penalty,
                l1_ratio=l1_ratio,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                random_state=random_state,
                warm_start=warm_start,
                options=options,
            )

        self.latent_dim = latent_dim

    def encode(self, X: Array_NxP) -> Array_NxP:

        split = self.coef_.shape[0]//2

        return encode_(
            X=X,
            b=self.coef_[:split],
            latent=self.latent_dim,
        )

    def decode(self, Xe: Array_NxP) -> Array_NxP:

        split = self.coef_.shape[0]//2

        return decode_(
            X=Xe,
            b=self.coef_[split:],
            size=self.n_inputs_,
        )

    def get_link_fn(self) -> LinkFunction:

        return partial(encode_decode_, latent=self.latent_dim)

    def fit(self, X: Array_NxP, sample_weight: Array_Nx1 = None):

        return super().fit(X, X, sample_weight=sample_weight)

    def partial_fit(self, X: Array_NxP, sample_weight: Array_Nx1 = None):

        return super().partial_fit(X, X, sample_weight=sample_weight)
