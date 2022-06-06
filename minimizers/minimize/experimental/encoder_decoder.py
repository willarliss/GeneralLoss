"""Experimental encoder-decoder objects and functions"""
# pylint: disable=invalid-name,protected-access

import warnings
from functools import partial
from typing import Union, Callable

import numpy as np

from minimizers.utils import check_loss_inputs
from minimizers.minimize.objects import CustomLossRegressor
from minimizers.minimize.base import BaseEstimatorABC
from minimizers.typing import (
    Array_NxP,
    Array_Nx1,
    Array_NxK,
    Array_1xP,
    LinkFunction,
    PenaltyFunction,
)


def activation_func(X: Array_NxP, func: Union[Callable, str] = 'linear') -> Array_NxP:

    if callable(func):
        Xa = func(X)
    elif func == 'linear':
        Xa = X
    elif func == 'softmax':
        Xa = np.exp(X)
        Xa = Xa / Xa.sum(1).reshape(-1,1)
    elif func == 'tanh':
        Xa = np.tanh(X)
    elif func == 'sigmoid':
        Xa = 1 / (1+np.exp(-X))
    else:
        raise ValueError(f'Unknown activation function: {func}')
    return Xa


def reconstruction_loss(X: Array_NxP, X_hat: Array_NxP) -> Array_Nx1:

    X, X_hat = check_loss_inputs(X, X_hat, multi_output=True)

    return np.linalg.norm(X-X_hat, 2, axis=1)


def encode_(X: Array_NxP, b: Array_1xP, *,
            latent: int, bias: bool = True, activation: str = 'linear') -> Array_NxK:

    if bias:
        b = b.reshape(X.shape[1]+1, latent)
        X_enc = X.dot(b[1:,:]) + b[0,:]

    else:
        b = b.reshape(X.shape[1], latent)
        X_enc = X.dot(b[:,:])

    return activation_func(X_enc, func=activation)


def decode_(X: Array_NxK, b: Array_1xP, *, size: int, bias: bool = True) -> Array_NxP:

    if bias:
        b = b.reshape(X.shape[1]+1, size)
        X_dec = X.dot(b[1:,:]) + b[0,:]

    else:
        b = b.reshape(X.shape[1], size)
        X_dec = X.dot(b[:,:])

    return X_dec


def encode_decode_(X: Array_NxP, b: Array_1xP,
                   latent: int = 1, bias: bool = True, activation: str = 'linear') -> Array_NxP:

    size = X.shape[1]
    split = X.shape[1]*latent + int(bias)*latent
    b_enc, b_dec = b[:split], b[split:]

    return decode_(
        X=encode_(
            X=X,
            b=b_enc,
            latent=latent,
            bias=bias,
            activation=activation,
        ),
        b=b_dec,
        size=size,
        bias=bias,
    )


class EncoderDecoder(CustomLossRegressor):

    def __init__(self, latent_dim: int, *,
                 activation: Union[Callable, str] = 'linear',
                 penalty: Union[PenaltyFunction, str] = 'none',
                 alpha: float = 0.1,
                 l1_ratio: float = 0.15,
                 fit_intercept: bool = True,
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

        self.activation = activation
        self.bias = fit_intercept
        self.latent_dim = latent_dim

    def _partial_fit(self, X, y, coef_0, sample_weight, n_iter):

        #coef = super()._partial_fit(X, y, coef_0, sample_weight, n_iter)
        coef = BaseEstimatorABC._partial_fit(self, X, y, coef_0, sample_weight, n_iter)

        return coef

    def initialize_coef(self):

        size = sum([
            2*(self.n_inputs_*self.latent_dim),
            int(self.bias)*self.latent_dim,
            int(self.bias)*self.n_inputs_,
        ])

        rng = np.random.default_rng(self.random_state)

        self.coef_ = rng.normal(size=size)

        self._split = sum([
            (self.n_inputs_*self.latent_dim),
            int(self.bias)*self.latent_dim,
        ])

        return self

    def encode(self, X: Array_NxP) -> Array_NxP:

        return encode_(
            X=X,
            b=self.coef_[:self._split],
            latent=self.latent_dim,
            bias=self.bias,
            activation=self.activation,
        )

    def decode(self, Xe: Array_NxP) -> Array_NxP:

        return decode_(
            X=Xe,
            b=self.coef_[self._split:],
            size=self.n_inputs_,
            bias=self.bias,
        )

    def get_link_fn(self) -> LinkFunction:

        return partial(encode_decode_,
                       latent=self.latent_dim,
                       bias=self.bias,
                       activation=self.activation)

    def fit(self, X: Array_NxP, sample_weight: Array_Nx1 = None):

        return super().fit(X, X, sample_weight=sample_weight)

    def partial_fit(self, X: Array_NxP, sample_weight: Array_Nx1 = None):

        return super().partial_fit(X, X, sample_weight=sample_weight)
