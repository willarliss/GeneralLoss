import numpy as np

from minimizers.minimize.objects import CustomLossRegressor
from minimizers.minimize.base import BaseEstimatorABC


def flat_encoder_decoder(X, b):
    B = np.dot(b.reshape(-1,1), b.reshape(1,-1))
    B = B / np.linalg.norm(B, 'fro')
    return X.dot(B)


def projection(X, b):
    b = b / np.linalg.norm(b, 2)
    return X.dot(b)


def reconstruction_loss(X, X_hat):
    return np.linalg.norm(X-X_hat, 2, axis=1)


class FlatEncoderDecoder(CustomLossRegressor):

    def __init__(self, *,
                 solver='bfgs',
                 tol=0.,
                 max_iter=1000,
                 verbose=0,
                 random_state=None,
                 warm_start=False,
                 options=None):

        super().__init__(
            loss_fn=None,
            link_fn=None,
            penalty='none',
            fit_intercept=False,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            warm_start=warm_start,
            options=options,
        )

    def _partial_fit(self, X, y, coef_0, sample_weight, n_iter):
        #coef = super()._partial_fit(X, y, coef_0, sample_weight, n_iter)
        coef = BaseEstimatorABC._partial_fit(self, X, y, coef_0, sample_weight, n_iter)
        return coef

    def get_link_fn(self, multi=True):

        if multi:
            return flat_encoder_decoder

        return projection

    def get_loss_fn(self):
        return reconstruction_loss

    def initialize_coef(self):
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(size=self.n_inputs_)
        return self

    def fit(self, X, sample_weight=None):
        return super().fit(X, X, sample_weight=sample_weight)

    def partial_fit(self, X, sample_weight=None):
        return super().partial_fit(X, X, sample_weight=sample_weight)

    def encode_decode(self, X):
        return flat_encoder_decoder(X, self.coef_)

    def project(self, X):
        return self.get_link_fn(multi=False)(X, self.coef_)

    def predict(self, X):
        return (self.project(X)>0).astype(int)
