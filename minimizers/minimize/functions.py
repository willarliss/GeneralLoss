"""Custom loss minimization functions"""

from .base import BaseEstimatorABC

def batch_iterator(X, y=None, batch_size=64, seed=None):

    rng = np.random.default_rng(seed)



def train(estimator,
          X_train,
          y_train=None,
          *,
          X_test=None,
          y_test=None,
          batch_size=64,
          epochs=10,
          random_state=None,
          bootstrap=True,
          metrics=(),
          verbose=0)

    if not issubclass(estimator.__class__, BaseEstimatorABC):
        raise ValueError


