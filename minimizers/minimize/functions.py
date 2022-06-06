"""Custom loss minimization functions"""

import numpy as np

from .base import BaseEstimatorABC


def batch_indices(length, batch_size=64, min_samples=1, seed=None):

    rng = np.random.default_rng(seed)
    splits = np.arange(batch_size, length, batch_size)
    index = rng.choice(length, size=length, replace=False)

    for batch in np.split(index, splits):
        if batch.shape[0] < min_samples:
            continue
        yield batch


def batch_indices_bootstrap(length, batch_size=64, min_visits=1, max_batches=100, seed=None):

    rng = np.random.default_rng(seed)
    seen = np.zeros(length)
    rounds = 1

    while True:
        batch = rng.choice(length, size=batch_size, replace=True)
        seen[batch] += 1
        rounds += 1

        yield batch
        if (seen>=min_visits).mean() == 1:
            break
        if rounds >= max_batches:
            break


def generate_batches(*arrays,
                     batch_size=64,
                     bootstrap=False,
                     max_batches=100,
                     min_visits=1,
                     min_samples=1,
                     seed=None):

    assert all(isinstance(a, np.ndarray) for a in arrays)
    length = arrays[0].shape[0]
    assert all(a.shape[0]==length for a in arrays)

    if bootstrap:
        indices = batch_indices_bootstrap(
            length=length,
            batch_size=batch_size,
            min_visits=min_visits,
            max_batches=100,
            seed=seed,
        )
    else:
        indices = batch_indices(
            length=length,
            batch_size=batch_size,
            min_samples=min_samples,
            seed=seed,
        )

    for batch in indices:
        yield tuple(a[batch] for a in arrays)


def train(estimator,
          *data,
          batch_size=64,
          epochs=10,
          random_state=None,
          bootstrap=False,
          verbose=0,
          **kwargs):

    if not issubclass(estimator.__class__, BaseEstimatorABC):
        raise ValueError

    if verbose:
        print('|', end='')

    for epoch in range(epochs):

        train_batches = generate_batches(*data,
                                         batch_size=batch_size,
                                         bootstrap=bootstrap,
                                         seed=random_state)

        for batch in train_batches:
            estimator.partial_fit(*batch, **kwargs)
            if verbose:
                print('.', end='')


        if verbose:
            print(f'|{epoch+1}|', end='')

    if verbose:
        print()
