"""Custom loss minimization functions"""

import numpy as np

from .base import BaseEstimatorABC


def batch_indices(length, batch_size=64, min_samples=1, seed=None):
    """Generate batches of input arrays (no replacement).

    Parameters:
        length: [int] Training data to return batches of.
        batch_size: [int] Size of each batch.
        min_samples: [int] Minimum number of samples required in a batch.
        seed: [int] Random state for consistent sampling.

    Returns:
        [generator] Generator of batches of input arrays.

    Raises:
        None.
    """

    rng = np.random.default_rng(seed)
    splits = np.arange(batch_size, length, batch_size)
    index = rng.choice(length, size=length, replace=False)

    for batch in np.split(index, splits):
        if batch.shape[0] < min_samples:
            continue
        yield batch


def batch_indices_bootstrap(length, batch_size=64, min_visits=1, max_batches=100, seed=None):
    """Generate batches of input arrays (with replacement). Continues iterating until all
    instances sampled.

    Parameters:
        length: [int] Training data to return batches of.
        batch_size: [int] Size of each batch.
        max_batches: [int] Maximum number to cutoff bootstrap sampling at.
        min_visits: [int] Minimum number of times each instance must be sampled during
            bootstrap sampling.
        seed: [int] Random state for consistent sampling.

    Returns:
        [generator] Generator of batches of input arrays.

    Raises:
        None.
    """

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
    """Generate batches of input arrays.

    Parameters:
        arrays: [*ndarray] Training data to return batches of.
        batch_size: [int] Size of each batch.
        bootstrap: [bool] Whether to sample with replacement. Continues iterating until all
            instances sampled.
        max_batches: [int] Maximum number to cutoff bootstrap sampling at.
        min_visits: [int] Minimum number of times each instance must be sampled during
            bootstrap sampling.
        min_samples: [int] Minimum number of samples required in a batch.
        seed: [int] Random state for consistent sampling.

    Returns:
        [generator] Generator of batches of input arrays.

    Raises:
        ValueError if not all items in `arrays` is an array of uniform length.
    """

    if not all(isinstance(a, np.ndarray) for a in arrays):
        raise ValueError('All input arrays must be numpy ndarrays')
    length = arrays[0].shape[0]
    if not all(a.shape[0]==length for a in arrays):
        raise ValueError('All input arrays must be of the same length')

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


def train(
    estimator,
    *data,
    batch_size=64,
    epochs=10,
    bootstrap=False,
    verbose=False,
    random_state=None,
    **kwargs,
):
    """Perform a set number of training passes on input data using a given estimator.

    Parameters:
        estimator: [object] Instance of minimizers Estimator.
        data: [*ndarray] Training data to return batches of.
        batch_size: [int] Size of each batch.
        epochs: [int] .
        bootstrap: [bool] Whether to sample with replacement. Each epoch continues iterating
            until all instances sampled.
        verbose: [bool] .
        random_state: [int] Random state for consistent sampling.
        kwargs: [**] Keyword arguments passed to partial_fit

    Returns:
        None.

    Raises:
        ValueError estimator is not a minimizers estimator (subclass of BaseEstimatorABC).
    """

    if not issubclass(estimator.__class__, BaseEstimatorABC):
        raise ValueError('Estimator must be a subclass of BaseEstimatorABC')

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
