"""Predefined penalty functions supported by minimizer objects"""

from typing import Union, Callable

from numpy.linalg import norm

from .typing import Array_KxP, Array_1xP, PenaltyFunction


def zero_penalty(b: Union[Array_1xP, Array_KxP]) -> float:
    """No penalty. Defined for API consistency.

    Parameters:
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [float] The value 0.

    Raises:
        None.
    """

    return 0.


def l1_penalty(b: Union[Array_1xP, Array_KxP]) -> float:
    """L1-norm penalty.

    Parameters:
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [float] A single float of the computed penalty.

    Raises:
        None.
    """

    return norm(b, 1)


def l2_penalty(b: Union[Array_1xP, Array_KxP]) -> float:
    """L2-norm penalty.

    Parameters:
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.

    Returns:
        [float] A single float of the computed penalty.

    Raises:
        None.
    """

    return norm(b, 2)


def elasticnet_penalty(b: Union[Array_1xP, Array_KxP], gamma: float = 0.5) -> float:
    """Elastic-net penalty.

    Parameters:
        b: [ndarray] A (1,P) or (P,) or (K,P) array of coefficients.
        gamma: [float] Elastic-net mixing parameter. Must be between 0 and 1 (inclusive).

    Returns:
        [float] A single float of the computed penalty.

    Raises:
        None.
    """
    return (gamma)*l1_penalty(b) + (1-gamma)*l2_penalty(b)


def penalty_functions(func: Union[str, Callable]) -> PenaltyFunction:
    """Validate and return given penalty/regularization functions. Supports 'none', 'l1',
    'l2', 'elasticnet', or a callable object.

    Parameters:
        func: [str, Callable] Name of penalty function or user defined penalty function.

    Raises:
        ValueError if is not supported or is not callable.
    """

    if isinstance(func, str) and func == 'none':
        return zero_penalty

    if isinstance(func, str) and func == 'l1':
        return l1_penalty

    if isinstance(func, str) and func == 'l2':
        return l2_penalty

    if isinstance(func, str) and func == 'elasticnet':
        return elasticnet_penalty

    if callable(func):
        return func

    raise ValueError(f'Unknown penalty function: {func}')
