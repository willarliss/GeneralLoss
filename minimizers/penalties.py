from typing import Union

from numpy.linalg import norm

from .typing import Array_Nx1, Array_NxK, PenaltyFunction


def zero_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return 0.


def l1_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return norm(b, 1)


def l2_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return norm(b, 2)


def elasticnet_penalty(b: Union[Array_Nx1, Array_NxK], gamma: float = 0.5) -> float:

    return (gamma)*l1_penalty(b) + (1-gamma)*l2_penalty(b)


def penalty_functions(func: str) -> PenaltyFunction:

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
