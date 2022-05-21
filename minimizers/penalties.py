from typing import Union, Callable

from numpy.linalg import norm

from .types import Array_Nx1, Array_NxK


def zero_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return 0.


def l1_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return norm(b, 1)


def l2_penalty(b: Union[Array_Nx1, Array_NxK]) -> float:

    return norm(b, 2)


def elasticnet_penalty(b: Union[Array_Nx1, Array_NxK], gamma: float = 0.5) -> float:

    return (gamma)*l1_penalty(b) + (1-gamma)*l2_penalty(b)


def frobenius_penalty(B: Array_NxK) -> float:

    return norm(B, 'fro')


def penalty_functions(name: str) -> Callable:

    if isinstance(name, str) and name == 'none':
        return zero_penalty

    if isinstance(name, str) and name == 'l1':
        return l1_penalty

    if isinstance(name, str) and name == 'l2':
        return l2_penalty

    if isinstance(name, str) and name == 'elasticnet':
        return elasticnet_penalty

    if isinstance(name, str) and name == 'frobenius':
        return frobenius_penalty

    raise ValueError(f'Unknown penalty function: {name}')
