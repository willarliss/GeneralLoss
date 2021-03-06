"""Type hints for minimizers"""

from typing import Callable, Union

from numpy import ndarray


class Array(ndarray):
    """Basic Numpy ndarray.
    """

class Array_KxP(Array):
    """Numpy ndarray of shape (K,P). Model parameters for multi-output.
    """

class Array_NxK(Array):
    """Numpy ndarray of shape (N,K). Train/test labels for multi-output.
    """

class Array_NxP(Array):
    """Numpy ndarray of shape (N,P). Train/test data.
    """

class Array_1xP(Array):
    """Numpy ndarray of shape (1,P) or (P,). Model parameters.
    """

class Array_Nx1(Array):
    """Numpy ndarray of shape (N,1) or (N,). Train/test labels or instance weights.
    """


LossFunction = Callable[
    [
        Union[Array_Nx1, Array_NxK],
        Union[Array_Nx1, Array_NxK],
    ],
    Array_Nx1,
]

LinkFunction = Callable[
    [
        Array_NxP,
        Union[Array_1xP, Array_KxP]
    ],
    Union[Array_Nx1, Array_NxK],
]

PenaltyFunction = Callable[
    [
        Union[Array_1xP, Array_KxP],
    ],
    float,
]
