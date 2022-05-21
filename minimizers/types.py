from numpy import ndarray

class Array_PxK(ndarray):
    """Numpy ndarray of shape (P,K). Model parameters for multi-output."""

class Array_NxK(ndarray):
    """Numpy ndarray of shape (N,K). Train/test labels for multi-output."""

class Array_NxP(ndarray):
    """Numpy ndarray of shape (N,P). Train/test data."""

class Array_1xP(ndarray):
    """Numpy ndarray of shape (1,P) or (P,). Model parameters."""

class Array_Nx1(ndarray):
    """Numpy ndarray of shape (N,1) or (N,). Train/test labels or instance weights."""
