"""
Reversed Operations not available in the stdlib operator module.
Defining these instead of using lambdas allows us to reference them by name.
"""

from __future__ import annotations

import operator
import numpy as np


def radd(left, right):
    return right + left


def rsub(left, right):
    return right - left


def rmul(left, right):
    return right * left


def rdiv(left, right):
    return right / left


def rtruediv(left, right):
    return right / left


def rfloordiv(left, right):
    return right // left


def rmod(left, right):
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if type(right) is str:  # marginally faster than isinstance
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    # Fast path for numpy ndarrays (broadcast-modulo), preserving Python's fallback for anything else
    # Use np.mod directly for numpy arrays and compatible types to avoid costly Python `%` operator loop
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        # Only use np.mod if dtypes are not object for full speed
        if left.dtype != object and right.dtype != object:
            # NumPy's np.mod properly handles broadcasting and is much faster for numeric types
            return np.mod(right, left)
    elif isinstance(right, np.ndarray) and right.dtype != object:
        return np.mod(right, left)
    # Note: intentionally do not optimize for sequence types (list, tuple); fallback to default
    return right % left


def rdivmod(left, right):
    return divmod(right, left)


def rpow(left, right):
    return right**left


def rand_(left, right):
    return operator.and_(right, left)


def ror_(left, right):
    return operator.or_(right, left)


def rxor(left, right):
    return operator.xor(right, left)
