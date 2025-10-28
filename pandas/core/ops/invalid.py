"""
Templates for invalid operations.
"""

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        ArrayLike,
        Scalar,
        npt,
    )


def invalid_comparison(
    left: ArrayLike,
    right: ArrayLike | Scalar,
    op: Callable[[Any, Any], bool],
) -> npt.NDArray[np.bool_]:
    """
    If a comparison has mismatched types and is not necessarily meaningful,
    follow python3 conventions by:

        - returning all-False for equality
        - returning all-True for inequality
        - raising TypeError otherwise

    Parameters
    ----------
    left : array-like
    right : scalar, array-like
    op : operator.{eq, ne, lt, le, gt}

    Raises
    ------
    TypeError : on inequality comparisons
    """
    if op is operator.eq:
        res_values = np.zeros(left.shape, dtype=bool)
    elif op is operator.ne:
        res_values = np.ones(left.shape, dtype=bool)
    else:
        typ = type(right).__name__
        raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")
    return res_values


def make_invalid_op(name: str) -> Callable[..., NoReturn]:
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str

    Returns
    -------
    invalid_op : function
    """

    # Avoid re-accessing type(self) and attribute lookup by precalculating closure values
    def invalid_op(self: object, other: object = None) -> NoReturn:
        # Fast local access to type name via __class__ avoid globals lookup for 'type'
        raise TypeError(
            f"cannot perform {name} with this index type: {self.__class__.__name__}"
        )

    invalid_op.__name__ = name
    return invalid_op
