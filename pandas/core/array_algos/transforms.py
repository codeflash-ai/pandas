"""
transforms.py is for shape-preserving functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Scalar,
    )


def shift(
    values: np.ndarray, periods: int, axis: AxisInt, fill_value: Scalar
) -> np.ndarray:
    new_values = values

    if periods == 0 or values.size == 0:
        return new_values.copy()

    # make sure array sent to np.roll is c_contiguous
    f_ordered = values.flags.f_contiguous
    if f_ordered:
        new_values = new_values.T
        axis = new_values.ndim - axis - 1

    if new_values.size:
        # Create output array and directly assign shifted values
        out = np.empty_like(new_values)
        n = new_values.shape[axis]

        if abs(periods) >= n:
            out.fill(fill_value)
        elif periods > 0:
            axis_indexer = [slice(None)] * new_values.ndim
            axis_indexer[axis] = slice(0, periods)
            out[tuple(axis_indexer)] = fill_value

            axis_indexer[axis] = slice(0, n - periods)
            out_indexer = [slice(None)] * new_values.ndim
            out_indexer[axis] = slice(periods, n)
            out[tuple(out_indexer)] = new_values[tuple(axis_indexer)]
        else:
            axis_indexer = [slice(None)] * new_values.ndim
            axis_indexer[axis] = slice(n + periods, n)
            out[tuple(axis_indexer)] = fill_value

            axis_indexer[axis] = slice(-periods, n)
            out_indexer = [slice(None)] * new_values.ndim
            out_indexer[axis] = slice(0, n + periods)
            out[tuple(out_indexer)] = new_values[tuple(axis_indexer)]

        new_values = out

    # restore original order
    if f_ordered:
        new_values = new_values.T

    return new_values
