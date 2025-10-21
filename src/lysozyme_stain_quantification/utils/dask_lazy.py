from __future__ import annotations

"""
Utilities to bridge numpy-centric routines with dask by wrapping arguments and
results in delayed objects. These helpers keep the core analysis functions lazy
without forcing eager numpy conversions at call time.
"""

from typing import Any, Tuple

import numpy as np
import dask.array as da
import xarray as xr
from dask import delayed
from dask.delayed import Delayed


DelayedInfo = Tuple[Delayed, Tuple[int, ...], np.dtype]


def _normalize_shape(shape: Tuple[Any, ...]) -> Tuple[int, ...]:
    """Return a concrete integer shape tuple suitable for da.from_delayed."""
    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, (int, np.integer)):
            normalized.append(int(dim))
        elif hasattr(dim, "item"):
            normalized.append(int(dim.item()))
        else:
            normalized.append(int(dim))
    return tuple(normalized)


def _extract_single_block(array: da.Array) -> DelayedInfo:
    """
    Return a delayed block plus shape/dtype metadata for a dask array.

    The array is rechunked into a single block to ensure downstream numpy
    routines receive the full payload per subject.
    """
    if array.ndim == 0:
        # Work around rechunk limitations on 0-D arrays by promoting to length-1.
        array = array.reshape((1,))
        promoted = True
    else:
        promoted = False

    target_chunks = tuple((int(dim),) for dim in array.shape)
    array = array.rechunk(target_chunks)
    delayed_block = array.to_delayed().ravel()
    if delayed_block.size != 1:
        raise ValueError("Expected a single delayed block after rechunking.")
    block = delayed_block[0]
    shape = _normalize_shape(array.shape)
    if promoted:
        shape = ()
    return block, shape, array.dtype


def to_delayed_array(value: Any) -> DelayedInfo:
    """
    Convert numpy/dask/xarray payloads into a single delayed block with metadata.
    """
    if isinstance(value, xr.DataArray):
        return to_delayed_array(value.data)
    if isinstance(value, da.Array):
        return _extract_single_block(value)
    array = np.asarray(value)
    block = delayed(np.asarray)(array)
    return block, _normalize_shape(array.shape), array.dtype


def to_delayed_scalar(value: Any) -> Tuple[Delayed, np.dtype]:
    """
    Convert scalar-like payload into a delayed object yielding a numpy scalar.
    """
    block, shape, dtype = to_delayed_array(value)
    if shape not in {(), (1,), (1, 1)}:
        raise ValueError(f"Expected scalar-compatible shape, got {shape!r}")
    return block, dtype


__all__ = ["to_delayed_array", "to_delayed_scalar"]
