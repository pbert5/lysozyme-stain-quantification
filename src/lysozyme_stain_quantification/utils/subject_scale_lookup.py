"""
Utility functions to map subject metadata to scalar values for AnalysisStackXR runs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import dask.array as da
import xarray as xr


def _as_string(value: Any) -> str:
    """Best-effort conversion of an AnalysisStackXR channel element to string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _as_string(value.item())
        raise ValueError(f"Expected a scalar subject identifier, got array with shape {value.shape}")
    if hasattr(value, "item"):
        try:
            return _as_string(value.item())
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not extract subject name from {type(value)!r}") from exc
    return str(value)


def subject_scale_from_name(
    *,
    channels: Sequence[Any],
    masks: Sequence[Any] | None = None,
    keys: Sequence[str],
    values: Sequence[float],
    default: float | None = None,
) -> float:
    """
    Match a subject name to a scale value based on provided key substrings.

    Parameters
    ----------
    channels:
        Sequence supplied by ``AnalysisStackXR.run``; the first entry is expected
        to come from the ``"subject_name"`` meta channel.
    masks:
        Present for signature compatibility; not used.
    keys:
        List of substrings to search for within the subject name.
    values:
        List of scale factors (e.g., microns-per-pixel) aligned with ``keys``.
    default:
        Optional fallback value if no key matches the subject name.

    Returns
    -------
    float
        The scale value corresponding to the first matching key, or ``default``.

    Raises
    ------
    ValueError
        If ``keys`` and ``values`` have different lengths or inputs are missing.
    KeyError
        If no key matches and no default is provided.
    """
    del masks  # masks are not used for scalar lookup

    if not channels:
        raise ValueError("Expected at least one channel containing the subject name.")
    if len(keys) != len(values):
        raise ValueError("Length of 'keys' must match length of 'values'.")

    subject_raw = channels[0]
    subject_name = _as_string(subject_raw)
    subject_lower = subject_name.lower()

    def _wrap(scale_value: float) -> float | np.ndarray | xr.DataArray:
        if isinstance(subject_raw, xr.DataArray):
            data = np.full(subject_raw.shape, scale_value, dtype=np.float64)
            return xr.DataArray(data, coords=subject_raw.coords, dims=subject_raw.dims, name="microns_per_px")
        if isinstance(subject_raw, da.Array):
            return da.full(subject_raw.shape, scale_value, dtype=np.float64, chunks=subject_raw.chunks)
        if isinstance(subject_raw, np.ndarray):
            return np.full(subject_raw.shape, scale_value, dtype=np.float64)
        return scale_value

    for idx, key in enumerate(keys):
        if key.lower() in subject_lower:
            try:
                value = values[idx]
            except IndexError:
                raise ValueError(f"Missing value for key '{key}'.") from None
            scale_value = float(value)
            return _wrap(scale_value)

    if default is not None:
        scale_value = float(default)
        return _wrap(scale_value)

    raise KeyError(f"No key matched subject '{subject_name}'.")
