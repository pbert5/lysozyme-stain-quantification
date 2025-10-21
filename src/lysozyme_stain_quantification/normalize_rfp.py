from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import dask.array as da
from dask import delayed
import xarray as xr
from skimage.filters import threshold_otsu


def _as_image(value: Any, *, dtype: type | None = None) -> tuple[np.ndarray | da.Array, tuple[int, ...]]:
    """Convert value to an image array, handling both numpy and dask arrays."""
    if isinstance(value, da.Array):
        arr = value.astype(dtype) if dtype else value
    else:
        arr = np.asarray(value, dtype=dtype)
    
    original_shape = arr.shape
    if arr.ndim > 2 and arr.shape[0] == 1:
        if isinstance(arr, da.Array):
            arr = da.squeeze(arr, axis=0)
        else:
            arr = np.squeeze(arr, axis=0)
    return arr, original_shape


def compute_normalized_rfp(
    channels: Sequence[np.ndarray | da.Array],
    *,
    masks: Sequence[np.ndarray] | None = None,
    **_: Any,
) -> da.Array:
    """
    Compute a normalized RFP image that reflects crypt staining relative to background tissue.

    Args:
        channels: Sequence containing the RFP image, DAPI image, and crypt label image.
        masks: Unused placeholder to satisfy the AnalysisStack run contract.
        **_: Additional keyword arguments are ignored; present for contract compatibility.

    Returns:
        Dask array of normalized RFP values where positive numbers indicate stronger crypt contrast.
    """
    del masks  # masks unused by normalization

    if len(channels) < 3:
        raise ValueError("compute_normalized_rfp requires RFP, DAPI, and crypt label images.")

    red_channel, red_shape = _as_image(channels[0], dtype=np.float64)
    blue_channel, blue_shape = _as_image(channels[1], dtype=np.float64)
    crypt_labels, crypt_shape = _as_image(channels[2])

    if red_channel.shape != blue_channel.shape or red_channel.shape != crypt_labels.shape:
        raise ValueError("RFP, DAPI, and crypt label images must share the same shape.")

    # Wrap the entire normalization pipeline in a delayed function
    @delayed(pure=True)
    def _normalize_pipeline(red, blue, crypts):
        """Execute the normalization pipeline on numpy arrays."""
        # Ensure inputs are numpy arrays (compute if dask)
        red = red.compute() if isinstance(red, da.Array) else np.asarray(red, dtype=np.float64)
        blue = blue.compute() if isinstance(blue, da.Array) else np.asarray(blue, dtype=np.float64)
        crypts = crypts.compute() if isinstance(crypts, da.Array) else np.asarray(crypts)

        tissue_threshold = threshold_otsu(blue)
        tissue_mask = blue > tissue_threshold
        if not np.any(tissue_mask):
            raise ValueError("Thresholding DAPI channel produced an empty tissue mask.")

        crypt_mask = crypts != 0
        if not np.any(crypt_mask):
            return np.zeros_like(red, dtype=np.float64)

        tissue_without_crypts = tissue_mask & ~crypt_mask
        if not np.any(tissue_without_crypts):
            raise ValueError("No tissue remains after removing crypt regions from the tissue mask.")

        valid_denominator = blue > 0
        ratio = np.zeros_like(red, dtype=np.float64)
        np.divide(red, blue, out=ratio, where=valid_denominator)

        tissue_ratio_mask = tissue_without_crypts & valid_denominator
        if not np.any(tissue_ratio_mask):
            raise ValueError("No tissue pixels with valid DAPI signal for intensity calculation.")

        background_tissue_intensity = float(np.mean(ratio[tissue_ratio_mask]))

        crypt_ratio_mask = crypt_mask & valid_denominator
        if not np.any(crypt_ratio_mask):
            raise ValueError("No crypt pixels with valid DAPI signal for intensity calculation.")

        average_crypt_intensity = float(np.mean(ratio[crypt_ratio_mask]))
        if average_crypt_intensity == 0.0:
            raise ValueError("Average crypt intensity is zero; cannot normalize.")

        normalized_image = (ratio - background_tissue_intensity) / average_crypt_intensity
        normalized_image = np.where(valid_denominator, normalized_image, 0.0)

        return normalized_image.astype(np.float64, copy=False)
    
    # Create the delayed computation
    result_delayed = _normalize_pipeline(red_channel, blue_channel, crypt_labels)
    
    # Get the actual 2D shape
    shape_2d = red_shape
    if len(shape_2d) > 2:
        shape_2d = red_shape[-2:]  # Get last 2 dimensions
    
    # Convert to dask array
    result = da.from_delayed(result_delayed, shape=shape_2d, dtype=np.float64)
    
    return result
