from __future__ import annotations

from typing import Any

import numpy as np
import dask.array as da

import xarray as xr
import logging
from skimage.filters import threshold_otsu


logger = logging.getLogger(__name__)


def _as_image(value: Any, *, dtype: type | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(value, dtype=dtype)
    original_shape = arr.shape
    if arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr, original_shape


def compute_normalized_rfp(
    rfp_image: np.ndarray | da.Array,
    dapi_image: np.ndarray | da.Array,
    crypt_labels: np.ndarray | da.Array,
    name: str | None = None,

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
    


    red_channel, red_shape = _as_image(rfp_image, dtype=np.float64)
    blue_channel, blue_shape = _as_image(dapi_image, dtype=np.float64)
    crypt_labels, crypt_shape = _as_image(crypt_labels)

    if red_channel.shape != blue_channel.shape or red_channel.shape != crypt_labels.shape:
        raise ValueError("RFP, DAPI, and crypt label images must share the same shape.")

    def reshape_result(result: np.ndarray) -> np.ndarray:
        output = result.astype(np.float64, copy=False)
        reshaped = output.reshape(red_shape)
        if reshaped.ndim != 2:
            reshaped = np.squeeze(reshaped)
        return reshaped

    def fallback(reason: str) -> np.ndarray:
        logger.warning("compute_normalized_rfp fallback: %s", reason)
        return reshape_result(np.zeros_like(red_channel, dtype=np.float64))

    tissue_threshold = threshold_otsu(blue_channel)
    tissue_mask = blue_channel > tissue_threshold
    if not np.any(tissue_mask):
        return fallback(f"Thresholding DAPI channel produced an empty tissue mask (likely no DAPI signal).{f' (name: {name})' if name else ''}")

    crypt_mask = crypt_labels != 0
    if not np.any(crypt_mask):
        return fallback(f"No crypt labels detected; returning zeroed normalized image.{f' (name: {name})' if name else ''}")

    tissue_without_crypts = tissue_mask & ~crypt_mask
    if not np.any(tissue_without_crypts):
        return fallback(f"Removing crypt regions left no background tissue for normalization.{f' (name: {name})' if name else ''}")

    valid_denominator = blue_channel > 0
    if not np.any(valid_denominator):
        return fallback(f"All DAPI intensities are zero; cannot compute RFP/DAPI ratio.{f' (name: {name})' if name else ''}")

    ratio = np.zeros_like(red_channel, dtype=np.float64)
    np.divide(red_channel, blue_channel, out=ratio, where=valid_denominator)

    tissue_ratio_mask = tissue_without_crypts & valid_denominator
    if not np.any(tissue_ratio_mask):
        return fallback(f"No non-crypt tissue pixels have valid DAPI signal for normalization.{f' (name: {name})' if name else ''}")

    background_tissue_intensity = float(np.mean(ratio[tissue_ratio_mask]))

    crypt_ratio_mask = crypt_mask & valid_denominator
    if not np.any(crypt_ratio_mask):
        return fallback(f"No crypt pixels have valid DAPI signal to compute crypt intensity.{f' (name: {name})' if name else ''}")

    average_crypt_intensity = float(np.mean(ratio[crypt_ratio_mask]))
    if average_crypt_intensity == 0.0:
        return fallback(f"Average crypt intensity is zero; skipping normalization.{f' (name: {name})' if name else ''}")

    normalized_image = (ratio - background_tissue_intensity) / average_crypt_intensity
    normalized_image = np.where(valid_denominator, normalized_image, 0.0)

    return reshape_result(normalized_image)
