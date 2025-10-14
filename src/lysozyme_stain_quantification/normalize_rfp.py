from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from skimage.filters import threshold_otsu


def compute_normalized_rfp(
    channels: Sequence[np.ndarray],
    *,
    masks: Sequence[np.ndarray] | None = None,
    **_: Any,
) -> np.ndarray:
    """
    Compute a normalized RFP image that reflects crypt staining relative to background tissue.

    Args:
        channels: Sequence containing the RFP image, DAPI image, and crypt label image.
        masks: Unused placeholder to satisfy the AnalysisStack run contract.
        **_: Additional keyword arguments are ignored; present for contract compatibility.

    Returns:
        Array of normalized RFP values where positive numbers indicate stronger crypt contrast.
    """
    del masks  # masks unused by normalization

    if len(channels) < 3:
        raise ValueError("compute_normalized_rfp requires RFP, DAPI, and crypt label images.")

    red_channel = np.asarray(channels[0], dtype=np.float64)
    blue_channel = np.asarray(channels[1], dtype=np.float64)
    crypt_labels = np.asarray(channels[2])

    if red_channel.shape != blue_channel.shape or red_channel.shape != crypt_labels.shape:
        raise ValueError("RFP, DAPI, and crypt label images must share the same shape.")

    tissue_threshold = threshold_otsu(blue_channel)
    tissue_mask = blue_channel > tissue_threshold
    if not np.any(tissue_mask):
        raise ValueError("Thresholding DAPI channel produced an empty tissue mask.")

    crypt_mask = crypt_labels != 0
    if not np.any(crypt_mask):
        return np.zeros_like(red_channel, dtype=np.float64)

    tissue_without_crypts = tissue_mask & ~crypt_mask
    if not np.any(tissue_without_crypts):
        raise ValueError("No tissue remains after removing crypt regions from the tissue mask.")

    valid_denominator = blue_channel > 0
    ratio = np.zeros_like(red_channel, dtype=np.float64)
    np.divide(red_channel, blue_channel, out=ratio, where=valid_denominator)

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
