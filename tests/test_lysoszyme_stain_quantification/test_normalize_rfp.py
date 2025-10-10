from __future__ import annotations

import numpy as np
import pytest
from skimage.filters import threshold_otsu

from src.lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp


def test_compute_normalized_rfp_expected_value() -> None:
    red = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 6, 6, 6, 6, 1],
            [1, 6, 8, 8, 6, 1],
            [1, 6, 8, 8, 6, 1],
            [1, 6, 6, 6, 6, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    blue = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    crypts = np.zeros_like(red, dtype=int)
    crypts[2:4, 2:4] = 1

    normalized = compute_normalized_rfp(channels=[red, blue, crypts])
    assert normalized.shape == red.shape
    assert normalized.dtype == np.float64

    threshold = threshold_otsu(blue)
    tissue_mask = blue > threshold
    tissue_without_crypts = tissue_mask & (crypts == 0)
    assert np.any(tissue_mask)
    assert np.any(tissue_without_crypts)
    assert tissue_without_crypts.sum() < tissue_mask.sum()

    ratio = np.divide(red, blue, out=np.zeros_like(red), where=blue > 0)
    background = np.mean(ratio[tissue_without_crypts])
    crypt_mean = np.mean(ratio[crypts != 0])
    expected = np.where(blue > 0, (ratio - background) / crypt_mean, 0.0)
    np.testing.assert_allclose(normalized, expected)


def test_compute_normalized_rfp_requires_crypt_pixels() -> None:
    red = np.array([[1, 1], [2, 2]], dtype=float)
    blue = np.array([[1, 1], [5, 5]], dtype=float)
    crypts = np.zeros_like(red, dtype=int)

    with pytest.raises(ValueError, match="Crypt label image contains no crypt regions"):
        compute_normalized_rfp(channels=[red, blue, crypts])
