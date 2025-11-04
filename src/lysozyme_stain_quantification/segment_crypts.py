
from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import dask.array as da

import xarray as xr

from .crypts.identify_potential_crypts_ import identify_potential_crypts
from .crypts.remove_edge_touching_regions_mod import remove_edge_touching_regions_sk
from .crypts.scoring_selector_mod import scoring_selector
from .crypts.crypt_detection_solutions.effective_crypt_estimation import (
    estimate_effective_selected_crypt_count as _estimate_effective_selected_crypt_count,
    EffectiveCryptEstimation,
)


def _to_float(value: np.ndarray | float | int) -> float:
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(()))
    raise ValueError(f"Expected scalar microns-per-pixel value, got shape {arr.shape}")


def _as_image(value: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(value)
    original_shape = arr.shape
    if arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr, original_shape


def segment_crypts(
    channels: Sequence[np.ndarray],
    blob_size_px: int | None = 15,
    *,
    blob_size_um: float | None = None,
    debug: bool = False,
    scoring_weights: dict[str, float] | None = None,
    masks: Sequence[np.ndarray] | None = None,
    max_regions: int = 5,
    microns_per_px: float | None = None,
) -> np.ndarray:
    """Segment crypts in the given image.

    Args:
        channels: Sequence containing the RFP channel, DAPI channel, and optionally a scalar microns-per-pixel value.
        blob_size_um: Approximate crypt size expressed in microns. Converted to pixels using the microns-per-pixel value.
        blob_size_px: Legacy approximate size of crypts in pixels. Used when ``blob_size_um`` is ``None``.
        microns_per_px: Explicit scalar fallback if the third channel is not provided.
    """
    if len(channels) < 2:
        raise ValueError("segment_crypts requires at least two channels (RFP and DAPI).")

    scoring_weights = scoring_weights if scoring_weights is not None else {
        "circularity": 0.35,  # Most important - want circular regions
        "area": 0.25,  # Second - want consistent sizes
        "line_fit": 0.15,  # Moderate - want aligned regions
        "red_intensity": 0.15,  # Moderate - want bright regions
        "com_consistency": 0.10,  # Least - center consistency
    }

    crypt_img, crypt_shape = _as_image(channels[0])
    tissue_image, tissue_shape = _as_image(channels[1])
    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    if len(channels) >= 3:
        microns_per_px = _to_float(channels[2])
    elif microns_per_px is not None:
        microns_per_px = float(microns_per_px)

    effective_blob_size_px: int
    if blob_size_um is not None:
        if microns_per_px is None:
            raise ValueError("Microns-per-pixel value is required when blob_size_um is provided.")
        if microns_per_px <= 0:
            raise ValueError("Microns-per-pixel value must be positive.")
        effective_blob_size_px = max(1, int(round(blob_size_um / microns_per_px)))
    else:
        if blob_size_px is None:
            raise ValueError("blob_size_px cannot be None when blob_size_um is not given.")
        effective_blob_size_px = int(blob_size_px)

    potential_crypts = identify_potential_crypts(crypt_img, tissue_image, effective_blob_size_px, debug)

    cleaned_crypts = remove_edge_touching_regions_sk(potential_crypts)

    best_crypts, crypt_scores = scoring_selector(
        cleaned_crypts,
        crypt_img,
        debug=debug,
        max_regions=max_regions,
        weights=scoring_weights,
        return_details=True,
    )
    
    shaped = best_crypts.reshape(crypt_shape)
    if shaped.ndim != 2:
        shaped = np.squeeze(shaped)
    return shaped


def segment_crypts_dual(
    channels: Sequence[np.ndarray],
    blob_size_px: int | None = 15,
    *,
    blob_size_um: float | None = None,
    debug: bool = False,
    scoring_weights: dict[str, float] | None = None,
    masks: Sequence[np.ndarray] | None = None,
    max_regions_best: int = 5,
    microns_per_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return both the candidate (cleaned) labels and the final best selection.
    This mirrors segment_crypts() but exposes the base labels so callers can
    perform alternative selections (e.g., larger max_regions for a medium set).
    """
    if len(channels) < 2:
        raise ValueError("segment_crypts_dual requires at least two channels (RFP and DAPI).")

    scoring_weights = scoring_weights if scoring_weights is not None else {
        "circularity": 0.35,
        "area": 0.25,
        "line_fit": 0.15,
        "red_intensity": 0.15,
        "com_consistency": 0.10,
    }

    crypt_img, crypt_shape = _as_image(channels[0])
    tissue_image, _ = _as_image(channels[1])
    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    if len(channels) >= 3:
        microns_per_px = _to_float(channels[2])
    elif microns_per_px is not None:
        microns_per_px = float(microns_per_px)

    if blob_size_um is not None:
        if microns_per_px is None:
            raise ValueError("Microns-per-pixel value is required when blob_size_um is provided.")
        if microns_per_px <= 0:
            raise ValueError("Microns-per-pixel value must be positive.")
        effective_blob_size_px = max(1, int(round(blob_size_um / microns_per_px)))
    else:
        if blob_size_px is None:
            raise ValueError("blob_size_px cannot be None when blob_size_um is not given.")
        effective_blob_size_px = int(blob_size_px)

    potential_crypts = identify_potential_crypts(crypt_img, tissue_image, effective_blob_size_px, debug)
    cleaned_crypts = remove_edge_touching_regions_sk(potential_crypts)

    best_crypts, _ = scoring_selector(
        cleaned_crypts,
        crypt_img,
        debug=debug,
        max_regions=max_regions_best,
        weights=scoring_weights,
        return_details=True,
    )

    # Ensure 2D return for best only; base is already 2D
    shaped_best = best_crypts.reshape(crypt_shape)
    if shaped_best.ndim != 2:
        shaped_best = np.squeeze(shaped_best)
    return cleaned_crypts, shaped_best


def estimate_effective_count_from_segmented(
    best_crypts: np.ndarray,
    rfp_image: np.ndarray,
    dapi_image: np.ndarray | None = None,
    *,
    blob_size_um: float | None = None,
    microns_per_px: float | None = None,
    subject_name: str | None = None,
    output_dir: str | None = None,
    scoring_weights: dict[str, float] | None = None,
    save_debug: bool = False,
    expansion_scale: float = 0.7,
) -> EffectiveCryptEstimation:
    """Convenience wrapper to estimate the effective crypt count for a label image.

    See crypts/crypt_detection_solutions/effective_crypt_estimation.py for details.
    """
    return _estimate_effective_selected_crypt_count(
        best_crypts=best_crypts,
        rfp_image=rfp_image,
        dapi_image=dapi_image,
        blob_size_um=blob_size_um,
        microns_per_px=microns_per_px,
        subject_name=subject_name,
        output_dir=output_dir,
        scoring_weights=scoring_weights,
        save_debug=save_debug,
        expansion_scale=expansion_scale,
    )
