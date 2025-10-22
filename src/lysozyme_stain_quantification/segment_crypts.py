
from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import dask.array as da
from dask import delayed
import xarray as xr

from .crypts.identify_potential_crypts_ import identify_potential_crypts
from .crypts.remove_edge_touching_regions_mod import remove_edge_touching_regions_sk
from .crypts.scoring_selector_mod import scoring_selector


def _to_float(value: np.ndarray | float | int) -> float:
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(()))
    raise ValueError(f"Expected scalar microns-per-pixel value, got shape {arr.shape}")


def _as_image(value: Any) -> tuple[np.ndarray | da.Array, tuple[int, ...]]:
    """Convert value to an image array, handling both numpy and dask arrays."""
    if isinstance(value, da.Array):
        arr = value
    else:
        arr = np.asarray(value)
    
    original_shape = arr.shape
    if arr.ndim > 2 and arr.shape[0] == 1:
        if isinstance(arr, da.Array):
            arr = da.squeeze(arr, axis=0)
        else:
            arr = np.squeeze(arr, axis=0)
    return arr, original_shape


def segment_crypts(
    channels: tuple[da.Array, da.Array, Optional[float | int]],
    blob_size_px: int | None = 15,
    *,
    blob_size_um: float | None = None,
    debug: bool = False,
    scoring_weights: dict[str, float] | None = None,
    masks: Sequence[np.ndarray] | None = None,
    max_regions: int = 5,
    microns_per_px: float | None = None,
) -> da.Array:
    """Segment crypts in the given image - returns dask array for lazy evaluation.

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

    # Wrap the entire pipeline in a delayed function

    def _segment_pipeline(crypt: da.Array, tissue: da.Array, blob_size: int, weights: dict[str, float], max_reg: int, dbg: bool):
        """Execute the full segmentation pipeline on numpy arrays."""
        # Ensure inputs are numpy arrays (compute if dask)
        
        
        # Call the three main functions sequentially
        potential_crypts = identify_potential_crypts(crypt, tissue, blob_size, dbg)
        cleaned_crypts = remove_edge_touching_regions_sk(potential_crypts)
        best_crypts, crypt_scores = scoring_selector(
            cleaned_crypts,
            crypt,
            debug=dbg,
            max_regions=max_reg,
            weights=weights,
            return_details=True,
        )
        
        # Return the final segmentation result
        return best_crypts
    
    # Create the delayed computation
    result_delayed = _segment_pipeline(
        crypt_img, tissue_image, effective_blob_size_px,
        scoring_weights, max_regions, debug
    )
    
    # Get the actual 2D shape (strip extra dimensions from crypt_shape if present)
    shape_2d = crypt_shape
    if len(shape_2d) > 2:
        shape_2d = crypt_shape[-2:]  # Get last 2 dimensions
    
    # Convert to dask array
    result = da.from_delayed(result_delayed, shape=shape_2d, dtype=np.int32)

    return result
