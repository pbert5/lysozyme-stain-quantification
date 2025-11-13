from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .crypt_detection_solutions.crypt_identification_methodologies import (
    MorphologyParams,
    identify_potential_crypts_hybrid,
    identify_potential_crypts_old_like,
)
from ..utils.debug_image_saver import DebugImageSession


MethodName = Literal["hybrid", "old", "old_like"]


def _calculate_intensity_metrics(
    labels: np.ndarray,
    red_img: np.ndarray,
    blue_img: np.ndarray,
) -> Tuple[float, float]:
    """Compute background and crypt intensity ratios given labeled regions."""
    if labels.shape != red_img.shape or labels.shape != blue_img.shape:
        raise ValueError("Label and intensity images must share the same shape.")

    background_mask = labels == 0
    if not np.any(background_mask):
        # Support legacy labelings where background was stored as 1.
        background_mask = labels == 1

    background_tissue_intensity = 0.0
    if np.any(background_mask):
        red_bg = red_img[background_mask]
        blue_bg = blue_img[background_mask]
        valid_bg = blue_bg > 1e-10
        if np.any(valid_bg):
            background_tissue_intensity = float(np.mean(red_bg[valid_bg] / blue_bg[valid_bg]))

    crypt_mask = labels > 0
    average_crypt_intensity = 0.0
    if np.any(crypt_mask):
        red_crypt = red_img[crypt_mask]
        blue_crypt = blue_img[crypt_mask]
        valid_crypt = blue_crypt > 1e-10
        if np.any(valid_crypt):
            average_crypt_intensity = float(np.mean(red_crypt[valid_crypt] / blue_crypt[valid_crypt]))

    return background_tissue_intensity, average_crypt_intensity


def identify_potential_crypts(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    blob_size_px: int = 30,
    debug: bool = False,
    *,
    method: MethodName = "hybrid",
    params: Optional[MorphologyParams] = None,
    hybrid_kwargs: Optional[Dict[str, Any]] = None,
    old_like_kwargs: Optional[Dict[str, Any]] = None,
    debug_recorder: DebugImageSession | None = None,
) -> np.ndarray:
    """
    Identify potential crypt regions.

    Parameters
    ----------
    crypt_img, tissue_image
        Input intensity images (same shape) for crypt and tissue channels.
    blob_size_px
        Approximate crypt radius in pixels; forwarded to the chosen method.
    method
        Selection between the hybrid pipeline (default) and the legacy watershed core.
    params
        Optional morphology parameter bundle forwarded to the underlying implementation.
    hybrid_kwargs, old_like_kwargs
        Optional per-method overrides. Values here take precedence over the defaults.
    """
    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Image shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    method_key = method.lower()
    method_debug: Dict[str, Any]

    if method_key == "hybrid":
        hybrid_kwargs = dict(hybrid_kwargs or {})
        hybrid_kwargs.pop("debug", None)  # Avoid duplicate debug flags.
        if blob_size_px is not None and "blob_size_px" not in hybrid_kwargs:
            hybrid_kwargs["blob_size_px"] = blob_size_px
        if params is not None and "params" not in hybrid_kwargs:
            hybrid_kwargs["params"] = params

        labels, method_debug = identify_potential_crypts_hybrid(
            crypt_img,
            tissue_image,
            debug=debug,
            debug_recorder=debug_recorder,
            **hybrid_kwargs,
        )
    elif method_key in {"old", "old_like"}:
        old_like_kwargs = dict(old_like_kwargs or {})
        old_like_kwargs.pop("debug", None)
        if blob_size_px is not None and "blob_size_px" not in old_like_kwargs:
            old_like_kwargs["blob_size_px"] = blob_size_px
        if params is not None and "params" not in old_like_kwargs:
            old_like_kwargs["params"] = params

        labels = identify_potential_crypts_old_like(
            crypt_img,
            tissue_image,
            **old_like_kwargs,
            debug_recorder=debug_recorder,
        )
        method_debug = {"mode": "old_like"}
    else:
        raise ValueError(f"Unknown crypt identification method '{method}'.")

    labels = np.asarray(labels).astype(np.int32, copy=False)
    red_img = np.asarray(crypt_img, dtype=np.float32)
    blue_img = np.asarray(tissue_image, dtype=np.float32)

    intensity_metrics = _calculate_intensity_metrics(labels, red_img, blue_img)
    identify_potential_crypts.last_intensity_metrics = intensity_metrics

    if debug:
        identify_potential_crypts.last_debug_info = {
            "method": method_key,
            "blob_size_px": blob_size_px,
            "params": params,
            "method_debug": method_debug,
            "intensity_metrics": intensity_metrics,
        }
    else:
        identify_potential_crypts.last_debug_info = {}

    return labels


identify_potential_crypts.last_intensity_metrics = (0.0, 0.0)
identify_potential_crypts.last_debug_info: Dict[str, Any] = {}
