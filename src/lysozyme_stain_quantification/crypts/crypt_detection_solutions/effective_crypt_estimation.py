from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import cv2
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import local_maxima, dilation, white_tophat, disk
from skimage.segmentation import find_boundaries
from skimage.feature import match_template
from skimage.restoration import inpaint
from skimage.color import label2rgb

try:
    # Local import (when running inside this repo)
    from ..scoring_selector_mod import scoring_selector
except Exception:  # pragma: no cover - fallback path
    # Package import name
    from lysozyme_stain_quantification.crypts.scoring_selector_mod import scoring_selector


# ----------------------------- helpers ----------------------------- #


def _to_numpy(a: Any) -> np.ndarray:
    """Best effort conversion to numpy array (computes dask arrays if needed)."""
    try:
        import dask.array as da  # local import to avoid hard dependency

        if isinstance(a, da.Array):
            return a.compute()
    except Exception:
        pass
    return np.asarray(a)


def minmax256(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = 255.0 * (arr - lo) / (hi - lo)
    return arr.astype(np.uint8)


def opencv_dilate_disk(img: np.ndarray, r: int) -> np.ndarray:
    k = 2 * int(r) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(img, se)


def caps(image: np.ndarray, small_r: int, big_r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contrast-adaptive "cap" decomposition used to isolate peaks.
    Returns (hats, clean, troughs), consistent with notebook exploration.
    """
    image = np.asarray(image, dtype=np.float32)
    # Historical pre-clean of small salt-and-pepper via white tophat
    image1 = image - white_tophat(image, disk(int(max(1, small_r))))
    hats = opencv_dilate_disk(image1, int(big_r)) - opencv_dilate_disk(image1, int(small_r))
    clean = image1 - np.minimum(image1, hats)
    troughs = np.maximum(image1, hats) - image1
    return hats, clean, troughs


# ------------------------ effective counts ------------------------- #


def effective_label_count_by_values(
    labels: np.ndarray,
    values: np.ndarray,
    region_mask: np.ndarray | None = None,
    *,
    bg_label: int = 0,
    value_agg: str = "sum_positive",  # "sum_positive" | "sum_abs" | "sum"
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute effective label counts by aggregating a scalar field per label.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image (0/`bg_label` is background, >0 are labels).
    values : np.ndarray
        Scalar field image whose mass is accumulated per label.
    region_mask : np.ndarray | None
        Optional boolean mask selecting a region of interest.
    value_agg : {"sum_positive", "sum_abs", "sum"}
        Aggregation of per-pixel values into label mass.
    eps : float
        Numerical guard for log/normalizations.

    Returns
    -------
    dict
        {K_raw, masses, present_labels, p, Neff_simpson, Neff_shannon, evenness}.
    """
    lab = np.asarray(labels)
    val = np.asarray(values)

    if region_mask is None:
        region_mask = np.ones_like(lab, dtype=bool)
    m = region_mask & np.isfinite(val)

    lab = lab[m]
    val = val[m]
    sel = lab != int(bg_label)
    if not np.any(sel):
        return {
            "K_raw": 0,
            "masses": np.array([]),
            "present_labels": np.array([]),
            "p": np.array([]),
            "Neff_simpson": 0.0,
            "Neff_shannon": 0.0,
            "evenness": np.nan,
        }

    lab = lab[sel]
    val = val[sel]

    if value_agg == "sum_positive":
        w = np.maximum(val, 0.0)
    elif value_agg == "sum_abs":
        w = np.abs(val)
    elif value_agg == "sum":
        w = val
    else:
        raise ValueError("value_agg must be 'sum_positive', 'sum_abs', or 'sum'")

    present = np.unique(lab)
    max_id = int(present.max())
    masses_full = np.bincount(lab, weights=w, minlength=max_id + 1)
    masses = masses_full[present]

    total = float(masses.sum())
    if value_agg == "sum" and total <= 0:
        shift = -float(masses.min()) + eps
        masses = masses + shift
        total = float(masses.sum())

    if total <= 0:
        return {
            "K_raw": int(present.size),
            "masses": masses,
            "present_labels": present,
            "p": np.zeros_like(masses),
            "Neff_simpson": 0.0,
            "Neff_shannon": 0.0,
            "evenness": 0.0,
        }

    p = masses / total
    neff_simpson = 1.0 / float(np.sum(p * p))
    neff_shannon = float(np.exp(-np.sum(p * np.log(p + eps))))
    evenness = neff_simpson / float(present.size) if present.size > 0 else np.nan

    return {
        "K_raw": int(present.size),
        "masses": masses,
        "present_labels": present,
        "p": p,
        "Neff_simpson": neff_simpson,
        "Neff_shannon": neff_shannon,
        "evenness": evenness,
    }


def effective_label_count(labels: np.ndarray, region_mask: np.ndarray | None = None, eps: float = 1e-12) -> tuple[int, float, float, float]:
    """
    Effective label count based on region areas only.

    Returns
    -------
    (K_raw, Neff_simpson, Neff_shannon, ratio_simpson)
    """
    lab = np.asarray(labels)
    if region_mask is None:
        region_mask = np.ones_like(lab, dtype=bool)

    sel = lab[region_mask]
    sel = sel[sel > 0]
    if sel.size == 0:
        return 0, 0.0, 0.0, np.nan

    K = int(np.unique(sel).size)
    max_lab = int(sel.max())
    counts = np.bincount(sel, minlength=max_lab + 1)[1:]
    counts = counts[counts > 0]
    total = float(counts.sum())
    if total <= 0:
        return K, 0.0, 0.0, 0.0
    p = counts / total
    neff_simpson = 1.0 / float(np.sum(p * p))
    neff_shannon = float(np.exp(-np.sum(p * np.log(p + eps))))
    ratio_simpson = neff_simpson / float(K) if K > 0 else np.nan
    return K, neff_simpson, neff_shannon, ratio_simpson


# -------------------------- match quantifier ------------------------ #


def _extract_slice_for_label(labels: np.ndarray, label_id: int) -> tuple[tuple[slice, ...], np.ndarray]:
    slcs = ndi.find_objects(labels == label_id)
    if not slcs:
        # fallback to whole-image to avoid indexing errors
        return (slice(0, labels.shape[0]), slice(0, labels.shape[1])), labels == label_id
    return slcs[0], labels == label_id


def quantify_matches(
    detections: np.ndarray,
    index: int,
    matching_image: np.ndarray,
    *,
    pad_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
    """
    Build a per-crypt template from the masked region and run match_template.
    Returns (result, template, slice) for the given label index.
    """
    slc, binary = _extract_slice_for_label(detections, index)
    # Broadcast mask to channels if needed
    if matching_image.ndim == 2:
        crypt_binary = binary
    else:
        crypt_binary = np.stack([binary] * matching_image.shape[-1], axis=-1)

    template = np.where(crypt_binary, matching_image, np.zeros_like(matching_image))
    template = template[slc]

    result = match_template(matching_image, template, pad_input=pad_input)
    # Fill undefined areas near the template footprint
    result = inpaint.inpaint_biharmonic(result, detections == index, channel_axis=None)
    result = np.maximum(result, 0)
    result = minmax256(result)
    return result.astype(np.uint16), template, slc


def create_match_stack(best_crypts: np.ndarray, matching_image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[slice, ...]]]:
    match_arrays: list[np.ndarray] = []
    slcs: list[tuple[slice, ...]] = []
    max_label = int(np.max(best_crypts))
    for label_id in range(1, max_label + 1):
        if not np.any(best_crypts == label_id):
            continue
        result, _template, slc = quantify_matches(best_crypts, label_id, matching_image, pad_input=True)
        match_arrays.append(result)
        slcs.append(slc)
    return match_arrays, slcs


# ---------------------------- main API ----------------------------- #


@dataclass
class EffectiveCryptEstimation:
    neff_simpson: float
    neff_shannon: float
    k_raw: int
    evenness: float | float
    selected_labels_k: int
    debug_render_path: Optional[Path]


def estimate_effective_selected_crypt_count(
    *,
    best_crypts: Any,
    rfp_image: Any,
    dapi_image: Any | None = None,
    subject_name: str | Path | None = None,
    output_dir: str | Path | None = None,
    scoring_weights: Optional[Dict[str, float]] = None,
    save_debug: bool = False,
) -> EffectiveCryptEstimation:
    """
    Estimate the effective number of selected crypts using template matching
    and Simpson's diversity based on recovered peaks inside the selection.

    This function implements the exploration code shared from notebooks while
    keeping side effects (debug render) optional.
    """
    labels = _to_numpy(best_crypts).astype(np.int32)
    rfp = _to_numpy(rfp_image).astype(np.float32)
    if dapi_image is not None:
        dapi = _to_numpy(dapi_image).astype(np.float32)
    else:
        dapi = None

    # Score current selections to obtain quality weights (debug metadata)
    _filtered, debug_info = scoring_selector(
        labels,
        rfp,
        debug=False,
        max_regions=False,
        weights=scoring_weights or {
            "circularity": 0.15,
            "area": 0.25,
            "line_fit": 0.20,
            "red_intensity": 0.55,
        },
        return_details=True,
    )
    properties_df = debug_info.get("properties_df")

    # Build match stack across crypt selections
    match_stack, _slcs = create_match_stack(labels, rfp)
    if len(match_stack) == 0:
        return EffectiveCryptEstimation(
            neff_simpson=0.0,
            neff_shannon=0.0,
            k_raw=0,
            evenness=np.nan,
            selected_labels_k=0,
            debug_render_path=None,
        )

    result_array = np.asarray(match_stack)
    # Use quality_score as weights per shared code (lower is better; no inversion)
    if properties_df is not None and len(properties_df) >= result_array.shape[0]:
        weights_array = np.asarray(properties_df["quality_score"], dtype=np.float64)[: result_array.shape[0]]
    else:
        weights_array = np.ones((result_array.shape[0],), dtype=np.float64)

    log_results = np.log(np.maximum(result_array, 1))  # avoid log(0)
    weighted_log_sum = np.sum(weights_array[:, np.newaxis, np.newaxis] * log_results, axis=0)
    weight_sum = float(np.sum(weights_array)) if float(np.sum(weights_array)) > 0 else float(result_array.shape[0])
    collapsed_results = np.exp(weighted_log_sum / weight_sum).astype(np.uint16)

    # Morphological cleanup and Otsu selection
    _hats, clean, _troughs = caps(collapsed_results, 2, 20)
    try:
        otsu_thresh = threshold_otsu(clean)
    except Exception:
        otsu_thresh = float(np.nanmedian(clean))
    binary_clean = clean > otsu_thresh
    labeled_binary_clean, _ = ndi.label(binary_clean)
    labeled_overlap = np.where(labels > 0, labeled_binary_clean, 0)

    # Extract the set of labels inside the crypt selections
    unique_labels = np.unique(labeled_overlap)
    selected_labels_mask = np.isin(labeled_binary_clean, unique_labels)
    selected_labels = np.where(selected_labels_mask, labeled_binary_clean, 0)

    K, Neff_simpson, Neff_shannon, ratio_simpson = effective_label_count(selected_labels)

    debug_path: Optional[Path] = None
    if save_debug and output_dir is not None:
        # Build a simple RGB for background if DAPI present; else grayscale RFP
        if dapi is not None:
            rfp01 = (rfp - np.min(rfp)) / (np.max(rfp) - np.min(rfp) + 1e-12)
            dapi01 = (dapi - np.min(dapi)) / (np.max(dapi) - np.min(dapi) + 1e-12)
            base_rgb = np.stack([rfp01, np.zeros_like(rfp01), dapi01], axis=-1)
        else:
            base_rgb = np.stack([rfp, rfp, rfp], axis=-1)
            lo = float(np.nanmin(base_rgb))
            hi = float(np.nanmax(base_rgb))
            if hi > lo:
                base_rgb = (base_rgb - lo) / (hi - lo)

        medium_crypts = labels  # We only have the final set; use it for boundaries
        best_crypts = labels
        medium_bounds = find_boundaries(medium_crypts)
        best_bounds = find_boundaries(best_crypts)

        fig, axs = plt.subplots(3, 2, figsize=(20, 25))
        axs[0, 0].imshow(base_rgb)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        axs[0, 0].contour(medium_bounds, colors="g", linewidths=1.5)
        axs[0, 0].contour(best_bounds, colors="b", linewidths=1)

        axs[0, 1].imshow(collapsed_results, cmap="gray")
        axs[0, 1].set_title("Identified Crypt Regions (collapsed)")
        axs[0, 1].axis("off")
        axs[0, 1].contour(medium_bounds, colors="r", linewidths=2)
        axs[0, 1].contour(best_bounds, colors="g", linewidths=1)

        axs[1, 0].imshow(label2rgb(labeled_binary_clean), cmap="nipy_spectral")
        axs[1, 0].set_title("Cleaned Peaks (all)")
        axs[1, 0].axis("off")
        axs[1, 0].contour(medium_bounds, colors="r", linewidths=1)
        axs[1, 0].contour(best_bounds, colors="g", linewidths=1)

        axs[1, 1].imshow(label2rgb(selected_labels), cmap="nipy_spectral")
        axs[1, 1].set_title("Selected Peaks (within crypts)")
        axs[1, 1].axis("off")
        axs[1, 1].contour(best_bounds, colors="g", linewidths=1)

        # Text summary
        axs[2, 0].axis("off")
        summary_text = (
            f"K={K}\nNeff_simpson={Neff_simpson:.3f}\n"
            f"Neff_shannon={Neff_shannon:.3f}\nEvenness={ratio_simpson:.3f}"
        )
        axs[2, 0].text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=14)
        axs[2, 1].axis("off")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if subject_name is None:
            safe_name = "subject"
        else:
            safe_name = str(subject_name).replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")

        debug_path = out_dir / f"{safe_name}_effective_crypt_matches_debug.png"
        fig.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return EffectiveCryptEstimation(
        neff_simpson=float(Neff_simpson),
        neff_shannon=float(Neff_shannon),
        k_raw=int(K),
        evenness=float(ratio_simpson) if np.isfinite(ratio_simpson) else float("nan"),
        selected_labels_k=int(np.max(selected_labels)) if np.any(selected_labels) else 0,
        debug_render_path=debug_path,
    )


__all__ = [
    "effective_label_count_by_values",
    "effective_label_count",
    "quantify_matches",
    "create_match_stack",
    "estimate_effective_selected_crypt_count",
    "EffectiveCryptEstimation",
]

