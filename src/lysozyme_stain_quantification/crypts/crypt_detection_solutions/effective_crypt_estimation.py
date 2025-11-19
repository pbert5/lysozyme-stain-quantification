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
    from ..identify_potential_crypts_ import identify_potential_crypts
    from ..remove_edge_touching_regions_mod import remove_edge_touching_regions_sk
    from .crypt_identification_methodologies import (
        MorphologyParams,
        DEFAULT_MORPHOLOGY_PARAMS,
    )
    from ...utils.debug_image_saver import DebugImageSession
except Exception:  # pragma: no cover - fallback path
    # Package import name
    from lysozyme_stain_quantification.crypts.scoring_selector_mod import scoring_selector
    from lysozyme_stain_quantification.crypts.identify_potential_crypts_ import identify_potential_crypts
    from lysozyme_stain_quantification.crypts.remove_edge_touching_regions_mod import remove_edge_touching_regions_sk
    from lysozyme_stain_quantification.crypts.crypt_detection_solutions.crypt_identification_methodologies import (
        MorphologyParams,
        DEFAULT_MORPHOLOGY_PARAMS,
    )
    from lysozyme_stain_quantification.utils.debug_image_saver import DebugImageSession


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


def _clip_line_to_bounds(width: int, height: int, x0: float, y0: float, vx: float, vy: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return two endpoints of the infinite line clipped to image bounds or None.

    The line passes through (x0, y0) with direction (vx, vy). Bounds are
    [0, width-1] x [0, height-1].
    """
    eps = 1e-12
    xs: list[float] = []
    ys: list[float] = []

    # Intersect with x = 0 and x = W-1
    if abs(vx) > eps:
        for x_edge in (0.0, float(width - 1)):
            t = (x_edge - x0) / vx
            y = y0 + t * vy
            if 0.0 <= y <= float(height - 1):
                xs.append(x_edge)
                ys.append(y)

    # Intersect with y = 0 and y = H-1
    if abs(vy) > eps:
        for y_edge in (0.0, float(height - 1)):
            t = (y_edge - y0) / vy
            x = x0 + t * vx
            if 0.0 <= x <= float(width - 1):
                xs.append(x)
                ys.append(y_edge)

    # Deduplicate close points
    pts: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if not pts or (abs(pts[-1][0] - x) > 1e-6 or abs(pts[-1][1] - y) > 1e-6):
            pts.append((x, y))

    if len(pts) < 2:
        return None
    if len(pts) == 2:
        return pts[0], pts[1]
    # Pick farthest two points
    max_d = -1.0
    best_pair = (pts[0], pts[1])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if d2 > max_d:
                max_d = d2
                best_pair = (pts[i], pts[j])
    return best_pair


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
    base_labels: Any | None = None,
    rfp_image: Any,
    dapi_image: Any | None = None,
    blob_size_um: float | None = None,
    microns_per_px: float | None = None,
    blob_size_px: int | None = None,
    subject_name: str | Path | None = None,
    output_dir: str | Path | None = None,
    scoring_weights: Optional[Dict[str, float]] = None,
    save_debug: bool = False,
    expansion_scale: float = 0.25,
    debug_recorder: DebugImageSession | None = None,
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

    def _capture(image: Any, stage: str, *, description: str | None = None) -> None:
        if debug_recorder is not None:
            debug_recorder.save_image(image, stage, source="effective_crypt_estimation", description=description)

    _capture(rfp, "rfp_input")
    if dapi is not None:
        _capture(dapi, "dapi_input")
    _capture(labels, "best_crypts_labels")

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

    # Build a broader medium set strictly by re-scoring the candidate labels
    # with a larger max_regions (no expansion). If base_labels is not provided,
    # fall back to using the best set itself.
    base = labels if base_labels is None else _to_numpy(base_labels).astype(np.int32)
    properties_df_med = properties_df
    selected_order_med = None
    try:
        scores = effective_label_count_by_values(
            labels=base,
            values=rfp,
            region_mask=base > 0,
            value_agg="sum_positive",
        )
        medium_target = int(max(1, round(scores.get("Neff_simpson", 0) or 0)))
        medium_crypts, med_debug = scoring_selector(
            base,
            rfp,
            debug=False,
            max_regions=medium_target,
            weights=scoring_weights or {
                "circularity": 0.15,
                "area": 0.25,
                "line_fit": 0.20,
                "red_intensity": 0.55,
            },
            return_details=True,
        )
        properties_df_med = med_debug.get("properties_df", properties_df)
        selected_order_med = med_debug.get("selected_labels")
    except Exception:
        medium_crypts = labels
        properties_df_med = properties_df
        selected_order_med = None

    _capture(medium_crypts, "medium_crypts_labels")

    # Build match stack across medium selections
    match_stack, _slcs = create_match_stack(medium_crypts, rfp)
    if debug_recorder is not None:
        for idx, match_img in enumerate(match_stack):
            _capture(match_img, f"match_stack_{idx:03d}")
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
    if result_array.ndim == 3:
        _capture(np.max(result_array, axis=0), "match_stack_max_projection")
        _capture(np.mean(result_array, axis=0), "match_stack_mean_projection")
    # Emphasize better quality (lower quality_score => higher weight)
    if properties_df_med is not None and len(properties_df_med) > 0:
        if selected_order_med is not None:
            pdf = properties_df_med.set_index("label_id")
            scores_arr = []
            for old_id in selected_order_med:
                if old_id in pdf.index:
                    scores_arr.append(float(pdf.loc[old_id]["quality_score"]))
            s = np.asarray(scores_arr, dtype=np.float64)
        else:
            s = np.asarray(properties_df_med["quality_score"], dtype=np.float64)
        if s.size >= result_array.shape[0] and np.all(np.isfinite(s)):
            s = s[: result_array.shape[0]]
            s_min = float(np.min(s))
            s_max = float(np.max(s))
            eps_w = 1e-9
            weights_array = (s_max - s) + eps_w if s_max > s_min else np.ones_like(s)
        else:
            weights_array = np.ones((result_array.shape[0],), dtype=np.float64)
    else:
        weights_array = np.ones((result_array.shape[0],), dtype=np.float64)

    # Normalize weights and compute weighted geometric mean
    ws = float(np.sum(weights_array))
    if ws <= 0:
        weights_array = np.ones_like(weights_array)
        ws = float(np.sum(weights_array))
    weights_array = weights_array / ws
    if weights_array.size > 0:
        weights_vis = np.tile(weights_array[np.newaxis, :], (max(1, min(50, weights_array.size)), 1))
        _capture(weights_vis, "match_stack_weights")
    log_results = np.log(np.maximum(result_array, 1))  # avoid log(0)
    weighted_log_sum = np.sum(weights_array[:, np.newaxis, np.newaxis] * log_results, axis=0)
    collapsed_results = np.exp(weighted_log_sum).astype(np.uint16)
    _capture(collapsed_results, "collapsed_results")

    # Morphological cleanup and Otsu selection
    # Derive cap radii from blob size, scaled by expansion_scale.
    def _resolve_blob_size_px(
        blob_size_px: int | None,
        blob_size_um: float | None,
        microns_per_px: float | None,
        fallback_labels: np.ndarray,
    ) -> int:
        if blob_size_px is not None and blob_size_px > 0:
            return int(blob_size_px)
        if blob_size_um is not None and microns_per_px is not None and microns_per_px > 0:
            return max(1, int(round(float(blob_size_um) / float(microns_per_px))))
        # Heuristic from labels: median equivalent diameter
        lbl = np.asarray(fallback_labels)
        if np.any(lbl > 0):
            max_lab = int(lbl.max())
            counts = np.bincount(lbl.ravel(), minlength=max_lab + 1)[1:]
            counts = counts[counts > 0]
            if counts.size > 0:
                eq_radius = np.sqrt(counts / np.pi)
                eq_diam = 2.0 * eq_radius
                return int(max(1, round(float(np.median(eq_diam)))))
        return 40  # safe default

    eff_blob_px = _resolve_blob_size_px(blob_size_px, blob_size_um, microns_per_px, base)
    scaled_blob = max(1, int(round(eff_blob_px * float(expansion_scale))))
    # Empirical fractions: small ≈ 0.1×, big ≈ 0.5× of diameter
    small_r = max(1, int(round(0.10 * scaled_blob)))
    big_r = max(small_r + 1, int(round(0.50 * scaled_blob)))

    _hats, clean, _troughs = caps(collapsed_results, small_r, big_r)
    try:
        otsu_thresh = threshold_otsu(clean)
    except Exception:
        otsu_thresh = float(np.nanmedian(clean))
    binary_clean = clean > otsu_thresh
    _capture(clean, "collapsed_caps_clean", description=f"otsu_threshold={otsu_thresh:.3f}")
    _capture(binary_clean.astype(np.float32), "otsu_binary_mask")
    _output = ndi.label(binary_clean)
    if isinstance(_output, tuple) and len(_output) == 2:
            labeled_binary_clean, _ = _output
    else:
        labeled_binary_clean, _ = [np.zeros_like(clean, dtype=np.int32), 0]
    labeled_overlap = np.where(labels > 0, labeled_binary_clean, 0)
    _capture(labeled_binary_clean, "labeled_binary_clean")

    # Extract peaks that overlap best crypts, then restrict to medium crypt region for counts
    unique_labels = np.unique(labeled_overlap)
    selected_labels_mask = np.isin(labeled_binary_clean, unique_labels)
    selected_labels = np.where(selected_labels_mask, labeled_binary_clean, 0)
    selected_intersection = np.where(medium_crypts > 0, selected_labels, 0)
    _capture(selected_labels_mask.astype(np.float32), "selected_labels_mask")
    _capture(selected_labels, "selected_labels")
    _capture(selected_intersection, "selected_labels_intersection")

    best_bounds = find_boundaries(labels)
    medium_bounds = find_boundaries(medium_crypts)
    _capture(best_bounds.astype(np.float32), "best_crypt_bounds")
    _capture(medium_bounds.astype(np.float32), "medium_crypt_bounds")

    K, Neff_simpson, Neff_shannon, ratio_simpson = effective_label_count(selected_intersection)

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

        best_crypts_local = labels

        fig, axs = plt.subplots(3, 2, figsize=(20, 25))
        axs[0, 0].imshow(base_rgb)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        # Draw medium (red, thicker) first, then best (blue, thinner)
        axs[0, 0].contour(medium_bounds, colors="r", linewidths=2.5)
        axs[0, 0].contour(best_bounds, colors="b", linewidths=1.2)
        # Draw intensity-weighted principal axis through global RFP COM within best crypts
        try:
            mask = best_crypts_local > 0
            if np.any(mask):
                yy, xx = np.nonzero(mask)
                vals = rfp[mask].astype(np.float64)
                # Guard against zero weights
                vals = np.maximum(vals, 1e-9)
                wsum = float(np.sum(vals))
                x_mean = float(np.sum(xx * vals) / wsum)
                y_mean = float(np.sum(yy * vals) / wsum)
                # Weighted covariance for principal direction
                x_c = xx - x_mean
                y_c = yy - y_mean
                cov_xx = float(np.sum(vals * x_c * x_c) / wsum)
                cov_xy = float(np.sum(vals * x_c * y_c) / wsum)
                cov_yy = float(np.sum(vals * y_c * y_c) / wsum)
                cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
                eigvals, eigvecs = np.linalg.eigh(cov)
                v = eigvecs[:, np.argmax(eigvals)]  # principal axis
                # Clip the infinite line to the image bounds
                seg = _clip_line_to_bounds(
                    width=base_rgb.shape[1], height=base_rgb.shape[0], x0=x_mean, y0=y_mean, vx=float(v[0]), vy=float(v[1])
                )
                if seg is not None:
                    (x1, y1), (x2, y2) = seg
                    axs[0, 0].plot([x1, x2], [y1, y2], color="y", linestyle="--", linewidth=2.0)
        except Exception:
            pass

        axs[0, 1].imshow(collapsed_results, cmap="gray")
        axs[0, 1].set_title("Identified Crypt Regions (collapsed)")
        axs[0, 1].axis("off")
        axs[0, 1].contour(medium_bounds, colors="r", linewidths=2.5)
        axs[0, 1].contour(best_bounds, colors="b", linewidths=1.2)

        axs[1, 0].imshow(label2rgb(labeled_binary_clean), cmap="nipy_spectral")
        axs[1, 0].set_title("Cleaned Peaks (all)")
        axs[1, 0].axis("off")
        axs[1, 0].contour(medium_bounds, colors="r", linewidths=2.5)
        axs[1, 0].contour(best_bounds, colors="b", linewidths=1.2)

        axs[1, 1].imshow(label2rgb(selected_labels), cmap="nipy_spectral")
        axs[1, 1].set_title("Selected Peaks (overlapping best)")
        axs[1, 1].axis("off")
        axs[1, 1].contour(medium_bounds, colors="r", linewidths=2.5)
        axs[1, 1].contour(best_bounds, colors="b", linewidths=1.2)

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
        selected_labels_k=int(np.max(selected_intersection)) if np.any(selected_intersection) else 0,
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
