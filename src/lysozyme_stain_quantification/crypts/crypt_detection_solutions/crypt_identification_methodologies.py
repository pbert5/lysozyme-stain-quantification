from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np #TODO: swap with dask
import tifffile as tiff
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as ndi_label
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.measure import label as sk_label
from skimage.morphology import (
    binary_erosion,
    binary_opening,
    disk,
    dilation,
    local_maxima,
    opening,
    remove_small_objects,
    white_tophat,
)
from skimage.segmentation import expand_labels, watershed
from skimage.util import invert
try:
    from src.lysozyme_stain_quantification.utils.remove_artifacts import remove_rectangles
except ImportError:
    from lysozyme_stain_quantification.utils.remove_artifacts import remove_rectangles


from ..scoring_selector_mod import scoring_selector

from ...utils.debug_image_saver import DebugImageSession


# ---------------------------- utilities ---------------------------- #


def to_float01(img: np.ndarray) -> np.ndarray:
    """Return an array scaled to [0, 1] float without modifying NaNs/Inf."""
    if img.dtype in (np.float32, np.float64):
        return np.clip(img, 0.0, 1.0)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return np.clip(img.astype(np.float32) / float(info.max), 0.0, 1.0)
    arr = img.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def minmax(x: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] (safe if constant)."""
    x = x.astype(np.float32, copy=False)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


# ---------------------------- scale metadata ---------------------------- #


@dataclass(frozen=True)
class MorphologyParams:
    """Per-subject morphology tuning recorded in pixels."""

    crypt_radius_px: int
    intervilli_distance_px: int
    salt_and_pepper_noise_size: Optional[int] = None
    peak_smoothing_sigma: Optional[float] = None
    microns_per_pixel: Optional[float] = None  # Reserved for future physical-scale use.


DEFAULT_MORPHOLOGY_PARAMS = MorphologyParams(
    crypt_radius_px=30,
    intervilli_distance_px=40,
    salt_and_pepper_noise_size=3,
    peak_smoothing_sigma=None,
    microns_per_pixel=None,
)

_SAMPLE_PARAM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Jej-2": dict(crypt_radius_px=30),
    "Jej-3": dict(crypt_radius_px=30),
    "G3FR": dict(crypt_radius_px=40),
    "G3FR_sep": dict(crypt_radius_px=40),
}


def get_morphology_params(subject: str, **overrides: Any) -> MorphologyParams:
    """
    Return morphology parameters for the provided subject.

    Values are stored in pixels for now. When micron metadata is available we can
    convert it here before returning the configuration.
    """
    params = DEFAULT_MORPHOLOGY_PARAMS
    subject_overrides = _SAMPLE_PARAM_OVERRIDES.get(subject)
    if subject_overrides:
        params = replace(params, **subject_overrides)
    if overrides:
        params = replace(params, **overrides)
    return params


def preprocess_for_caps(image: np.ndarray, salt_and_pepper_noise_size: Optional[int]) -> np.ndarray:
    """
    Normalize the image and optionally remove salt-and-pepper noise.

    Historically we applied a white top-hat cleanup before cap decomposition; this
    keeps that option configurable per subject.
    """
    image = to_float01(image)
    if salt_and_pepper_noise_size is None or salt_and_pepper_noise_size <= 0:
        return image

    radius = max(1, int(round(salt_and_pepper_noise_size)))
    footprint = disk(radius)
    cleaned = image - white_tophat(image, footprint)
    return cleaned


# ---------------------------- morphology helpers ---------------------------- #


def _caps_preprocess(
    image: np.ndarray,
    small_r: int,
    big_r: int,
    *,
    return_details: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared preprocessing for cap-style morphology computations."""
    image = to_float01(image)
    big_dilation = dilation(image, disk(big_r))
    small_dilation = dilation(image, disk(small_r))
    hats = minmax(big_dilation - small_dilation)
    if return_details:
        return image, hats, big_dilation, small_dilation
    return image, hats


def caps(image: np.ndarray, small_r: int, big_r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contrast-adaptive cap decomposition returning (hats, clean, troughs)."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)

    clean = image_eq - np.minimum(image_eq, hats)
    clean = minmax(clean)

    troughs = np.maximum(image_eq, hats) - image_eq
    troughs = minmax(troughs)
    return hats, clean, troughs


def caps_clean(image: np.ndarray, small_r: int, big_r: int) -> np.ndarray:
    """Return only the 'clean' component of the cap decomposition."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)
    clean = image_eq - np.minimum(image_eq, hats)
    return minmax(clean)


def caps_trough(image: np.ndarray, small_r: int, big_r: int) -> np.ndarray:
    """Return only the 'troughs' component of the cap decomposition."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)
    troughs = np.maximum(image_eq, hats) - image_eq
    return minmax(troughs)


def caps_clean_troughs(image: np.ndarray, small_r: int, big_r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return both clean and trough components while avoiding extra work."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)

    clean = image_eq - np.minimum(image_eq, hats)
    clean = minmax(clean)

    troughs = np.maximum(image_eq, hats) - image_eq
    troughs = minmax(troughs)
    return clean, troughs


def _caps_clean_troughs_debug(
    image: np.ndarray,
    small_r: int,
    big_r: int,
    *,
    stage_prefix: str,
    source: str,
    debug_recorder: DebugImageSession | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Variant of caps_clean_troughs that records intermediate components when requested."""
    if debug_recorder is None:
        return caps_clean_troughs(image, small_r, big_r)

    image_eq, hats, big_dilation, small_dilation = _caps_preprocess(
        image,
        small_r,
        big_r,
        return_details=True,
    )
    clean = image_eq - np.minimum(image_eq, hats)
    clean = minmax(clean)
    troughs = np.maximum(image_eq, hats) - image_eq
    troughs = minmax(troughs)

    debug_recorder.save_image(image_eq, f"{stage_prefix}_caps_input", source=source)
    debug_recorder.save_image(big_dilation, f"{stage_prefix}_big_dilation", source=source)
    debug_recorder.save_image(small_dilation, f"{stage_prefix}_small_dilation", source=source)
    debug_recorder.save_image(hats, f"{stage_prefix}_caps_hats", source=source)
    debug_recorder.save_image(clean, f"{stage_prefix}_caps_clean", source=source)
    debug_recorder.save_image(troughs, f"{stage_prefix}_caps_troughs", source=source)

    return clean, troughs


def show_caps(image: np.ndarray, small_r: int, big_r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visualize the cap decomposition for debugging."""
    hats, clean, troughs = caps(image, small_r, big_r)

    fig, ax = plt.subplots(1, 4, figsize=(30, 5))
    for axis in ax:
        axis.axis("off")
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(hats, cmap="gray")
    ax[2].imshow(clean, cmap="gray")
    ax[3].imshow(troughs, cmap="gray")
    fig.colorbar(ax[1].images[0], ax=ax[1])
    fig.colorbar(ax[2].images[0], ax=ax[2])
    fig.colorbar(ax[3].images[0], ax=ax[3])
    fig.tight_layout()
    return hats, clean, troughs


def identify_crypt_seeds(
    crypt_img: NDArray,
    tissue_image: NDArray,
    blob_size_px: Optional[int] = None,
) -> NDArray:
    """Legacy seed generator kept for reference."""
    del blob_size_px  # legacy placeholder
    tissue_clean, tissue_troughs = caps_clean_troughs(tissue_image, 1, 40)
    crypt_clean = caps_clean(crypt_img, 2, 40)

    thinned_crypts = np.maximum(crypt_clean - tissue_clean, 0)
    split_crypts = np.maximum(crypt_clean - tissue_troughs, 0)

    good_crypts = minmax(opening(split_crypts * thinned_crypts, footprint=disk(5)) ** 0.5)
    distance = tissue_troughs - good_crypts
    maxi = local_maxima(good_crypts)
    crypt_seeds = watershed(distance, markers=maxi, mask=tissue_troughs < good_crypts)
    labeled: np.ndarray = sk_label(crypt_seeds > 0)  # type: ignore
    return labeled


def limited_expansion(
    crypt_img: NDArray,
    tissue_image: NDArray,
    blob_size_px: Optional[int] = None,
) -> NDArray:
    """Limit vertical expansion of crypts by suppressing tissue spillover."""
    del blob_size_px  # not used but kept for interface parity
    outer_troughs = caps_trough(minmax(tissue_image), 10, 50)
    outer_troughs = minmax(outer_troughs)

    adjusted = np.maximum(crypt_img - outer_troughs, 0)
    crypt_max = binary_opening(local_maxima(adjusted, indices=False), footprint=disk(1)).astype(bool)
    tissue_max = binary_opening(local_maxima(outer_troughs), disk(2))

    alt = invert(dilation(minmax(adjusted), footprint=disk(2)) + minmax(outer_troughs))
    spread = watershed(image=alt, markers=np.maximum(crypt_max * 1, tissue_max * 2))
    return spread


def identify_crypt_seeds_new(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    *,
    params: Optional[MorphologyParams] = None,
    debug_recorder: DebugImageSession | None = None,
) -> np.ndarray:
    """Newer seed logic that prefers solid crypts."""
    params = params or DEFAULT_MORPHOLOGY_PARAMS

    crypt_img = preprocess_for_caps(crypt_img, params.salt_and_pepper_noise_size)
    tissue_image = preprocess_for_caps(equalize_adapthist(tissue_image), params.salt_and_pepper_noise_size)

    def _capture(image: np.ndarray, stage: str) -> None:
        if debug_recorder is not None:
            debug_recorder.save_image(image, stage, source="identify_crypt_seeds_new")

    _capture(crypt_img, "crypt_preprocessed")
    _capture(tissue_image, "tissue_preprocessed")

    max_expansion = max(1, int(round(params.crypt_radius_px)))
    tissue_clean, tissue_troughs = _caps_clean_troughs_debug(
        tissue_image,
        1,
        max_expansion,
        stage_prefix="tissue",
        source="identify_crypt_seeds_new",
        debug_recorder=debug_recorder,
    )
    crypt_clean, crypt_troughs = _caps_clean_troughs_debug(
        crypt_img,
        2,
        max_expansion,
        stage_prefix="crypt",
        source="identify_crypt_seeds_new",
        debug_recorder=debug_recorder,
    )

    _capture(tissue_clean, "tissue_clean")
    _capture(tissue_troughs, "tissue_troughs")
    _capture(crypt_clean, "crypt_clean")
    _capture(crypt_troughs, "crypt_troughs")

    thinned_crypts = np.maximum(crypt_clean - tissue_clean, 0)
    split_crypts = np.maximum(crypt_clean - tissue_troughs, 0)
    _capture(thinned_crypts, "thinned_crypts")
    _capture(split_crypts, "split_crypts")

    combined = split_crypts * thinned_crypts
    _capture(combined, "split_times_thinned")

    opened = opening(combined, footprint=disk(5))
    _capture(opened, "opened_split_times_thinned")

    good_crypts = minmax(opened ** 0.5)
    _capture(good_crypts, "good_crypts")
    if params.peak_smoothing_sigma and params.peak_smoothing_sigma > 0:
        good_crypts = gaussian_filter(good_crypts, params.peak_smoothing_sigma)
        good_crypts = minmax(good_crypts)
        _capture(good_crypts, "good_crypts_smoothed")

    distance = tissue_troughs - good_crypts
    maxi = local_maxima(good_crypts)
    _capture(distance, "distance_image")
    _capture(maxi.astype(np.float32), "local_maxima_mask")

    watershed_mask = tissue_troughs < good_crypts
    _capture(watershed_mask.astype(np.float32), "watershed_mask")
    seeds = watershed(distance, markers=maxi, mask=watershed_mask)
    _capture(seeds, "watershed_labels")
    labeled: np.ndarray = sk_label(seeds > 0)  # type: ignore
    _capture(labeled, "seed_labels")
    return labeled


def identify_potential_crypts_old_like(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    blob_size_px: Optional[int] = None,
    external_crypt_seeds: Optional[np.ndarray] = None,
    params: Optional[MorphologyParams] = None,
    debug_recorder: DebugImageSession | None = None,
) -> np.ndarray:
    """Original watershed skeleton with optional external seeds."""
    crypt_img = to_float01(crypt_img)
    tissue_image = to_float01(tissue_image)

    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Image shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    params = params or DEFAULT_MORPHOLOGY_PARAMS
    effective_blob = float(blob_size_px) if blob_size_px else float(params.crypt_radius_px)
    erosion_dim = _odd(max(3, int(round(effective_blob / 10.0))))
    erosion_footprint = np.ones((erosion_dim, erosion_dim), dtype=bool)

    if external_crypt_seeds is None:
        crypt_seeds_bool = crypt_img > np.minimum(tissue_image, crypt_img)
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seeds_bool = binary_erosion(crypt_seeds_bool, footprint=erosion_footprint)
        crypt_seeds_bool = remove_small_objects(crypt_seeds_bool, min_size=min_region_area)
        labeled_diff_r: np.ndarray
        labeled_diff_r, _ = ndi_label(crypt_seeds_bool)  # type: ignore
    else:
        external_crypt_seeds = external_crypt_seeds.astype(np.int32)
        crypt_seed_mask = external_crypt_seeds > 0
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seed_mask = remove_small_objects(crypt_seed_mask, min_size=min_region_area)
        labeled_diff_r = sk_label(crypt_seed_mask)  # type: ignore

    abs_diff = np.maximum(tissue_image - crypt_img, 0)
    mask_gt_red = abs_diff > crypt_img

    erosion_kernel_size = max(3, int(round(effective_blob * 0.15)))
    erosion_kernel_size = erosion_kernel_size if erosion_kernel_size % 2 == 0 else erosion_kernel_size + 1
    cv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    mask_u8 = (mask_gt_red.astype(np.uint8) * 255)
    erosion_iterations = max(1, int(round(effective_blob / 20.0)))
    mask_eroded_u8 = cv2.erode(mask_u8, cv_kernel, iterations=erosion_iterations)
    tissue_eroded = mask_eroded_u8.astype(bool)

    combined_labels = np.zeros_like(labeled_diff_r, dtype=int)
    combined_labels[tissue_eroded] = 2
    combined_labels[labeled_diff_r > 0] = 1
    if debug_recorder is not None:
        debug_recorder.save_image(combined_labels, "old_like_combined_labels", source="identify_potential_crypts_old_like")

    if params.intervilli_distance_px:
        expand_distance = max(1, int(round(params.intervilli_distance_px)))
    else:
        expand_distance = max(1, int(round(effective_blob * 2.5)))
    expanded_labels = expand_labels(combined_labels, distance=expand_distance)
    if debug_recorder is not None:
        debug_recorder.save_image(expanded_labels, "old_like_expanded_labels", source="identify_potential_crypts_old_like")

    reworked = np.zeros_like(expanded_labels, dtype=np.int32)
    reworked[expanded_labels == 2] = 1
    mask_copy = (expanded_labels != 2) & (labeled_diff_r != 0)
    reworked[mask_copy] = labeled_diff_r[mask_copy] + 1

    mask_ws = expanded_labels > 0

    elevation = minmax(distance_transform_edt(combined_labels == 2)) - minmax(  # type: ignore
        distance_transform_edt(combined_labels == 1)  # type: ignore
    )

    ws_labels = watershed(elevation, markers=reworked, mask=mask_ws).copy()
    ws_labels[ws_labels == 1] = 0
    ws_labels[ws_labels > 1] -= 1
    ws_labels = ws_labels.astype(np.int32)
    if debug_recorder is not None:
        debug_recorder.save_image(ws_labels, "old_like_watershed", source="identify_potential_crypts_old_like")
    return ws_labels


def _seed_health_metrics(seed_labels: np.ndarray) -> Dict[str, Any]:
    """Compute basic metrics to judge seed quality."""
    if seed_labels is None or seed_labels.size == 0:
        return dict(n_labels=0, coverage=0.0, mean_area=0.0)
    n_labels = int(seed_labels.max())
    coverage = float(np.count_nonzero(seed_labels)) / seed_labels.size
    if n_labels > 0:
        areas = np.bincount(seed_labels.ravel())[1:]
        mean_area = float(areas.mean()) if areas.size else 0.0
    else:
        mean_area = 0.0
    return dict(n_labels=n_labels, coverage=coverage, mean_area=mean_area)


def _score_label_set(
    labels: Optional[np.ndarray],
    raw_img: np.ndarray,
    *,
    debug: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a set of labels using the scoring selector.

    Returns a tuple of (evaluation_score, summary_dict) where higher scores are better.
    """
    summary: Dict[str, Any] = {}
    if labels is None or labels.size == 0:
        summary.update(region_count=0, mean_quality_score=None, evaluation_score=None, reason="empty")
        return float("-inf"), summary

    labels = np.asarray(labels)
    region_count = int(labels.max())
    summary["region_count"] = region_count

    if region_count == 0:
        summary.update(mean_quality_score=None, evaluation_score=None, reason="no_regions")
        return float("-inf"), summary

    try:
        _, scoring_debug = scoring_selector(
            labels,
            raw_img,
            debug=debug,
            max_regions=False,
            return_details=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        summary.update(mean_quality_score=None, evaluation_score=None, reason=f"scoring_failed:{exc!r}")
        return float("-inf"), summary

    properties_df = scoring_debug.get("properties_df")
    if properties_df is None or len(properties_df) == 0:
        summary.update(mean_quality_score=None, evaluation_score=None, reason="no_properties")
        return float("-inf"), summary

    mean_quality = float(properties_df["quality_score"].mean())
    evaluation_score = -mean_quality  # Lower quality_score is better; invert so higher is better.
    summary.update(
        mean_quality_score=mean_quality,
        evaluation_score=evaluation_score,
    )

    if debug:
        summary["quality_scores_sample"] = properties_df["quality_score"].tolist()[:10]

    return evaluation_score, summary


def identify_potential_crypts_hybrid(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    blob_size_px: Optional[int] = None,
    use_new_seeds: bool = True,
    auto_fallback: bool = True,
    min_seed_count: int = 10,
    min_coverage: float = 0.002,
    debug: bool = False,
    params: Optional[MorphologyParams] = None,
    debug_recorder: DebugImageSession | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Hybrid pipeline that evaluates both seed strategies and keeps the better result.
      1) Generate new seeds (optionally) and always run the legacy seed body.
      2) Score both label sets using the quality selector.
      3) Select the higher scoring segmentation unless auto-fallback is disabled.
    """
    crypt_img = to_float01(crypt_img)
    tissue_image = to_float01(tissue_image)

    params = params or DEFAULT_MORPHOLOGY_PARAMS
    if blob_size_px is None:
        blob_size_px = params.crypt_radius_px

    dbg: Dict[str, Any] = {}

    new_seeds = (
        identify_crypt_seeds_new(crypt_img, tissue_image, params=params, debug_recorder=debug_recorder)
        if use_new_seeds
        else None
    )
    new_metrics = _seed_health_metrics(
        new_seeds if new_seeds is not None else np.zeros_like(crypt_img, dtype=np.int32)
    )
    dbg["new_seed_metrics"] = new_metrics

    metrics_meet_threshold = (new_metrics["n_labels"] >= min_seed_count) and (new_metrics["coverage"] >= min_coverage)
    dbg["seed_metrics_meet_threshold"] = bool(metrics_meet_threshold)

    candidate_labels: Dict[str, np.ndarray] = {}
    candidate_summaries: Dict[str, Dict[str, Any]] = {}

    if use_new_seeds and new_seeds is not None:
        candidate_labels["new_seeded"] = identify_potential_crypts_old_like(
            crypt_img,
            tissue_image,
            blob_size_px=blob_size_px,
            external_crypt_seeds=new_seeds,
            params=params,
            debug_recorder=debug_recorder,
        )

    candidate_labels["legacy_seeded"] = identify_potential_crypts_old_like(
        crypt_img,
        tissue_image,
        blob_size_px=blob_size_px,
        external_crypt_seeds=None,
        params=params,
        debug_recorder=debug_recorder,
    )

    if debug_recorder is not None:
        if "legacy_seeded" in candidate_labels:
            debug_recorder.save_image(
                candidate_labels["legacy_seeded"], "legacy_seed_labels", source="identify_potential_crypts_hybrid"
            )
        if "new_seeded" in candidate_labels:
            debug_recorder.save_image(
                candidate_labels["new_seeded"], "new_seed_labels", source="identify_potential_crypts_hybrid"
            )

    selected_key = "legacy_seeded"
    best_score = float("-inf")

    if auto_fallback:
        for key, label_img in candidate_labels.items():
            if key == "new_seeded" and not metrics_meet_threshold:
                summary = dict(
                    region_count=int(label_img.max()),
                    mean_quality_score=None,
                    evaluation_score=float("-inf"),
                    reason="seed_metrics_below_threshold",
                )
                candidate_summaries[key] = summary
                continue
            score, summary = _score_label_set(label_img, crypt_img, debug=debug)
            candidate_summaries[key] = summary
            if score > best_score:
                best_score = score
                selected_key = key
    else:
        if "new_seeded" in candidate_labels and use_new_seeds and metrics_meet_threshold:
            selected_key = "new_seeded"
        candidate_summaries = {selected_key: dict(evaluation_score=None, reason="auto_fallback_disabled")}
        best_score = float("nan")

    labels = candidate_labels[selected_key]
    if debug_recorder is not None:
        debug_recorder.save_image(labels, "selected_label_set", source="identify_potential_crypts_hybrid")

    dbg["selected_candidate"] = selected_key
    dbg["candidate_summaries"] = candidate_summaries
    dbg["using_new_seeds"] = selected_key == "new_seeded"

    if debug:
        dbg["labels_max"] = int(labels.max())
        dbg["labels_coverage"] = float(np.count_nonzero(labels)) / labels.size
        dbg["best_score"] = best_score

    return labels, dbg


# ---------------------------- driver ---------------------------- #


def build_image_sets() -> list[Tuple[str, np.ndarray, np.ndarray]]:
    """Load source images and return [(subject, crypt_channel, tissue_channel), ...]."""
    jej2 = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c2.tif"
    )
    jej2_tissue = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c1.tif"
    )
    jej3 = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c2.tif"
    )
    jej3_tissue = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c1.tif"
    )
    g3fr_rgb = tiff.imread("/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2.tif")
    g3fr_sep_rfp = tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_RFP.tif"
    )
    g3fr_sep_dapi = tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_DAPI.tif"
    )

    evil_g3fr = remove_rectangles(g3fr_rgb)

    return [
        ("Jej-2", jej2[..., 1], jej2_tissue[..., 2]),
        ("Jej-3", jej3[..., 1], jej3_tissue[..., 2]),
        ("G3FR", evil_g3fr[..., 0], evil_g3fr[..., 2]),
        ("G3FR_sep", g3fr_sep_rfp[..., 0], g3fr_sep_dapi[..., 2]),
    ]


def _load_reference_image() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single reference RGB image (crypt, crypt channel, dapi channel)."""
    image = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2.tif"
    )
    crypt_channel = minmax(image[:, :, 1])
    dapi_channel = minmax(image[:, :, 2])
    first_channel = minmax(image[:, :, 0])
    return first_channel, crypt_channel, dapi_channel


def main() -> None:

    list_of_images_to_try = build_image_sets()
    crypt_img = minmax(list_of_images_to_try[0][1])
    tissue_image = minmax(list_of_images_to_try[0][2])

    _first_channel, _crypts, _dapi = _load_reference_image()
    del _first_channel, _crypts, _dapi  # kept for parity with notebook but unused

    scale = 5
    width = 4
    fig, ax = plt.subplots(len(list_of_images_to_try), width, figsize=(scale * width, len(list_of_images_to_try) * scale))

    for i, (subject, crypt, tissue) in enumerate(list_of_images_to_try):
        crypt = minmax(crypt)
        tissue = minmax(tissue)
        params = get_morphology_params(subject)

        ax[i, 0].imshow(np.stack([crypt, np.zeros_like(crypt), tissue], axis=-1))
        ax[i, 0].axis("off")
        ax[i, 0].set_title(f"{subject} - Crypt")

        old_labels = identify_potential_crypts_old_like(crypt, tissue, params=params)
        ax[i, 1].imshow(label2rgb(old_labels))
        ax[i, 1].axis("off")
        ax[i, 1].set_title(f"{subject} - Old identify potential crypts")

        hybrid_labels, _ = identify_potential_crypts_hybrid(crypt, tissue, params=params)
        ax[i, 2].imshow(label2rgb(hybrid_labels), cmap="nipy_spectral")
        ax[i, 2].axis("off")
        ax[i, 2].set_title(f"{subject} - Hybrid identify potential crypts")

        new_seeds = identify_crypt_seeds_new(crypt, tissue, params=params)
        ax[i, 3].imshow(label2rgb(new_seeds), cmap="nipy_spectral")
        ax[i, 3].axis("off")
        ax[i, 3].set_title(f"{subject} - New")

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "crypt_segmentation_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
