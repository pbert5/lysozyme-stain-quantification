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

from src.lysozyme_stain_quantification.utils.remove_artifacts import remove_rectangles
import dask.array as da

from dask.delayed import delayed

from ..scoring_selector_mod import scoring_selector

# ---------------------------- utilities ---------------------------- #


def to_float01(img: da.Array) -> da.Array:
    """Return an array scaled to [0, 1] float without modifying NaNs/Inf."""
    if img.dtype in (np.float32, np.float64):
        return da.clip(img, 0.0, 1.0)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return da.clip(img.astype(np.float32) / float(info.max), 0.0, 1.0)
    arr = img.astype(np.float32)
    lo, hi = da.nanpercentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return da.clip(arr, 0.0, 1.0)


def minmax(x: da.Array) -> da.Array:
    """Normalize an array to [0, 1] (safe if constant)."""
    x = x.astype(np.float32, copy=False)
    lo, hi = da.nanmin(x), da.nanmax(x)
    if not da.isfinite(lo) or not da.isfinite(hi) or hi <= lo:
        return da.zeros_like(x, dtype=np.float32)
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


def preprocess_for_caps(img: da.Array, salt_and_pepper_noise_size: Optional[int]) -> da.Array:
    """
    Normalize the image and optionally remove salt-and-pepper noise.

    Historically we applied a white top-hat cleanup before cap decomposition; this
    keeps that option configurable per subject.
    """
    image: da.Array = to_float01(img)
    if salt_and_pepper_noise_size is None or salt_and_pepper_noise_size <= 0:
        return image

    radius = max(1, int(round(salt_and_pepper_noise_size)))
    footprint = disk(radius)
    white_hat = delayed(white_tophat)(image, footprint)
    cleaned: da.Array = image - da.from_delayed(white_hat, shape=image.shape, dtype=image.dtype)
    return cleaned


# ---------------------------- morphology helpers ---------------------------- #


def _caps_preprocess(image: da.Array, small_r: int, big_r: int) -> Tuple[da.Array, da.Array]:
    """Shared preprocessing for cap-style morphology computations."""
    image = to_float01(image)
    #image = equalize_adapthist(image, clip_limit=0.01)

    hats: da.Array = dilation(image, disk(big_r)) - dilation(image, disk(small_r))
    hats: da.Array = minmax(hats)
    return image, hats


def caps(image: da.Array, small_r: int, big_r: int) -> Tuple[da.Array, da.Array, da.Array]:
    """Contrast-adaptive cap decomposition returning (hats, clean, troughs)."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)

    clean = image_eq - da.minimum(image_eq, hats)
    clean = minmax(clean)

    troughs = da.maximum(image_eq, hats) - image_eq
    troughs = minmax(troughs)
    return hats, clean, troughs


def caps_clean(image: da.Array, small_r: int, big_r: int) -> da.Array:
    """Return only the 'clean' component of the cap decomposition."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)
    clean = image_eq - da.minimum(image_eq, hats)
    return minmax(clean)


def caps_trough(image: da.Array, small_r: int, big_r: int) -> da.Array:
    """Return only the 'troughs' component of the cap decomposition."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)
    troughs = da.maximum(image_eq, hats) - image_eq
    return minmax(troughs)


def caps_clean_troughs(image: da.Array, small_r: int, big_r: int) -> Tuple[da.Array, da.Array]:
    """Return both clean and trough components while avoiding extra work."""
    image_eq, hats = _caps_preprocess(image, small_r, big_r)

    clean = image_eq - da.minimum(image_eq, hats)
    clean = minmax(clean)

    troughs = da.maximum(image_eq, hats) - image_eq
    troughs = minmax(troughs)
    return clean, troughs


def show_caps(image: da.Array, small_r: int, big_r: int) -> Tuple[da.Array, da.Array, da.Array]:
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
    crypt_img: da.Array,
    tissue_image: da.Array,
    blob_size_px: Optional[int] = None,
) -> da.Array:
    """Legacy seed generator kept for reference."""
    del blob_size_px  # legacy placeholder
    tissue_clean, tissue_troughs = caps_clean_troughs(tissue_image, 1, 40)
    crypt_clean = caps_clean(crypt_img, 2, 40)

    thinned_crypts = da.maximum(crypt_clean - tissue_clean, 0)
    split_crypts = da.maximum(crypt_clean - tissue_troughs, 0)

    good_crypts = minmax(opening(split_crypts * thinned_crypts, footprint=disk(5)) ** 0.5)
    distance = tissue_troughs - good_crypts
    maxi = local_maxima(good_crypts)
    crypt_seeds = watershed(distance, markers=maxi, mask=tissue_troughs < good_crypts)
    labeled: da.Array = sk_label(crypt_seeds > 0)  # type: ignore
    return labeled


def limited_expansion(
    crypt_img: da.Array,
    tissue_image: da.Array,
    blob_size_px: Optional[int] = None,
) -> da.Array:
    """Limit vertical expansion of crypts by suppressing tissue spillover."""
    del blob_size_px  # not used but kept for interface parity
    outer_troughs = caps_trough(minmax(tissue_image), 10, 50)
    outer_troughs = minmax(outer_troughs)

    adjusted = da.maximum(crypt_img - outer_troughs, 0)
    crypt_max: da.Array = binary_opening(local_maxima(adjusted, indices=False), footprint=disk(1)).astype(bool)
    tissue_max = binary_opening(local_maxima(outer_troughs), disk(2))

    alt: da.Array = invert(dilation(minmax(adjusted), footprint=disk(2)) + minmax(outer_troughs))
    spread_early= delayed(watershed, pure=True)(image=alt, markers=da.maximum(crypt_max * 1, tissue_max * 2))
    spread = da.from_delayed(spread_early, shape=crypt_img.shape, dtype=np.int32)
    return spread


def identify_crypt_seeds_new(
    crypt_img: da.Array,
    tissue_img: da.Array,
    *,
    params: Optional[MorphologyParams] = None,
) -> da.Array:
    """Newer seed logic that prefers solid crypts."""
    params = params or DEFAULT_MORPHOLOGY_PARAMS

    crypt_img = preprocess_for_caps(crypt_img, params.salt_and_pepper_noise_size)
    
    tissue_img_processed: da.Array = preprocess_for_caps(
        da.from_delayed(
            delayed(equalize_adapthist)(tissue_img),
            shape=tissue_img.shape, dtype=tissue_img.dtype), params.salt_and_pepper_noise_size)

    max_expansion = max(1, int(round(params.crypt_radius_px)))
    tissue_clean, tissue_troughs = caps_clean_troughs(tissue_img_processed, 1, max_expansion)
    crypt_clean, crypt_troughs = caps_clean_troughs(crypt_img, 2, max_expansion)

    thinned_crypts: da.Array = da.maximum(crypt_clean - tissue_clean, 0)
    split_crypts: da.Array = da.maximum(crypt_clean - tissue_troughs, 0)

    good_crypts = minmax(
        da.from_delayed(delayed(opening)(split_crypts * thinned_crypts, footprint=disk(5)), shape=split_crypts.shape, dtype=split_crypts.dtype)** 0.5
        )
    if params.peak_smoothing_sigma and params.peak_smoothing_sigma > 0:

        good_crypts = delayed(gaussian_filter)(good_crypts, params.peak_smoothing_sigma)
        good_crypts = minmax(da.from_delayed(good_crypts, shape=good_crypts.shape, dtype=good_crypts.dtype))

    distance = tissue_troughs - good_crypts
    maxi = delayed(local_maxima)(good_crypts)
    seeds = delayed(watershed)(distance, markers=maxi, mask=tissue_troughs < good_crypts)

    pre_labeled = delayed(sk_label)(seeds > 0)  # type: ignore
    labeled: da.Array = da.from_delayed(pre_labeled, shape=good_crypts.shape, dtype=good_crypts.dtype)
    return labeled


def identify_potential_crypts_old_like(
    crypt_img_in: da.Array,
    tissue_image_in: da.Array,
    blob_size_px: Optional[int] = None,
    external_crypt_seeds: Optional[da.Array] = None,
    params_in: Optional[MorphologyParams] = None,
) -> da.Array:
    """Original watershed skeleton with optional external seeds."""
    crypt_img: da.Array = to_float01(crypt_img_in)
    tissue_image: da.Array = to_float01(tissue_image_in)

    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Image shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    params: Optional[MorphologyParams] = params_in or DEFAULT_MORPHOLOGY_PARAMS
    effective_blob = float(blob_size_px) if blob_size_px else float(params.crypt_radius_px)
    erosion_dim = _odd(max(3, int(round(effective_blob / 10.0))))
    erosion_footprint: da.Array = da.ones((erosion_dim, erosion_dim), dtype=bool) # this conversion may be a little sketchy

    if external_crypt_seeds is None:
        crypt_seeds_bool: da.Array = crypt_img > da.minimum(tissue_image, crypt_img)
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seeds_bool_cleaned = delayed(binary_erosion)(crypt_seeds_bool, footprint=erosion_footprint)
        crypt_seeds_bool_cleaned = delayed(remove_small_objects)(crypt_seeds_bool, min_size=min_region_area)
        
        labeled_diff_r_del = delayed(sk_label)(crypt_seeds_bool_cleaned)  # type: ignore
    else:
        external_crypt_seeds = external_crypt_seeds.astype(da.int32)
        crypt_seed_mask = external_crypt_seeds > 0
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seed_mask = delayed(remove_small_objects)(crypt_seed_mask, min_size=min_region_area)
        labeled_diff_r_del = delayed(sk_label)(crypt_seed_mask)  # type: ignore
    labeled_diff_r = da.from_delayed(labeled_diff_r_del, shape=crypt_img.shape, dtype=da.int32)

    abs_diff = da.maximum(tissue_image - crypt_img, 0)
    mask_gt_red = abs_diff > crypt_img

    erosion_kernel_size = max(3, int(round(effective_blob * 0.15)))
    erosion_kernel_size = erosion_kernel_size if erosion_kernel_size % 2 == 0 else erosion_kernel_size + 1
    cv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    mask_u8 = (mask_gt_red.astype(np.uint8) * 255)
    erosion_iterations = max(1, int(round(effective_blob / 20.0)))
    mask_eroded_u8 = delayed(cv2.erode)(mask_u8, cv_kernel, iterations=erosion_iterations)
    tissue_eroded = da.from_delayed(mask_eroded_u8, shape=mask_u8.shape, dtype=da.bool)


    combined_labels = da.zeros_like(labeled_diff_r, dtype=int)
    combined_labels[tissue_eroded] = 2
    combined_labels[labeled_diff_r > 0] = 1

    if params.intervilli_distance_px:
        expand_distance = max(1, int(round(params.intervilli_distance_px)))
    else:
        expand_distance = max(1, int(round(effective_blob * 2.5)))
    expanded_labels_del = delayed(expand_labels)(combined_labels, distance=expand_distance)
    expanded_labels: da.Array = da.from_delayed(expanded_labels_del, shape=combined_labels.shape, dtype=combined_labels.dtype)

    reworked: da.Array = da.zeros_like(expanded_labels, dtype=np.int32)
    reworked: da.Array = da.where(expanded_labels == 2, 1, reworked)
    mask_copy: da.Array = (expanded_labels != 2) & (labeled_diff_r != 0)
    reworked = da.where(mask_copy, labeled_diff_r + 1, reworked)

    mask_ws = expanded_labels > 0
    
    dst1 = delayed(distance_transform_edt)(combined_labels == 1)  
    dst2 = delayed(distance_transform_edt)(combined_labels == 2)  



    elevation = minmax(da.from_delayed(dst2, shape=combined_labels.shape, dtype=da.float64)) - minmax(  # type: ignore
        da.from_delayed(dst1, shape=combined_labels.shape, dtype=da.float64)  # type: ignore
    )

    ws_labels_del = delayed(watershed)(elevation, markers=reworked, mask=mask_ws)
    ws_labels: da.Array = da.from_delayed(ws_labels_del, shape=combined_labels.shape, dtype=np.int32)
    ws_labels = da.where(ws_labels == 1, 0, ws_labels)
    ws_labels = da.where(ws_labels > 1, ws_labels - 1, ws_labels)
    return ws_labels.astype(da.int32)


def _seed_health_metrics(seed_labels: da.Array ) -> Dict[str, Any]:
    """Compute basic metrics to judge seed quality."""
    if seed_labels is None or seed_labels.size == 0:
        return dict(n_labels=0, coverage=0.0, mean_area=0.0)
    n_labels: int = da.max(seed_labels)
    coverage = da.count_nonzero(seed_labels) / seed_labels.size
    if n_labels > 0:
        areas = da.bincount(da.ravel(seed_labels))
        mean_area = areas.mean() if areas.size else 0.0
    else:
        mean_area = 0.0
    return dict(n_labels=n_labels, coverage=coverage, mean_area=mean_area)


def _score_label_set(
    labels: Optional[da.Array],
    raw_img: da.Array,
    *,
    debug: bool = False,
) -> Tuple[da.Array, da.Array]:
    """
    Evaluate a set of labels using the scoring selector.

    Returns a tuple of (evaluation_score, summary_dict) as delayed objects where higher scores are better.
    Both outputs are dask delayed objects that need to be computed.
    """
    
    @delayed
    def _compute_score_delayed(labels_arr: np.ndarray, raw_arr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Compute score from numpy arrays."""
        from skimage.measure import regionprops
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        
        summary: Dict[str, Any] = {}
        
        if labels_arr is None or labels_arr.size == 0:
            summary.update(region_count=0, mean_quality_score=None, evaluation_score=None, reason="empty")
            return float("-inf"), summary

        region_count = int(labels_arr.max())
        summary["region_count"] = region_count

        if region_count == 0:
            summary.update(mean_quality_score=None, evaluation_score=None, reason="no_regions")
            return float("-inf"), summary

        try:
            # Compute region properties directly instead of calling scoring_selector
            regions = regionprops(labels_arr, intensity_image=raw_arr)
            if len(regions) == 0:
                summary.update(mean_quality_score=None, evaluation_score=None, reason="no_regions")
                return float("-inf"), summary
            
            # Build properties dataframe
            properties = []
            for region in regions:
                if raw_arr is not None:
                    red_intensity_per_area = region.mean_intensity
                else:
                    red_intensity_per_area = 0.0
                
                circularity = (
                    4 * np.pi * region.area / (region.perimeter ** 2)
                    if region.perimeter > 0
                    else 0.0
                )
                
                properties.append({
                    "label_id": int(region.label),
                    "area": float(region.area),
                    "physical_com": tuple(region.centroid),
                    "red_intensity_per_area": float(red_intensity_per_area),
                    "circularity": float(circularity),
                    "perimeter": float(region.perimeter),
                })
            
            properties_df = pd.DataFrame(properties)
            
            # Calculate line fit deviation
            if len(properties_df) >= 2:
                centers = np.array(list(properties_df["physical_com"]))
                X = centers[:, 1].reshape(-1, 1)
                y = centers[:, 0]
                reg = LinearRegression().fit(X, y)
                m = float(reg.coef_[0])
                b = float(reg.intercept_)
                x_coords = centers[:, 1]
                y_coords = centers[:, 0]
                distances = np.abs(m * x_coords - y_coords + b) / np.sqrt(m**2 + 1)
                areas = properties_df["area"].to_numpy(dtype=float)
                radius_approx = np.sqrt(areas / 2.0)
                radius_approx[radius_approx == 0] = 1.0
                normalized_distances = distances / radius_approx
                properties_df["normalized_line_distance"] = normalized_distances
            else:
                properties_df["normalized_line_distance"] = 0.0
            
            # Calculate scores
            max_circularity = float(properties_df["circularity"].max())
            properties_df["circularity_score"] = (
                1 - (properties_df["circularity"] / max_circularity)
                if max_circularity > 0
                else 1.0
            )
            
            max_area = float(properties_df["area"].max())
            properties_df["area_score"] = (
                1 - (properties_df["area"] / max_area) if max_area > 0 else 0.0
            )
            
            max_line_dist = float(properties_df["normalized_line_distance"].max())
            properties_df["line_fit_score"] = (
                properties_df["normalized_line_distance"] / max_line_dist
                if max_line_dist > 0
                else 0.0
            )
            
            max_red_intensity = float(properties_df["red_intensity_per_area"].max())
            properties_df["red_intensity_score"] = (
                1 - (properties_df["red_intensity_per_area"] / max_red_intensity)
                if max_red_intensity > 0
                else 1.0
            )
            
            # Default weights
            weights = {
                "circularity": 0.35,
                "area": 0.25,
                "line_fit": 0.15,
                "red_intensity": 0.15,
            }
            
            properties_df["quality_score"] = (
                weights["circularity"] * properties_df["circularity_score"]
                + weights["area"] * properties_df["area_score"]
                + weights["line_fit"] * properties_df["line_fit_score"]
                + weights["red_intensity"] * properties_df["red_intensity_score"]
            )
            
            mean_quality = float(properties_df["quality_score"].mean())
            evaluation_score = -mean_quality  # Lower quality_score is better; invert
            summary.update(
                mean_quality_score=mean_quality,
                evaluation_score=evaluation_score,
            )
            
            if debug:
                summary["quality_scores_sample"] = properties_df["quality_score"].tolist()[:10]
                
        except Exception as exc:  # pragma: no cover - defensive
            summary.update(mean_quality_score=None, evaluation_score=None, reason=f"scoring_failed:{exc!r}")
            return float("-inf"), summary

        return evaluation_score, summary
    
    if labels is None:
        # Return delayed values for None case
        score_delayed = delayed(lambda: (float("-inf"), {"reason": "labels_none"}))()
        score_da = da.from_delayed(score_delayed, shape=(), dtype=object)
        return score_da, score_da
    
    # Create delayed computation
    result_delayed = _compute_score_delayed(labels, raw_img)
    
    # Extract score and summary as separate delayed values
    @delayed
    def extract_score(result):
        return result[0]
    
    @delayed  
    def extract_summary(result):
        return result[1]
    
    score_delayed = extract_score(result_delayed)
    summary_delayed = extract_summary(result_delayed)
    
    # Convert to dask arrays (scalars)
    score_da = da.from_delayed(score_delayed, shape=(), dtype=np.float64)
    summary_da = da.from_delayed(summary_delayed, shape=(), dtype=object)
    
    return score_da, summary_da


def identify_potential_crypts_hybrid(
    crypt_img: da.Array,
    tissue_image: da.Array,
    blob_size_px: Optional[int] = None,
    use_new_seeds: bool = True,
    auto_fallback: bool = True,
    min_seed_count: int = 10,
    min_coverage: float = 0.002,
    debug: bool = False,
    params: Optional[MorphologyParams] = None,
) -> Tuple[da.Array, Dict[str, Any]]:
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

    new_seeds: da.Array | None = identify_crypt_seeds_new(crypt_img, tissue_image, params=params) if use_new_seeds else None
    new_metrics: Dict[str, Any] = _seed_health_metrics(
        new_seeds if new_seeds is not None else da.zeros_like(crypt_img, dtype=da.int32)
    )
    dbg["new_seed_metrics"] = new_metrics

    metrics_meet_threshold = (new_metrics["n_labels"] >= min_seed_count) and (new_metrics["coverage"] >= min_coverage)
    dbg["seed_metrics_meet_threshold"] = bool(metrics_meet_threshold)

    candidate_labels: Dict[str, da.Array] = {}

    if use_new_seeds and new_seeds is not None:
        candidate_labels["new_seeded"] = identify_potential_crypts_old_like(
            crypt_img,
            tissue_image,
            blob_size_px=blob_size_px,
            external_crypt_seeds=new_seeds,
            params_in=params,
        )

    candidate_labels["legacy_seeded"] = identify_potential_crypts_old_like(
        crypt_img,
        tissue_image,
        blob_size_px=blob_size_px,
        external_crypt_seeds=None,
        params_in=params,
    )

    # Wrap the selection logic in a delayed function
    @delayed
    def _select_best_candidate(
        crypt_arr: np.ndarray,
        candidate_label_dict: Dict[str, np.ndarray],
        metrics_ok: bool,
        use_auto_fallback: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select the best candidate based on scoring."""
        from skimage.measure import regionprops
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        
        selected_key = "legacy_seeded"
        best_score = float("-inf")
        candidate_summaries: Dict[str, Dict[str, Any]] = {}

        if use_auto_fallback:
            for key, label_img_np in candidate_label_dict.items():
                if key == "new_seeded" and not metrics_ok:
                    summary = dict(
                        region_count=int(label_img_np.max()),
                        mean_quality_score=None,
                        evaluation_score=float("-inf"),
                        reason="seed_metrics_below_threshold",
                    )
                    candidate_summaries[key] = summary
                    continue
                
                # Compute score for this candidate - inline scoring logic
                try:
                    regions = regionprops(label_img_np, intensity_image=crypt_arr)
                    if len(regions) == 0:
                        summary = dict(
                            mean_quality_score=None,
                            evaluation_score=float("-inf"),
                            reason="no_regions",
                        )
                    else:
                        # Build properties
                        properties = []
                        for region in regions:
                            if crypt_arr is not None:
                                red_intensity_per_area = region.mean_intensity
                            else:
                                red_intensity_per_area = 0.0
                            
                            circularity = (
                                4 * np.pi * region.area / (region.perimeter ** 2)
                                if region.perimeter > 0
                                else 0.0
                            )
                            
                            properties.append({
                                "label_id": int(region.label),
                                "area": float(region.area),
                                "physical_com": tuple(region.centroid),
                                "red_intensity_per_area": float(red_intensity_per_area),
                                "circularity": float(circularity),
                                "perimeter": float(region.perimeter),
                            })
                        
                        properties_df = pd.DataFrame(properties)
                        
                        # Calculate line fit
                        if len(properties_df) >= 2:
                            centers = np.array(list(properties_df["physical_com"]))
                            X = centers[:, 1].reshape(-1, 1)
                            y = centers[:, 0]
                            reg = LinearRegression().fit(X, y)
                            m = float(reg.coef_[0])
                            b = float(reg.intercept_)
                            x_coords = centers[:, 1]
                            y_coords = centers[:, 0]
                            distances = np.abs(m * x_coords - y_coords + b) / np.sqrt(m**2 + 1)
                            areas = properties_df["area"].to_numpy(dtype=float)
                            radius_approx = np.sqrt(areas / 2.0)
                            radius_approx[radius_approx == 0] = 1.0
                            normalized_distances = distances / radius_approx
                            properties_df["normalized_line_distance"] = normalized_distances
                        else:
                            properties_df["normalized_line_distance"] = 0.0
                        
                        # Calculate scores
                        max_circularity = float(properties_df["circularity"].max())
                        properties_df["circularity_score"] = (
                            1 - (properties_df["circularity"] / max_circularity)
                            if max_circularity > 0
                            else 1.0
                        )
                        
                        max_area = float(properties_df["area"].max())
                        properties_df["area_score"] = (
                            1 - (properties_df["area"] / max_area) if max_area > 0 else 0.0
                        )
                        
                        max_line_dist = float(properties_df["normalized_line_distance"].max())
                        properties_df["line_fit_score"] = (
                            properties_df["normalized_line_distance"] / max_line_dist
                            if max_line_dist > 0
                            else 0.0
                        )
                        
                        max_red_intensity = float(properties_df["red_intensity_per_area"].max())
                        properties_df["red_intensity_score"] = (
                            1 - (properties_df["red_intensity_per_area"] / max_red_intensity)
                            if max_red_intensity > 0
                            else 1.0
                        )
                        
                        # Default weights
                        weights_dict = {
                            "circularity": 0.35,
                            "area": 0.25,
                            "line_fit": 0.15,
                            "red_intensity": 0.15,
                        }
                        
                        properties_df["quality_score"] = (
                            weights_dict["circularity"] * properties_df["circularity_score"]
                            + weights_dict["area"] * properties_df["area_score"]
                            + weights_dict["line_fit"] * properties_df["line_fit_score"]
                            + weights_dict["red_intensity"] * properties_df["red_intensity_score"]
                        )
                        
                        mean_quality = float(properties_df["quality_score"].mean())
                        score = -mean_quality  # Lower quality_score is better; invert
                        summary = dict(
                            mean_quality_score=mean_quality,
                            evaluation_score=score,
                        )
                        
                        if score > best_score:
                            best_score = score
                            selected_key = key
                            
                except Exception as exc:
                    summary = dict(
                        mean_quality_score=None,
                        evaluation_score=float("-inf"),
                        reason=f"scoring_failed:{exc!r}",
                    )
                
                candidate_summaries[key] = summary
        else:
            if "new_seeded" in candidate_label_dict and use_new_seeds and metrics_ok:
                selected_key = "new_seeded"
            candidate_summaries = {selected_key: dict(evaluation_score=None, reason="auto_fallback_disabled")}
            best_score = float("nan")

        selected_labels = candidate_label_dict[selected_key]
        
        debug_info = {
            "selected_candidate": selected_key,
            "candidate_summaries": candidate_summaries,
            "using_new_seeds": selected_key == "new_seeded",
            "best_score": best_score,
        }
        
        return selected_labels, debug_info

    # Prepare inputs for the delayed selection
    if auto_fallback:
        # Call the delayed function with all candidates
        result_delayed = _select_best_candidate(
            crypt_img,
            candidate_labels,
            metrics_meet_threshold,
            auto_fallback,
        )
        
        # Extract labels and debug info
        @delayed
        def get_labels(result):
            return result[0]
        
        @delayed
        def get_debug(result):
            return result[1]
        
        labels_delayed = get_labels(result_delayed)
        debug_delayed = get_debug(result_delayed)
        
        # Convert to dask array
        shape = crypt_img.shape
        labels = da.from_delayed(labels_delayed, shape=shape, dtype=np.int32)
        
        # Merge static and delayed debug info
        dbg_final = {**dbg}
        # Note: debug_delayed is a delayed object, needs compute to access
        dbg_final["_delayed_debug"] = debug_delayed
        
    else:
        # Simple case: just pick one without scoring
        if "new_seeded" in candidate_labels and use_new_seeds and metrics_meet_threshold:
            selected_key = "new_seeded"
        else:
            selected_key = "legacy_seeded"
        
        labels = candidate_labels[selected_key]
        dbg_final = {
            **dbg,
            "selected_candidate": selected_key,
            "candidate_summaries": {selected_key: dict(evaluation_score=None, reason="auto_fallback_disabled")},
            "using_new_seeds": selected_key == "new_seeded",
            "best_score": float("nan"),
        }

    if debug:
        dbg_final["labels_max"] = labels.max()
        dbg_final["labels_coverage"] = da.count_nonzero(labels) / labels.size

    return labels, dbg_final


# ---------------------------- driver ---------------------------- #


def build_image_sets() -> list[Tuple[str, da.Array, da.Array]]:
    """Load source images and return [(subject, crypt_channel, tissue_channel), ...]."""
    jej2 = da.array(tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c2.tif"
    ))
    jej2_tissue = da.array(tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c1.tif"
    ))
    jej3 = da.array(tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c2.tif"
    ))
    jej3_tissue = da.array(tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c1.tif"
    ))
    g3fr_rgb = tiff.imread("/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2.tif")
    g3fr_sep_rfp = da.array(tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_RFP.tif"
    ))
    g3fr_sep_dapi = da.array(tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_DAPI.tif"
    ))

    evil_g3fr = da.from_delayed(delayed(remove_rectangles)(g3fr_rgb), shape=g3fr_rgb.shape, dtype=g3fr_rgb.dtype)


    return [
        ("Jej-2", jej2[..., 1], jej2_tissue[..., 2]),
        ("Jej-3", jej3[..., 1], jej3_tissue[..., 2]),
        ("G3FR", evil_g3fr[..., 0], evil_g3fr[..., 2]),
        ("G3FR_sep", g3fr_sep_rfp[..., 0], g3fr_sep_dapi[..., 2]),
    ]


def _load_reference_image() -> Tuple[da.Array, da.Array, da.Array]:
    """Load a single reference RGB image (crypt, crypt channel, dapi channel)."""
    image = da.array(tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2.tif"
    ))
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

        ax[i, 0].imshow(da.stack([crypt, da.zeros_like(crypt), tissue], axis=-1))
        ax[i, 0].axis("off")
        ax[i, 0].set_title(f"{subject} - Crypt")

        old_labels = identify_potential_crypts_old_like(crypt, tissue, params_in=params)
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
