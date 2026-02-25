"""
Scoring-based region selector that replaces the complex merge pipeline.
Instead of merging, we score all detections based on quality metrics and select the best ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops
import matplotlib.pyplot as plt


DEFAULT_WEIGHTS = {
    "circularity": 0.35,  # Most important - want circular regions
    "area": 0.25,         # Second - want consistent sizes
    "line_fit": 0.15,     # Moderate - want aligned regions
    "red_intensity": 0.15,  # Moderate - want bright regions
    "com_consistency": 0.10,  # Retained for backwards compatibility
}


def _sanitize_positive_weights(weights: np.ndarray | None, size: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(size), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.size != int(size):
        w = np.ones(int(size), dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        w = np.ones(int(size), dtype=np.float64)
    return w


def fit_weighted_centroid_curve(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    max_degree: int = 2,
) -> dict[str, object] | None:
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return None

    w = _sanitize_positive_weights(weights, x.size)
    wsum = float(np.sum(w))

    x_mean = float(np.sum(w * x) / wsum)
    y_mean = float(np.sum(w * y) / wsum)
    var_x = float(np.sum(w * (x - x_mean) ** 2) / wsum)
    var_y = float(np.sum(w * (y - y_mean) ** 2) / wsum)

    if var_x >= var_y:
        model = "y_from_x"
        indep = x
        dep = y
    else:
        model = "x_from_y"
        indep = y
        dep = x

    degree = int(max(1, min(int(max_degree), int(indep.size) - 1)))
    indep_mean = float(np.sum(w * indep) / wsum)
    indep_centered = indep - indep_mean
    indep_scale = float(np.sqrt(np.sum(w * indep_centered * indep_centered) / wsum))
    if not np.isfinite(indep_scale) or indep_scale <= 1e-6:
        indep_span = float(np.max(indep) - np.min(indep))
        indep_scale = max(indep_span / 2.0, 1.0)

    scaled = indep_centered / indep_scale
    fit_w = np.sqrt(w)
    coeffs_desc = np.polyfit(scaled, dep, deg=degree, w=fit_w)

    return {
        "model": model,
        "degree": degree,
        "coeffs_desc": coeffs_desc.astype(np.float64),
        "indep_mean": indep_mean,
        "indep_scale": float(indep_scale),
    }


def evaluate_centroid_curve(
    curve_model: dict[str, object],
    indep_values: np.ndarray,
) -> np.ndarray:
    coeffs_desc = np.asarray(curve_model.get("coeffs_desc"), dtype=np.float64)
    indep_mean = float(curve_model.get("indep_mean", 0.0))
    indep_scale = float(curve_model.get("indep_scale", 1.0))
    if not np.isfinite(indep_scale) or indep_scale <= 1e-12:
        indep_scale = 1.0
    indep = np.asarray(indep_values, dtype=np.float64)
    scaled = (indep - indep_mean) / indep_scale
    return np.polyval(coeffs_desc, scaled)


def centroid_curve_distances(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    curve_model: dict[str, object],
) -> np.ndarray:
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)
    model = str(curve_model.get("model", "y_from_x"))
    if model == "x_from_y":
        x_hat = evaluate_centroid_curve(curve_model, y)
        return np.abs(x - x_hat)
    y_hat = evaluate_centroid_curve(curve_model, x)
    return np.abs(y - y_hat)


def sample_centroid_curve_points(
    curve_model: dict[str, object] | None,
    image_shape: tuple[int, ...],
    *,
    num_samples: int = 512,
) -> np.ndarray:
    if curve_model is None:
        return np.empty((0, 2), dtype=np.float64)

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 1 or w <= 1:
        return np.empty((0, 2), dtype=np.float64)

    model = str(curve_model.get("model", "y_from_x"))
    sample_count = max(2, int(num_samples))
    if model == "x_from_y":
        ys = np.linspace(0.0, float(h - 1), sample_count, dtype=np.float64)
        xs = evaluate_centroid_curve(curve_model, ys)
    else:
        xs = np.linspace(0.0, float(w - 1), sample_count, dtype=np.float64)
        ys = evaluate_centroid_curve(curve_model, xs)

    valid = np.isfinite(xs) & np.isfinite(ys)
    valid &= (xs >= 0.0) & (xs <= float(w - 1)) & (ys >= 0.0) & (ys <= float(h - 1))
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((xs[valid], ys[valid])).astype(np.float64, copy=False)


def fit_centroid_curve_from_properties(
    properties_df: pd.DataFrame,
    *,
    max_degree: int = 2,
) -> dict[str, object] | None:
    if properties_df is None or len(properties_df) < 2:
        return None
    if "physical_com" not in properties_df.columns:
        return None
    centers = np.asarray(list(properties_df["physical_com"]), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        return None
    y_coords = centers[:, 0]
    x_coords = centers[:, 1]
    weights = (
        np.asarray(properties_df["total_red_intensity"], dtype=np.float64)
        if "total_red_intensity" in properties_df.columns
        else None
    )
    return fit_weighted_centroid_curve(
        x_coords=x_coords,
        y_coords=y_coords,
        weights=weights,
        max_degree=max_degree,
    )


def fit_centroid_curve_from_labels(
    label_img: np.ndarray,
    intensity_img: np.ndarray | None = None,
    *,
    max_degree: int = 2,
) -> dict[str, object] | None:
    labels = np.asarray(label_img)
    if labels.ndim != 2:
        labels = np.squeeze(labels)
    regions = regionprops(labels.astype(np.int32), intensity_image=intensity_img)
    if len(regions) < 2:
        return None

    x_coords = np.array([float(r.centroid[1]) for r in regions], dtype=np.float64)
    y_coords = np.array([float(r.centroid[0]) for r in regions], dtype=np.float64)
    if intensity_img is None:
        weights = np.ones_like(x_coords, dtype=np.float64)
    else:
        weights = np.array(
            [
                float(
                    (
                        getattr(r, "intensity_mean", getattr(r, "mean_intensity", 0.0))
                        or 0.0
                    )
                    * float(r.area)
                )
                for r in regions
            ],
            dtype=np.float64,
        )
    return fit_weighted_centroid_curve(
        x_coords=x_coords,
        y_coords=y_coords,
        weights=weights,
        max_degree=max_degree,
    )


def scoring_selector(
    label_img: np.ndarray,
    raw_img: np.ndarray | None = None,
    *,
    debug: bool = False,
    max_regions: int | bool = 5,
    weights: dict[str, float] | None = None,
    return_details: bool = False,
):
    """Select the best regions based on quality metrics.

    Args:
        label_img: Labeled image array with detected regions.
        raw_img: Optional raw red channel image for quality assessment.
        debug: Whether to enable debug output.
        max_regions: Maximum number of regions to select; falsy values keep all.
        weights: Optional weighting for the scoring components.
        return_details: When True, also return debug metadata alongside labels.

    Returns:
        Filtered label array, optionally paired with debug metadata when
        ``return_details`` is True.
    """

    working_labels = np.asarray(label_img).copy()
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    scoring_history: list[dict[str, float]] = []
    curve_model: dict[str, object] | None = None

    def _log(message: str) -> None:
        if debug:
            print(message)

    def _calculate_region_properties() -> pd.DataFrame:
        if debug:
            unique_labels = np.unique(working_labels)
            _log(
                f"[SCORING DEBUG] Input label array has {len(unique_labels)} "
                f"unique labels: {unique_labels}"
            )

        regions = regionprops(working_labels, intensity_image=raw_img)
        if len(regions) == 0:
            return pd.DataFrame()

        properties: list[dict[str, object]] = []
        for region in regions:
            if debug:
                _log(
                    f"[SCORING DEBUG] Processing label {region.label}: "
                    f"area = {region.area} pixels"
                )

            if raw_img is not None:
                total_red_intensity = region.mean_intensity * region.area
                red_intensity_per_area = region.mean_intensity
            else:
                total_red_intensity = 0.0
                red_intensity_per_area = 0.0

            circularity = (
                4 * np.pi * region.area / (region.perimeter ** 2)
                if region.perimeter > 0
                else 0.0
            )

            properties.append(
                {
                    "label_id": int(region.label),
                    "area": float(region.area),
                    "physical_com": tuple(region.centroid),
                    "red_intensity_per_area": float(red_intensity_per_area),
                    "total_red_intensity": float(total_red_intensity),
                    "circularity": float(circularity),
                    "perimeter": float(region.perimeter),
                }
            )

        return pd.DataFrame(properties)

    def _calculate_line_fit_deviation(properties_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute distance from an intensity-weighted centroid curve.
        - Fit quadratic (fallback linear) with adaptive orientation.
        - Distances normalized by sqrt(area/2) as before.
        """
        nonlocal curve_model
        if len(properties_df) < 2:
            properties_df = properties_df.copy()
            properties_df["distance_from_curve"] = 0.0
            properties_df["normalized_curve_distance"] = 0.0
            properties_df["distance_from_line"] = 0.0
            properties_df["normalized_line_distance"] = 0.0
            curve_model = None
            return properties_df

        centers = np.array(list(properties_df["physical_com"]))
        x_coords = centers[:, 1].astype(float)
        y_coords = centers[:, 0].astype(float)

        weights_arr = (
            np.asarray(properties_df["total_red_intensity"], dtype=np.float64)
            if "total_red_intensity" in properties_df.columns
            else None
        )
        curve_model = fit_weighted_centroid_curve(
            x_coords=x_coords,
            y_coords=y_coords,
            weights=weights_arr,
            max_degree=2,
        )
        if curve_model is None:
            distances = np.zeros_like(x_coords, dtype=np.float64)
        else:
            distances = centroid_curve_distances(x_coords, y_coords, curve_model)

        areas = properties_df["area"].to_numpy(dtype=float)
        radius_approx = np.sqrt(areas / 2.0)
        radius_approx[radius_approx == 0] = 1.0
        normalized_distances = distances / radius_approx

        properties_df = properties_df.copy()
        properties_df["distance_from_curve"] = distances
        properties_df["normalized_curve_distance"] = normalized_distances
        # Keep legacy column names for downstream compatibility.
        properties_df["distance_from_line"] = distances
        properties_df["normalized_line_distance"] = normalized_distances
        return properties_df

    def _score_regions(properties_df: pd.DataFrame) -> pd.DataFrame:
        if len(properties_df) == 0:
            return properties_df

        properties_df = properties_df.copy()

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
            1
            - (properties_df["red_intensity_per_area"] / max_red_intensity)
            if max_red_intensity > 0
            else 1.0
        )

        properties_df["quality_score"] = (
            weights.get("circularity", 0.4) * properties_df["circularity_score"]
            + weights.get("area", 0.3) * properties_df["area_score"]
            + weights.get("line_fit", 0.2) * properties_df["line_fit_score"]
            + weights.get("red_intensity", 0.1)
            * properties_df["red_intensity_score"]
        )

        if debug:
            scoring_history.append(
                {
                    "max_circularity": max_circularity,
                    "max_area": max_area,
                    "max_line_dist": max_line_dist,
                    "max_red_intensity": max_red_intensity,
                    "weights": dict(weights),
                }
            )

        return properties_df.sort_values("quality_score").reset_index(drop=True)

    def _select_best_regions(properties_df: pd.DataFrame) -> list[int]:
        sorted_df = properties_df.sort_values("quality_score")
        selection_cap = len(sorted_df) if not max_regions else min(int(max_regions), len(sorted_df))
        best_regions = sorted_df.head(selection_cap)

        if debug:
            _log(
                f"[SCORING DEBUG] Selected top {selection_cap} regions out of {len(sorted_df)} total"
            )
            _log(f"[SCORING DEBUG] Selected labels: {best_regions['label_id'].tolist()}")
            _log(f"[SCORING DEBUG] Quality scores: {best_regions['quality_score'].tolist()}")
            for _, row in best_regions.iterrows():
                _log(
                    f"[SCORING DEBUG] Label {int(row['label_id'])}: "
                    f"area={row['area']:.1f}, circ={row['circularity']:.3f}, "
                    f"red_int={row['red_intensity_per_area']:.2f}, "
                    f"line_dist={row['normalized_line_distance']:.2f}, "
                    f"final_score={row['quality_score']:.3f}"
                )

        return [int(label) for label in best_regions["label_id"].tolist()]

    def _create_filtered_labels(selected_label_ids: list[int]) -> np.ndarray:
        filtered = np.zeros_like(working_labels)
        if not selected_label_ids:
            return filtered

        if debug:
            _log(
                f"[SCORING DEBUG] Creating filtered labels from {len(selected_label_ids)} selected regions"
            )
            _log(f"[SCORING DEBUG] Selected IDs: {selected_label_ids}")

        lookup = np.zeros(max(selected_label_ids) + 1, dtype=filtered.dtype)
        for new_id, old_id in enumerate(selected_label_ids, start=1):
            lookup[old_id] = new_id
            if debug:
                pixels_count = int(np.sum(working_labels == old_id))
                _log(f"[SCORING DEBUG] Relabeled {old_id} -> {new_id}, {pixels_count} pixels")

        mask = np.isin(working_labels, selected_label_ids)
        filtered[mask] = lookup[working_labels[mask]]

        if debug:
            _log(
                f"[SCORING DEBUG] Filtered label array unique labels: {np.unique(filtered)}"
            )

        return filtered

    if debug:
        print("Starting quality-based region selection...")

    properties_df = _calculate_region_properties()
    if len(properties_df) == 0:
        if debug:
            _log("[SCORING DEBUG] No regions found to score")
        filtered_labels = np.zeros_like(working_labels)
        debug_info = {
            "properties_df": properties_df,
            "selected_labels": [],
            "scoring_history": scoring_history,
            "original_regions": max(len(np.unique(working_labels)) - 1, 0),
            "selected_regions": 0,
        }
        return (filtered_labels, debug_info) if return_details else filtered_labels

    properties_df = _calculate_line_fit_deviation(properties_df)
    properties_df = _score_regions(properties_df)

    if debug:
        _log(f"[SCORING DEBUG] Scored {len(properties_df)} regions")

    selected_labels = _select_best_regions(properties_df)
    filtered_labels = _create_filtered_labels(selected_labels)

    if debug:
        original_count = len(np.unique(working_labels)) - 1
        selected_count = len(np.unique(filtered_labels)) - 1
        _log(
            f"[SCORING DEBUG] Selection complete: {max(original_count, 0)} -> {max(selected_count, 0)} regions"
        )

    debug_info = {
        "properties_df": properties_df,
        "selected_labels": selected_labels,
        "scoring_history": scoring_history,
        "original_regions": max(len(np.unique(working_labels)) - 1, 0),
        "selected_regions": len(selected_labels),
        "curve_model": curve_model,
        "curve_points_xy": sample_centroid_curve_points(curve_model, working_labels.shape),
    }

    return (filtered_labels, debug_info) if return_details else filtered_labels


def plot_scoring_results(
    properties_df: pd.DataFrame,
    selected_labels: list[int] | None = None,
    *,
    save_path: str | None = None,
) -> None:
    """Plot scoring results for visualization using returned metadata."""

    if properties_df is None or len(properties_df) == 0:
        print("No scoring results to plot. Run scoring_selector() first.")
        return

    selected_labels = selected_labels or []

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    score_columns = [
        "circularity_score",
        "area_score",
        "line_fit_score",
        "red_intensity_score",
        "quality_score",
        "area",
    ]

    for ax, column in zip(axes, score_columns):
        if column not in properties_df.columns:
            continue

        properties_df.plot.scatter(x="label_id", y=column, ax=ax, alpha=0.7)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("Label ID")
        ax.set_ylabel("Score" if "score" in column else "Value")

        if selected_labels:
            highlighted = properties_df[properties_df["label_id"].isin(selected_labels)]
            ax.scatter(
                highlighted["label_id"],
                highlighted[column],
                c="red",
                s=100,
                alpha=0.8,
                marker="x",
                linewidth=3,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
