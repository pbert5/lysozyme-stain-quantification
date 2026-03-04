from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.io import imread

from .crypts.crypt_detection_solutions.effective_crypt_estimation import (
    EffectiveCryptEstimation,
    estimate_effective_selected_crypt_count,
)
from .crypts.scoring_selector_mod import fit_centroid_curve_from_labels, sample_centroid_curve_points
from .normalize_rfp import compute_normalized_rfp
from .quantify.crypt_fluorescence_summary import (
    SUMMARY_FIELD_ORDER,
    summarize_crypt_fluorescence,
    summarize_crypt_fluorescence_per_crypt,
)
from .segment_crypts import segment_crypts_dual
from .utils.overlays import render_label_overlay


@dataclass(frozen=True)
class SingleSubjectAnalysisConfig:
    blob_size_um: float = 22.38
    max_regions_per_image: int = 5
    scoring_weights: Optional[Dict[str, float]] = None
    effective_count_scoring_weights: Optional[Dict[str, float]] = None
    rfp_channel_index: int = 0
    dapi_channel_index: int = 2
    rfp_gt_threshold: int = 71


def _identify_channel_axis(shape: Tuple[int, ...]) -> Optional[int]:
    if len(shape) < 3:
        return None
    if shape[-1] <= 4 and (len(shape) == 3 or shape[-1] != shape[-2]):
        return -1
    if shape[0] <= 4 and shape[0] != shape[1]:
        return 0
    return None


def _to_2d_channel(image: np.ndarray, preferred_index: int = 0) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        axis = _identify_channel_axis(arr.shape)
        if axis is None:
            axis = -1
        arr = np.moveaxis(arr, axis, -1)
        n_channels = int(arr.shape[-1])
        idx = preferred_index if 0 <= preferred_index < n_channels else 0
        return np.asarray(arr[..., idx])

    raise ValueError(f"Expected 2D or 3D image for channel extraction, got shape {arr.shape}.")


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .strip("_")
    )


def _save_overlay_image(
    *,
    subject_name: str,
    output_dir: Path,
    source_type: str,
    rfp_image: np.ndarray,
    dapi_image: np.ndarray,
    crypt_labels: np.ndarray,
) -> Path:
    overlay_dir = output_dir / "renderings"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def _minmax01(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    overlay_xr = render_label_overlay(
        channels=[_minmax01(rfp_image), _minmax01(dapi_image), crypt_labels],
        fill_alpha=0.35,
        outline_alpha=1.0,
        outline_width=2,
        normalize_scalar=True,
    )
    overlay_rgb = np.moveaxis(overlay_xr.values, 0, -1)

    # Visualize the centroid-fit line used by crypt scoring.
    try:
        curve_model = fit_centroid_curve_from_labels(crypt_labels, rfp_image, max_degree=2)
        curve_pts = sample_centroid_curve_points(
            curve_model,
            crypt_labels.shape,
            num_samples=max(128, int(np.hypot(crypt_labels.shape[0], crypt_labels.shape[1])) * 2),
        )
        if curve_pts.shape[0] >= 2:
            xs = np.rint(curve_pts[:, 0]).astype(np.int32)
            ys = np.rint(curve_pts[:, 1]).astype(np.int32)
            valid = (xs >= 0) & (xs < crypt_labels.shape[1]) & (ys >= 0) & (ys < crypt_labels.shape[0])
            if np.any(valid):
                curve_mask = np.zeros(crypt_labels.shape, dtype=bool)
                curve_mask[ys[valid], xs[valid]] = True
                curve_mask = ndi.binary_dilation(curve_mask, iterations=2)
                curve_color = np.asarray([0.13, 1.0, 0.93], dtype=np.float32)
                curve_alpha = 0.90
                overlay_rgb[curve_mask] = (
                    (1.0 - curve_alpha) * overlay_rgb[curve_mask] + curve_alpha * curve_color
                )
                overlay_rgb = np.clip(overlay_rgb, 0.0, 1.0)
    except Exception:
        pass

    output_path = overlay_dir / f"{_safe_name(subject_name)}_{_safe_name(source_type)}_overlay.png"
    plt.imsave(output_path, overlay_rgb)
    return output_path


def _label_area_stats(labels: np.ndarray) -> Tuple[Optional[int], float, float]:
    if labels is None:
        return None, float("nan"), float("nan")
    arr = np.asarray(labels)
    if arr.size == 0:
        return 0, 0.0, 0.0
    counts = np.bincount(arr.reshape(-1).astype(np.int64))
    if counts.size <= 1:
        return 0, 0.0, 0.0
    counts = counts[1:]
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0, 0.0, 0.0
    return int(counts.size), float(counts.sum()), float(counts.std())


def _count_rfp_pixels_gt_threshold(*, rfp: np.ndarray, labels: np.ndarray, threshold: int) -> Optional[int]:
    rfp_arr = np.asarray(rfp)
    labels_arr = np.asarray(labels)
    if rfp_arr.shape != labels_arr.shape:
        return None
    return int(np.count_nonzero((labels_arr > 0) & (rfp_arr > threshold)))


def _serialize_effective_estimation(estimate: Optional[EffectiveCryptEstimation]) -> Optional[Dict[str, Any]]:
    if estimate is None:
        return None
    return {
        "neff_simpson": float(estimate.neff_simpson),
        "neff_shannon": float(estimate.neff_shannon),
        "k_raw": int(estimate.k_raw),
        "evenness": float(estimate.evenness),
        "selected_labels_k": int(estimate.selected_labels_k),
        "debug_render_path": str(estimate.debug_render_path) if estimate.debug_render_path else "",
    }


def _build_image_rows(
    *,
    subject_name: str,
    source_type: str,
    source_dataset: str,
    source_label: str,
    microns_per_px: float,
    image_pixel_count: int,
    summary: Dict[str, float],
    summary_raw: Dict[str, float],
    effective_estimation: Optional[EffectiveCryptEstimation],
    initial_detected_count: Optional[int],
    selected_crypt_area_px_sum: float,
    selected_crypt_area_px_std: float,
    detected_crypt_area_px_sum: float,
    detected_crypt_area_px_std: float,
    rfp_gt_threshold: int,
    selected_rfp_px_gt_threshold: Optional[int],
    detected_rfp_px_gt_threshold: Optional[int],
) -> Dict[str, Optional[Dict[str, Any]]]:
    crypt_count = float(summary.get("crypt_count", float("nan")))
    area_sum_um2 = float(summary.get("crypt_area_um2_sum", float("nan")))
    area_mean_um2 = float(summary.get("crypt_area_um2_mean", float("nan")))
    rfp_sum_mean = float(summary.get("rfp_sum_mean", float("nan")))
    rfp_sum_total = float(summary.get("rfp_sum_total", float("nan")))
    effective_full_mean_um2 = float(summary.get("effective_full_intensity_um2_mean", float("nan")))

    mean_intensity = float("nan")
    if np.isfinite(rfp_sum_mean):
        mean_intensity = float(rfp_sum_mean) * float(microns_per_px)

    percent_area = float("nan")
    if image_pixel_count > 0:
        image_area_um2 = float(microns_per_px) * float(microns_per_px) * float(image_pixel_count)
        if image_area_um2 > 0:
            percent_area = (float(area_sum_um2) / image_area_um2) * 100.0

    rfp_intensity_um2_sum = float("nan")
    if np.isfinite(rfp_sum_total):
        rfp_intensity_um2_sum = float(rfp_sum_total) * float(microns_per_px) * float(microns_per_px)

    image_summary_row: Dict[str, Any] = {
        "subject name": subject_name,
        "image_source_type": source_type,
        "source_dataset": source_dataset,
        "source_label": source_label,
        "Count": crypt_count,
        "Total Area": area_sum_um2,
        "Average Size": area_mean_um2,
        "% Area": percent_area,
        "Mean": mean_intensity,
        "effective_full_intensity_um2_mean": effective_full_mean_um2,
        "rfp_intensity_um2_sum": rfp_intensity_um2_sum,
        "selected_crypt_area_px_sum": selected_crypt_area_px_sum,
        "selected_crypt_area_px_std": selected_crypt_area_px_std,
        "detected_crypt_area_px_sum": detected_crypt_area_px_sum,
        "detected_crypt_area_px_std": detected_crypt_area_px_std,
        "rfp_gt_threshold": rfp_gt_threshold,
        "selected_rfp_px_gt_threshold": selected_rfp_px_gt_threshold,
        "detected_rfp_px_gt_threshold": detected_rfp_px_gt_threshold,
    }

    detailed_row: Dict[str, Any] = {
        "subject_name": subject_name,
        "image_source_type": source_type,
        "source_dataset": source_dataset,
        "source_label": source_label,
        "microns_per_px": microns_per_px,
        "selected_crypt_area_px_sum": selected_crypt_area_px_sum,
        "selected_crypt_area_px_std": selected_crypt_area_px_std,
        "detected_crypt_area_px_sum": detected_crypt_area_px_sum,
        "detected_crypt_area_px_std": detected_crypt_area_px_std,
        "rfp_gt_threshold": rfp_gt_threshold,
        "selected_rfp_px_gt_threshold": selected_rfp_px_gt_threshold,
        "detected_rfp_px_gt_threshold": detected_rfp_px_gt_threshold,
    }
    for field in SUMMARY_FIELD_ORDER:
        detailed_row[field] = float(summary.get(field, float("nan")))

    raw_fields = (
        "rfp_sum_total",
        "rfp_sum_mean",
        "rfp_sum_std",
        "rfp_intensity_mean",
        "rfp_intensity_std",
        "rfp_intensity_min",
        "rfp_intensity_max",
        "rfp_max_intensity_mean",
        "rfp_max_intensity_std",
        "effective_full_intensity_um2_sum",
        "effective_full_intensity_um2_mean",
        "effective_full_intensity_um2_std",
    )
    for field in raw_fields:
        detailed_row[f"raw_{field}"] = float(summary_raw.get(field, float("nan")))

    image_summary_row_simpson: Optional[Dict[str, Any]] = None
    detailed_row_simpson: Optional[Dict[str, Any]] = None
    if effective_estimation is not None:
        simpson_n = float(effective_estimation.neff_simpson)
        if not np.isfinite(simpson_n) or simpson_n <= 0:
            simpson_n = crypt_count

        area_mean_um2_simpson = float(area_sum_um2 / simpson_n) if np.isfinite(simpson_n) and simpson_n > 0 else float("nan")
        rfp_sum_mean_simpson = float(rfp_sum_total / simpson_n) if np.isfinite(simpson_n) and simpson_n > 0 else float("nan")
        eff_full_sum = float(summary.get("effective_full_intensity_um2_sum", float("nan")))
        eff_full_mean_um2_simpson = (
            float(eff_full_sum / simpson_n) if np.isfinite(eff_full_sum) and np.isfinite(simpson_n) and simpson_n > 0 else float("nan")
        )
        mean_intensity_simpson = (
            float(rfp_sum_mean_simpson) * float(microns_per_px) if np.isfinite(rfp_sum_mean_simpson) else float("nan")
        )

        image_summary_row_simpson = {
            "subject name": subject_name,
            "image_source_type": source_type,
            "source_dataset": source_dataset,
            "source_label": source_label,
            "Effective Count": simpson_n,
            "Total Area": area_sum_um2,
            "Average Size (Simpson)": area_mean_um2_simpson,
            "% Area": percent_area,
            "Mean (Simpson)": mean_intensity_simpson,
            "effective_full_intensity_um2_mean (Simpson)": eff_full_mean_um2_simpson,
            "rfp_intensity_um2_sum": rfp_intensity_um2_sum,
            "selected_crypt_area_px_sum": selected_crypt_area_px_sum,
            "selected_crypt_area_px_std": selected_crypt_area_px_std,
            "detected_crypt_area_px_sum": detected_crypt_area_px_sum,
            "detected_crypt_area_px_std": detected_crypt_area_px_std,
            "rfp_gt_threshold": rfp_gt_threshold,
            "selected_rfp_px_gt_threshold": selected_rfp_px_gt_threshold,
            "detected_rfp_px_gt_threshold": detected_rfp_px_gt_threshold,
        }

        detailed_row_simpson = dict(detailed_row)
        detailed_row_simpson["simpson_effective_count"] = simpson_n
        detailed_row_simpson["simpson_k_raw"] = int(effective_estimation.k_raw)
        detailed_row_simpson["simpson_evenness"] = float(effective_estimation.evenness)
        detailed_row_simpson["simpson_selected_labels_k"] = int(effective_estimation.selected_labels_k)
        detailed_row_simpson["crypt_area_um2_mean_simpson"] = area_mean_um2_simpson
        detailed_row_simpson["rfp_sum_mean_simpson"] = rfp_sum_mean_simpson
        detailed_row_simpson["effective_full_intensity_um2_mean_simpson"] = eff_full_mean_um2_simpson

    return {
        "image_summary_row": image_summary_row,
        "detailed_summary_row": detailed_row,
        "image_summary_row_simpson": image_summary_row_simpson,
        "detailed_summary_row_simpson": detailed_row_simpson,
    }


def analyze_single_subject(
    *,
    lysozyme_path: str | Path,
    tissue_path: str | Path,
    subject_id: str,
    microns_per_pixel: float,
    output_dir: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[SingleSubjectAnalysisConfig] = None,
    save_overlay: bool = True,
    save_effective_count_debug: bool = False,
) -> Dict[str, Any]:
    """
    Run complete lysozyme analysis for one subject.

    Returns a dict containing image-level rows, detailed rows, per-crypt records,
    and optional rendering/debug paths.
    """
    cfg = config or SingleSubjectAnalysisConfig()
    meta = dict(metadata or {})

    lysozyme = Path(lysozyme_path).expanduser().resolve()
    tissue = Path(tissue_path).expanduser().resolve()
    if not lysozyme.exists():
        raise FileNotFoundError(f"Lysozyme image not found: {lysozyme}")
    if not tissue.exists():
        raise FileNotFoundError(f"Tissue image not found: {tissue}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = str(meta.get("source_dataset", "")).strip()
    source_label = str(meta.get("source_label", "")).strip()
    source_type = str(meta.get("image_source_type", "")).strip() or "csv_input"

    rfp_img = _to_2d_channel(imread(str(lysozyme)), preferred_index=cfg.rfp_channel_index)
    dapi_img = _to_2d_channel(imread(str(tissue)), preferred_index=cfg.dapi_channel_index)
    if rfp_img.shape != dapi_img.shape:
        raise ValueError(
            f"RFP and DAPI channel shapes must match for '{subject_id}'. "
            f"Got {rfp_img.shape} and {dapi_img.shape}."
        )

    base_labels, crypt_labels = segment_crypts_dual(
        channels=(rfp_img, dapi_img),
        microns_per_px=float(microns_per_pixel),
        blob_size_um=float(cfg.blob_size_um),
        debug=False,
        max_regions_best=int(cfg.max_regions_per_image),
        scoring_weights=cfg.scoring_weights,
    )

    effective_estimation = estimate_effective_selected_crypt_count(
        best_crypts=crypt_labels,
        base_labels=base_labels,
        rfp_image=rfp_img,
        dapi_image=dapi_img,
        blob_size_um=float(cfg.blob_size_um),
        microns_per_px=float(microns_per_pixel),
        subject_name=subject_id,
        output_dir=out_dir / "renderings",
        scoring_weights=(cfg.effective_count_scoring_weights or cfg.scoring_weights),
        save_debug=bool(save_effective_count_debug),
        expansion_scale=0.5,
    )

    normalized_rfp = np.asarray(
        compute_normalized_rfp(
            rfp_image=rfp_img,
            dapi_image=dapi_img,
            crypt_labels=crypt_labels,
            name=subject_id,
        )
    )
    summary = summarize_crypt_fluorescence(
        normalized_rfp=normalized_rfp,
        crypt_labels=crypt_labels,
        microns_per_px=float(microns_per_pixel),
    )
    summary_raw = summarize_crypt_fluorescence(
        normalized_rfp=rfp_img,
        crypt_labels=crypt_labels,
        microns_per_px=float(microns_per_pixel),
    )
    per_crypt = summarize_crypt_fluorescence_per_crypt(
        normalized_rfp=normalized_rfp,
        crypt_labels=crypt_labels,
        microns_per_px=float(microns_per_pixel),
        subject_name=subject_id,
    )

    initial_detected_count, _, _ = _label_area_stats(base_labels)
    selected_crypt_count, selected_crypt_area_px_sum, selected_crypt_area_px_std = _label_area_stats(crypt_labels)
    detected_crypt_count, detected_crypt_area_px_sum, detected_crypt_area_px_std = _label_area_stats(base_labels)

    threshold = int(cfg.rfp_gt_threshold)
    selected_rfp_px_gt_threshold = _count_rfp_pixels_gt_threshold(
        rfp=rfp_img, labels=crypt_labels, threshold=threshold
    )
    detected_rfp_px_gt_threshold = _count_rfp_pixels_gt_threshold(
        rfp=rfp_img, labels=base_labels, threshold=threshold
    )

    overlay_paths: list[str] = []
    if save_overlay:
        overlay_path = _save_overlay_image(
            subject_name=subject_id,
            output_dir=out_dir,
            source_type=source_type,
            rfp_image=rfp_img,
            dapi_image=dapi_img,
            crypt_labels=crypt_labels,
        )
        overlay_paths.append(str(overlay_path))

    rows = _build_image_rows(
        subject_name=subject_id,
        source_type=source_type,
        source_dataset=source_dataset,
        source_label=source_label,
        microns_per_px=float(microns_per_pixel),
        image_pixel_count=int(rfp_img.shape[0]) * int(rfp_img.shape[1]),
        summary=summary,
        summary_raw=summary_raw,
        effective_estimation=effective_estimation,
        initial_detected_count=initial_detected_count,
        selected_crypt_area_px_sum=selected_crypt_area_px_sum,
        selected_crypt_area_px_std=selected_crypt_area_px_std,
        detected_crypt_area_px_sum=detected_crypt_area_px_sum,
        detected_crypt_area_px_std=detected_crypt_area_px_std,
        rfp_gt_threshold=threshold,
        selected_rfp_px_gt_threshold=selected_rfp_px_gt_threshold,
        detected_rfp_px_gt_threshold=detected_rfp_px_gt_threshold,
    )

    per_crypt_records = []
    for record in per_crypt.get("records", []):
        row = dict(record)
        row["subject_name"] = row.get("subject_name", subject_id)
        row["image_source_type"] = source_type
        row["source_dataset"] = source_dataset
        row["source_label"] = source_label
        per_crypt_records.append(row)

    return {
        "subject_name": subject_id,
        "source_type": source_type,
        "source_dataset": source_dataset,
        "source_label": source_label,
        "microns_per_px": float(microns_per_pixel),
        "image_summary_row": rows["image_summary_row"],
        "detailed_summary_row": rows["detailed_summary_row"],
        "image_summary_row_simpson": rows["image_summary_row_simpson"],
        "detailed_summary_row_simpson": rows["detailed_summary_row_simpson"],
        "per_crypt_records": per_crypt_records,
        "overlay_paths": overlay_paths,
        "effective_crypt_estimation": _serialize_effective_estimation(effective_estimation),
        "initial_detected_count": initial_detected_count,
        "selected_crypt_count": selected_crypt_count,
        "detected_crypt_count": detected_crypt_count,
    }
