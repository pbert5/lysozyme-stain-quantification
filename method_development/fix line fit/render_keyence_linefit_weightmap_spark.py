from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import colors
from skimage.io import imread
from skimage.segmentation import find_boundaries

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEBASE_ROOT = REPO_ROOT / "codeBase"
CRYPT_CODE_ROOT = CODEBASE_ROOT / "crypt_detection_code"
for path in (CODEBASE_ROOT, CRYPT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lysozyme_stain_quantification.crypts.identify_potential_crypts_ import (  # type: ignore  # noqa: E402
    identify_potential_crypts,
)
from lysozyme_stain_quantification.crypts.remove_edge_touching_regions_mod import (  # type: ignore  # noqa: E402
    remove_edge_touching_regions_sk,
)
from lysozyme_stain_quantification.crypts.scoring_selector_mod import (  # type: ignore  # noqa: E402
    centroid_curve_distances,
    fit_weighted_centroid_curve,
    scoring_selector,
    sample_centroid_curve_points,
)
from lysozyme_stain_quantification.single_subject import _to_2d_channel  # type: ignore  # noqa: E402
from pipeline_implementations.spark_implementation.cluster import (  # type: ignore  # noqa: E402
    create_local_spark_session,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis" / "lysozyme_pipeline_config.yaml"
)
DEFAULT_CSV = REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis" / "lysozyme_input_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "method_development" / "fix line fit" / "out_weight_maps_threshold_area"
def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return data


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_gray(image: np.ndarray, low_q: float = 1.0, high_q: float = 99.5) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, low_q))
    hi = float(np.percentile(arr, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got shape {arr.shape}")
    rgb = arr[..., :3]
    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(3):
        out[..., idx] = _normalize_gray(rgb[..., idx])
    return out


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


def _find_overlay_path(subject_dir: Path) -> Path | None:
    overlays = sorted(subject_dir.glob("*Overlay.*"))
    return overlays[0] if overlays else None


def _weight_map_from_properties(
    labels: np.ndarray,
    properties_df: pd.DataFrame,
    *,
    weight_column: str,
) -> np.ndarray:
    weight_map = np.zeros_like(labels, dtype=np.float32)
    if properties_df is None or len(properties_df) == 0:
        return weight_map
    for _, row in properties_df.iterrows():
        label_id = int(row["label_id"])
        weight_val = float(row.get(weight_column, 0.0))
        weight_map[labels == label_id] = weight_val
    return weight_map


def _threshold_area_map_from_labels(
    labels: np.ndarray,
    rfp: np.ndarray,
    *,
    threshold: int,
) -> tuple[np.ndarray, dict[int, float]]:
    labels_arr = np.asarray(labels)
    rfp_arr = np.asarray(rfp)
    weight_map = np.zeros_like(labels_arr, dtype=np.float32)
    per_label_weights: dict[int, float] = {}
    for label_id in np.unique(labels_arr):
        label_val = int(label_id)
        if label_val <= 0:
            continue
        mask = labels_arr == label_val
        weight_val = float(np.count_nonzero(mask & (rfp_arr > int(threshold))))
        per_label_weights[label_val] = weight_val
        weight_map[mask] = weight_val
    return weight_map, per_label_weights


def _table_text(properties_df: pd.DataFrame, selected_labels: list[int], *, weight_column: str) -> str:
    if properties_df is None or len(properties_df) == 0:
        return "No regions"
    rows = []
    df = properties_df.copy()
    df["selected"] = df["label_id"].isin(selected_labels)
    df = df.sort_values(weight_column, ascending=False)
    for _, row in df.iterrows():
        rows.append(
            f"L{int(row['label_id']):02d} "
            f"{'*' if bool(row['selected']) else ' '} "
            f"w={float(row[weight_column]):8.1f} "
            f"a={float(row['area']):6.0f} "
            f"d={float(row['custom_normalized_line_distance']):5.2f} "
            f"q={float(row['quality_score']):5.2f}"
        )
    return "\n".join(rows)


def _compute_custom_curve_and_metrics(
    properties_df: pd.DataFrame,
    *,
    weight_column: str,
    explicit_weights: dict[int, float] | None = None,
    image_shape: tuple[int, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    df = properties_df.copy()
    if len(df) < 2 or "physical_com" not in df.columns:
        df["custom_distance_from_curve"] = 0.0
        df["custom_normalized_line_distance"] = 0.0
        return df, np.empty((0, 2), dtype=np.float64)

    centers = np.asarray(list(df["physical_com"]), dtype=np.float64)
    y_coords = centers[:, 0]
    x_coords = centers[:, 1]
    if explicit_weights is not None:
        weights = np.asarray(
            [float(explicit_weights.get(int(label_id), 0.0)) for label_id in df["label_id"]],
            dtype=np.float64,
        )
        df[weight_column] = weights
    else:
        weights = np.asarray(df[weight_column], dtype=np.float64)
    curve_model = fit_weighted_centroid_curve(
        x_coords=x_coords,
        y_coords=y_coords,
        weights=weights,
        max_degree=2,
    )
    if curve_model is None:
        distances = np.zeros_like(x_coords, dtype=np.float64)
        curve_points = np.empty((0, 2), dtype=np.float64)
    else:
        distances = centroid_curve_distances(x_coords, y_coords, curve_model)
        curve_points = sample_centroid_curve_points(curve_model, image_shape)

    areas = df["area"].to_numpy(dtype=np.float64)
    radius_approx = np.sqrt(areas / 2.0)
    radius_approx[radius_approx <= 0.0] = 1.0
    df["custom_distance_from_curve"] = distances
    df["custom_normalized_line_distance"] = distances / radius_approx
    return df, curve_points


def _render_weightmap_figure(
    *,
    subject_id: str,
    overlay_rgb: np.ndarray,
    rfp_norm: np.ndarray,
    weight_map: np.ndarray,
    properties_df: pd.DataFrame,
    cleaned_labels: np.ndarray,
    curve_points_xy: np.ndarray,
    selected_labels: list[int],
    weight_column: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    ax_overlay = axes[0, 0]
    ax_rfp = axes[0, 1]
    ax_weight = axes[1, 0]
    ax_table = axes[1, 1]

    ax_overlay.imshow(overlay_rgb)
    ax_overlay.set_title("Multichannel Overlay")
    ax_overlay.set_axis_off()

    ax_rfp.imshow(rfp_norm, cmap="gray", vmin=0.0, vmax=1.0)
    ax_rfp.set_title("RFP")
    ax_rfp.set_axis_off()

    positive = weight_map[weight_map > 0]
    if positive.size > 0:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
        im = ax_weight.imshow(weight_map + 1e-6, cmap="magma", norm=norm)
        fig.colorbar(im, ax=ax_weight, fraction=0.046, pad=0.04)
    else:
        ax_weight.imshow(weight_map, cmap="magma")
    ax_weight.set_title("Exact line-fit weight map")
    ax_weight.set_axis_off()

    boundary_mask = find_boundaries(cleaned_labels, mode="outer")
    for ax in (ax_overlay, ax_rfp, ax_weight):
        ax.contour(boundary_mask.astype(np.uint8), levels=[0.5], colors=["#FFD84D"], linewidths=0.5)

    if curve_points_xy.shape[0] >= 2:
        for ax in (ax_overlay, ax_rfp, ax_weight):
            ax.plot(curve_points_xy[:, 0], curve_points_xy[:, 1], color="#21FFE1", linewidth=2.5)

    if properties_df is not None and len(properties_df) > 0:
        for _, row in properties_df.iterrows():
            y_coord, x_coord = row["physical_com"]
            label_id = int(row["label_id"])
            is_selected = label_id in selected_labels
            marker_color = "#00FFAA" if is_selected else "#FFFFFF"
            for ax in (ax_overlay, ax_rfp, ax_weight):
                ax.scatter([x_coord], [y_coord], s=40 if is_selected else 20, c=marker_color, edgecolors="black", linewidths=0.5)
            ax_weight.text(
                float(x_coord) + 6.0,
                float(y_coord),
                f"L{label_id}",
                color="white",
                fontsize=8,
                ha="left",
                va="center",
            )

    ax_table.set_axis_off()
    ax_table.set_title("Per-label weights and distances")
    ax_table.text(
        0.0,
        1.0,
        _table_text(properties_df, selected_labels, weight_column=weight_column),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_table.transAxes,
    )

    fig.suptitle(f"Keyence line-fit weight map: {subject_id}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_one_subject(task: dict[str, Any]) -> dict[str, Any]:
    try:
        subject_id = str(task["subject_id"])
        rfp_path = Path(task["lysozyme_path"]).expanduser().resolve()
        dapi_path = Path(task["tissue_path"]).expanduser().resolve()
        subject_dir = rfp_path.parent
        overlay_path = _find_overlay_path(subject_dir)
        if overlay_path is None:
            raise FileNotFoundError(f"No overlay image found in {subject_dir}")

        rfp = _to_2d_channel(imread(str(rfp_path)), preferred_index=int(task["rfp_channel_index"]))
        dapi = _to_2d_channel(imread(str(dapi_path)), preferred_index=int(task["dapi_channel_index"]))
        if rfp.shape != dapi.shape:
            raise ValueError(f"RFP and DAPI shape mismatch: {rfp.shape} vs {dapi.shape}")

        blob_size_um = float(task["blob_size_um"])
        microns_per_pixel = float(task["microns_per_pixel"])
        effective_blob_size_px = max(1, int(round(blob_size_um / microns_per_pixel)))

        potential = identify_potential_crypts(rfp, dapi, effective_blob_size_px, False)
        cleaned = remove_edge_touching_regions_sk(potential)
        best_labels, debug_info = scoring_selector(
            cleaned,
            rfp,
            debug=False,
            max_regions=int(task["max_regions_per_image"]),
            weights=dict(task["scoring_weights"]),
            return_details=True,
        )

        properties_df = pd.DataFrame(debug_info.get("properties_df", pd.DataFrame())).copy()
        selected_labels = [int(v) for v in debug_info.get("selected_labels", [])]
        weight_column = str(task["weight_column"])
        explicit_weights = None
        if weight_column == "rfp_area_gt_threshold":
            weight_map, per_label_weights = _threshold_area_map_from_labels(
                cleaned,
                rfp,
                threshold=int(task["rfp_gt_threshold"]),
            )
            explicit_weights = per_label_weights
        else:
            weight_map = _weight_map_from_properties(cleaned, properties_df, weight_column=weight_column)
        properties_df, curve_points_xy = _compute_custom_curve_and_metrics(
            properties_df,
            weight_column=weight_column,
            explicit_weights=explicit_weights,
            image_shape=cleaned.shape,
        )
        if weight_column != "rfp_area_gt_threshold":
            weight_map = _weight_map_from_properties(cleaned, properties_df, weight_column=weight_column)
        overlay_rgb = _normalize_rgb(imread(str(overlay_path)))
        rfp_norm = _normalize_gray(rfp)

        output_dir = Path(task["output_dir"])
        suffix = str(task["output_suffix"])
        figure_path = output_dir / f"{_safe_name(subject_id)}_{suffix}_linefit_weightmap.png"
        csv_path = output_dir / f"{_safe_name(subject_id)}_{suffix}_linefit_weights.csv"

        _render_weightmap_figure(
            subject_id=subject_id,
            overlay_rgb=overlay_rgb,
            rfp_norm=rfp_norm,
            weight_map=weight_map,
            properties_df=properties_df,
            cleaned_labels=cleaned,
            curve_points_xy=curve_points_xy,
            selected_labels=selected_labels,
            weight_column=weight_column,
            output_path=figure_path,
        )

        export_df = properties_df.copy()
        export_df["selected"] = export_df["label_id"].isin(selected_labels)
        if "physical_com" in export_df.columns:
            export_df["centroid_y"] = export_df["physical_com"].apply(lambda value: float(value[0]))
            export_df["centroid_x"] = export_df["physical_com"].apply(lambda value: float(value[1]))
            export_df = export_df.drop(columns=["physical_com"])
        export_df.to_csv(csv_path, index=False)

        return {
            "subject_id": subject_id,
            "status": "ok",
            "figure_path": str(figure_path),
            "csv_path": str(csv_path),
            "label_count": int(len(properties_df)),
            "selected_count": int(len(selected_labels)),
            "error": "",
        }
    except Exception as exc:
        return {
            "subject_id": str(task.get("subject_id", "")),
            "status": "error",
            "figure_path": "",
            "csv_path": "",
            "label_count": 0,
            "selected_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render exact line-fit weight maps for selected Keyence subjects with Spark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        help="Subject id to render. Repeat for multiple.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of subjects after filtering.",
    )
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--master", type=str, default=None)
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="threshold_area",
        choices=("mean_rfp", "total_red_intensity", "threshold_area"),
        help="Which per-label weight to use for the debug curve fit.",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    dataset_cfg = dict(config.get("dataset_config", {}))
    pipeline_cfg = dict(config.get("pipeline_config", {}))
    spark_cfg = dict(pipeline_cfg.get("spark", {}))
    all_rows = _load_rows(args.input_csv.resolve())
    if args.subjects:
        requested_set = set(args.subjects)
        rows = [row for row in all_rows if row["subject_id"] in requested_set]
    else:
        rows = list(all_rows)
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise ValueError("No matching subjects found in input CSV.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.weight_mode == "mean_rfp":
        weight_column = "red_intensity_per_area"
        output_suffix = "mean_rfp"
    elif args.weight_mode == "total_red_intensity":
        weight_column = "total_red_intensity"
        output_suffix = "total_red"
    else:
        weight_column = "rfp_area_gt_threshold"
        output_suffix = "threshold_area"

    task_payloads = []
    for row in rows:
        task_payloads.append(
            {
                "subject_id": row["subject_id"],
                "lysozyme_path": row["lysozyme_path"],
                "tissue_path": row["tissue_path"],
                "microns_per_pixel": float(row["microns_per_pixel"]),
                "blob_size_um": float(dataset_cfg.get("blob_size_um", 22.38)),
                "max_regions_per_image": int(dataset_cfg.get("max_regions_per_image", 5)),
                "rfp_channel_index": int(dataset_cfg.get("rfp_channel_index", 0)),
                "dapi_channel_index": int(dataset_cfg.get("dapi_channel_index", 2)),
                "rfp_gt_threshold": int(dataset_cfg.get("rfp_gt_threshold", 71)),
                "scoring_weights": dict(dataset_cfg.get("scoring_weights", {})),
                "weight_column": weight_column,
                "output_suffix": output_suffix,
                "output_dir": str(output_dir),
            }
        )

    effective_master = args.master if args.master else spark_cfg.get("master")
    spark = create_local_spark_session(
        app_name="keyence-linefit-weightmap-render",
        master=effective_master,
        n_workers=int(args.n_workers),
        spark_config=dict(spark_cfg.get("config", {})),
        log_level=str(spark_cfg.get("log_level", "WARN")),
    )
    try:
        results = (
            spark.sparkContext.parallelize(task_payloads, numSlices=max(1, int(args.partitions)))
            .map(_render_one_subject)
            .collect()
        )
    finally:
        spark.stop()

    results = sorted(results, key=lambda item: item["subject_id"])
    manifest_json = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "status", "figure_path", "csv_path", "label_count", "selected_count", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for item in results if item["status"] == "ok")
    err_count = sum(1 for item in results if item["status"] != "ok")
    print(f"Rendered {ok_count} subjects with {err_count} errors.")
    print(manifest_csv)


if __name__ == "__main__":
    main()
