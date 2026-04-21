from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.io import imread

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEBASE_ROOT = REPO_ROOT / "codeBase"
CRYPT_CODE_ROOT = CODEBASE_ROOT / "crypt_detection_code"
for path in (CODEBASE_ROOT, CRYPT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lysozyme_stain_quantification.crypts.scoring_selector_mod import (  # type: ignore  # noqa: E402
    fit_centroid_curve_from_labels,
    sample_centroid_curve_points,
)
from lysozyme_stain_quantification.segment_crypts import segment_crypts_dual  # type: ignore  # noqa: E402
from lysozyme_stain_quantification.single_subject import _to_2d_channel  # type: ignore  # noqa: E402
from pipeline_implementations.spark_implementation.cluster import (  # type: ignore  # noqa: E402
    create_local_spark_session,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis" / "lysozyme_pipeline_config.yaml"
)
DEFAULT_CSV = REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis" / "lysozyme_input_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "method_development" / "fix line fit" / "out_batch"


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


def _curve_points(labels: np.ndarray, rfp: np.ndarray) -> np.ndarray:
    curve_model = fit_centroid_curve_from_labels(labels, rfp, max_degree=2)
    if curve_model is None:
        return np.empty((0, 2), dtype=np.float64)
    return sample_centroid_curve_points(
        curve_model,
        labels.shape,
        num_samples=max(128, int(np.hypot(labels.shape[0], labels.shape[1])) * 2),
    )


def _render_debug_figure(
    *,
    overlay_rgb: np.ndarray,
    rfp_norm: np.ndarray,
    curve_pts: np.ndarray,
    subject_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    panels = (
        (overlay_rgb, "Multichannel Overlay"),
        (rfp_norm, "RFP"),
    )
    for ax, (image, title) in zip(axes, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
        if curve_pts.shape[0] >= 2:
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], color="#21FFE1", linewidth=2.5)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"Keyence line-fit debug: {subject_label}", fontsize=14)
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

        _, best_labels = segment_crypts_dual(
            channels=(rfp, dapi),
            blob_size_um=float(task["blob_size_um"]),
            microns_per_px=float(task["microns_per_pixel"]),
            debug=False,
            max_regions_best=int(task["max_regions_per_image"]),
            scoring_weights=dict(task["scoring_weights"]),
        )

        curve_pts = _curve_points(best_labels, rfp)
        overlay_rgb = _normalize_rgb(imread(str(overlay_path)))
        rfp_norm = _normalize_gray(rfp)

        output_path = Path(task["output_dir"]) / f"{_safe_name(subject_id)}_linefit_debug.png"
        _render_debug_figure(
            overlay_rgb=overlay_rgb,
            rfp_norm=rfp_norm,
            curve_pts=curve_pts,
            subject_label=subject_id,
            output_path=output_path,
        )
        return {
            "subject_id": subject_id,
            "status": "ok",
            "output_path": str(output_path),
            "curve_point_count": int(curve_pts.shape[0]),
            "error": "",
        }
    except Exception as exc:
        return {
            "subject_id": str(task.get("subject_id", "")),
            "status": "error",
            "output_path": "",
            "curve_point_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render line-fit debug figures for many Keyence images with PySpark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects.")
    parser.add_argument("--partitions", type=int, default=12, help="Spark partitions.")
    parser.add_argument("--n-workers", type=int, default=12, help="Local Spark worker slots.")
    parser.add_argument("--master", type=str, default=None, help="Explicit Spark master.")
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    dataset_cfg = dict(config.get("dataset_config", {}))
    pipeline_cfg = dict(config.get("pipeline_config", {}))
    spark_cfg = dict(pipeline_cfg.get("spark", {}))

    rows = _load_rows(args.input_csv.resolve())
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise ValueError("No rows available for rendering.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_payloads: list[dict[str, Any]] = []
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
                "scoring_weights": dict(dataset_cfg.get("scoring_weights", {})),
                "output_dir": str(output_dir),
            }
        )

    effective_master = args.master if args.master else spark_cfg.get("master")
    spark = create_local_spark_session(
        app_name="keyence-linefit-debug-render",
        master=effective_master,
        n_workers=int(args.n_workers),
        spark_config=dict(spark_cfg.get("config", {})),
        log_level=str(spark_cfg.get("log_level", "WARN")),
    )

    try:
        partitions = max(1, int(args.partitions))
        results = (
            spark.sparkContext.parallelize(task_payloads, numSlices=partitions)
            .map(_render_one_subject)
            .collect()
        )
    finally:
        spark.stop()

    results = sorted(results, key=lambda item: item["subject_id"])
    manifest_path = output_dir / "manifest.json"
    manifest_csv_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "status", "output_path", "curve_point_count", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for item in results if item["status"] == "ok")
    err_count = sum(1 for item in results if item["status"] != "ok")
    print(f"Rendered {ok_count} subjects with {err_count} errors.")
    print(manifest_csv_path)


if __name__ == "__main__":
    main()
