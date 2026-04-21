from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .cluster import create_local_spark_session

PROJECT_CODEBASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_AUTO_PARTITIONS = 8

IMAGE_SUMMARY_COLS = [
    "subject name",
    "image_source_type",
    "source_dataset",
    "source_label",
    "Count",
    "Total Area",
    "Average Size",
    "% Area",
    "Mean",
    "effective_full_intensity_um2_mean",
    "rfp_intensity_um2_sum",
    "selected_crypt_area_px_sum",
    "selected_crypt_area_px_std",
    "detected_crypt_area_px_sum",
    "detected_crypt_area_px_std",
    "rfp_gt_threshold",
    "selected_rfp_px_gt_threshold",
    "detected_rfp_px_gt_threshold",
]

IMAGE_SUMMARY_SIMPSON_COLS = [
    "subject name",
    "image_source_type",
    "source_dataset",
    "source_label",
    "Effective Count",
    "Total Area",
    "Average Size (Simpson)",
    "% Area",
    "Mean (Simpson)",
    "effective_full_intensity_um2_mean (Simpson)",
    "rfp_intensity_um2_sum",
    "selected_crypt_area_px_sum",
    "selected_crypt_area_px_std",
    "detected_crypt_area_px_sum",
    "detected_crypt_area_px_std",
    "rfp_gt_threshold",
    "selected_rfp_px_gt_threshold",
    "detected_rfp_px_gt_threshold",
]

DETAILED_SUMMARY_BASE_COLS = [
    "subject_name",
    "image_source_type",
    "source_dataset",
    "source_label",
    "microns_per_px",
    "selected_crypt_area_px_sum",
    "selected_crypt_area_px_std",
    "detected_crypt_area_px_sum",
    "detected_crypt_area_px_std",
    "rfp_gt_threshold",
    "selected_rfp_px_gt_threshold",
    "detected_rfp_px_gt_threshold",
]


def _setup_results_dir(results_root: Path, exp_name: str) -> Path:
    out = results_root / "results" / exp_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text.strip()


def _run_single_row(record: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        codebase_root = runtime_cfg.get("codebase_root")
        if codebase_root and str(codebase_root) not in sys.path:
            sys.path.insert(0, str(codebase_root))
        project_root = runtime_cfg.get("project_root")
        if project_root and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from crypt_detection_code.lysozyme_stain_quantification import (
            SingleSubjectAnalysisConfig,
            analyze_single_subject,
        )

        subject_id = _safe_string(record.get("subject_id"))
        lysozyme_path = _safe_string(record.get("lysozyme_path"))
        tissue_path = _safe_string(record.get("tissue_path"))
        tissue_aux_path = _safe_string(record.get("tissue_aux_path"))
        tissue_combine_mode = _safe_string(record.get("tissue_combine_mode"))

        mpp = _safe_float(record.get("microns_per_pixel"))
        if mpp is None:
            mpp = _safe_float(runtime_cfg.get("default_microns_per_pixel"))
        if mpp is None:
            raise ValueError(f"Missing microns_per_pixel for subject '{subject_id}'.")

        scoring_weights = runtime_cfg.get("scoring_weights")
        if isinstance(scoring_weights, dict) and not scoring_weights:
            scoring_weights = None
        effective_weights = runtime_cfg.get("effective_count_scoring_weights")
        if isinstance(effective_weights, dict) and not effective_weights:
            effective_weights = None

        cfg = SingleSubjectAnalysisConfig(
            blob_size_um=float(runtime_cfg.get("blob_size_um", 22.38)),
            max_regions_per_image=int(runtime_cfg.get("max_regions_per_image", 5)),
            scoring_weights=scoring_weights,
            effective_count_scoring_weights=effective_weights,
            rfp_channel_index=int(runtime_cfg.get("rfp_channel_index", 0)),
            dapi_channel_index=int(runtime_cfg.get("dapi_channel_index", 2)),
            rfp_gt_threshold=int(runtime_cfg.get("rfp_gt_threshold", 71)),
        )

        payload = analyze_single_subject(
            lysozyme_path=lysozyme_path,
            tissue_path=tissue_path,
            tissue_aux_path=tissue_aux_path or None,
            tissue_combine_mode=tissue_combine_mode or None,
            subject_id=subject_id,
            microns_per_pixel=float(mpp),
            output_dir=runtime_cfg["results_dir"],
            metadata={
                "source_dataset": _safe_string(record.get("source_dataset")),
                "source_label": _safe_string(record.get("source_label")),
                "image_source_type": "csv_input",
            },
            config=cfg,
            save_overlay=bool(runtime_cfg.get("save_images", True)),
            save_effective_count_debug=bool(runtime_cfg.get("save_effective_count_debug", False)),
            save_debug_intermediates=bool(runtime_cfg.get("debug_image_capture", False)),
            debug_stage_whitelist=runtime_cfg.get("debug_stage_whitelist", None),
        )
        return {"ok": True, "subject_id": subject_id, "payload": payload}
    except Exception as exc:  # pragma: no cover - executed in spark worker
        return {
            "ok": False,
            "subject_id": _safe_string(record.get("subject_id")),
            "lysozyme_path": _safe_string(record.get("lysozyme_path")),
            "tissue_path": _safe_string(record.get("tissue_path")),
            "tissue_aux_path": _safe_string(record.get("tissue_aux_path")),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_spark_pipeline(
    *,
    input_csv: Path,
    results_root: Path,
    exp_name: str,
    blob_size_um: float,
    max_regions_per_image: int,
    scoring_weights: Optional[Mapping[str, float]],
    effective_count_scoring_weights: Optional[Mapping[str, float]],
    default_microns_per_pixel: Optional[float],
    rfp_channel_index: int = 0,
    dapi_channel_index: int = 2,
    rfp_gt_threshold: int = 71,
    spark_master: Optional[str] = None,
    spark_app_name: str = "lysozyme-spark-pipeline",
    spark_config: Optional[Mapping[str, str]] = None,
    n_workers: Optional[int] = None,
    max_subjects: Optional[int] = None,
    partitions: Optional[int] = None,
    save_images: bool = True,
    save_effective_count_debug: bool = False,
    debug_image_capture: bool = False,
    debug_stage_whitelist: Optional[List[str]] = None,
    log_level: str = "WARN",
    debug: bool = False,
) -> Dict[str, Any]:
    input_csv = Path(input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    results_root = Path(results_root).expanduser().resolve()
    results_dir = _setup_results_dir(results_root, exp_name=exp_name)

    input_df = pd.read_csv(input_csv)
    required_columns = {"subject_id", "lysozyme_path", "tissue_path"}
    missing = required_columns - set(input_df.columns)
    if missing:
        raise ValueError(f"input_csv missing required columns: {sorted(missing)}")

    records: List[Dict[str, Any]] = []
    skipped_empty_subject = 0
    skipped_missing_paths = 0
    skipped_invalid_paths = 0
    for _, row in input_df.iterrows():
        subject_id = _safe_string(row.get("subject_id"))
        lysozyme_path = _safe_string(row.get("lysozyme_path"))
        tissue_path = _safe_string(row.get("tissue_path"))
        if not subject_id:
            skipped_empty_subject += 1
            continue
        tissue_aux_path = _safe_string(row.get("tissue_aux_path"))
        if not lysozyme_path or not tissue_path:
            skipped_invalid_paths += 1
            continue
        lyso_path_obj = Path(lysozyme_path).expanduser()
        tissue_path_obj = Path(tissue_path).expanduser()
        tissue_aux_path_obj = Path(tissue_aux_path).expanduser() if tissue_aux_path else None
        if (
            not lyso_path_obj.exists()
            or not tissue_path_obj.exists()
            or (tissue_aux_path_obj is not None and not tissue_aux_path_obj.exists())
        ):
            skipped_missing_paths += 1
            continue

        records.append(
            {
                "subject_id": subject_id,
                "lysozyme_path": str(lyso_path_obj.resolve()),
                "tissue_path": str(tissue_path_obj.resolve()),
                "tissue_aux_path": str(tissue_aux_path_obj.resolve()) if tissue_aux_path_obj is not None else "",
                "tissue_combine_mode": _safe_string(row.get("tissue_combine_mode")),
                "microns_per_pixel": row.get("microns_per_pixel"),
                "source_dataset": _safe_string(row.get("source_dataset")),
                "source_label": _safe_string(row.get("source_label")),
            }
        )

    if max_subjects is not None:
        records = records[: max(0, int(max_subjects))]

    if debug:
        print(
            "[spark] CSV ingestion summary: "
            f"loaded={len(records)}, "
            f"skipped_empty_subject={skipped_empty_subject}, "
            f"skipped_missing_paths={skipped_missing_paths}, "
            f"skipped_invalid_paths={skipped_invalid_paths}"
        )

    if not records:
        raise ValueError("No valid rows available for spark analysis after CSV validation.")

    spark = create_local_spark_session(
        app_name=spark_app_name,
        master=spark_master,
        n_workers=n_workers,
        spark_config=spark_config,
        log_level=log_level,
    )
    created_here = True
    runtime_bc = None
    results: List[Dict[str, Any]]

    try:
        runtime_cfg: Dict[str, Any] = {
            "codebase_root": str(PROJECT_CODEBASE_ROOT),
            "project_root": str(PROJECT_ROOT),
            "results_dir": str(results_dir),
            "blob_size_um": float(blob_size_um),
            "max_regions_per_image": int(max_regions_per_image),
            "scoring_weights": dict(scoring_weights) if scoring_weights is not None else None,
            "effective_count_scoring_weights": (
                dict(effective_count_scoring_weights) if effective_count_scoring_weights is not None else None
            ),
            "default_microns_per_pixel": default_microns_per_pixel,
            "rfp_channel_index": int(rfp_channel_index),
            "dapi_channel_index": int(dapi_channel_index),
            "rfp_gt_threshold": int(rfp_gt_threshold),
            "save_images": bool(save_images),
            "save_effective_count_debug": bool(save_effective_count_debug),
            "debug_image_capture": bool(debug_image_capture),
            "debug_stage_whitelist": list(debug_stage_whitelist) if debug_stage_whitelist is not None else None,
        }

        if partitions is None:
            auto_parallelism = max(1, spark.sparkContext.defaultParallelism)
            partitions = min(auto_parallelism, len(records), DEFAULT_MAX_AUTO_PARTITIONS)
            if debug:
                print(
                    "[spark] Auto-selected partitions="
                    f"{partitions} (defaultParallelism={auto_parallelism}, "
                    f"records={len(records)}, cap={DEFAULT_MAX_AUTO_PARTITIONS})"
                )
        partitions = max(1, int(partitions))
        runtime_bc = spark.sparkContext.broadcast(runtime_cfg)
        try:
            results = (
                spark.sparkContext.parallelize(records, numSlices=partitions)
                .map(lambda rec: _run_single_row(rec, runtime_bc.value))
                .collect()
            )
        finally:
            runtime_bc.unpersist(blocking=False)

    finally:
        if created_here:
            spark.stop()

    successes = [item for item in results if bool(item.get("ok"))]
    failures = [item for item in results if not bool(item.get("ok"))]

    image_summary_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    image_summary_simpson_rows: List[Dict[str, Any]] = []
    detailed_simpson_rows: List[Dict[str, Any]] = []
    per_crypt_rows: List[Dict[str, Any]] = []

    for item in successes:
        payload = item.get("payload", {}) or {}
        image_row = payload.get("image_summary_row")
        if isinstance(image_row, dict):
            image_summary_rows.append(image_row)

        detailed_row = payload.get("detailed_summary_row")
        if isinstance(detailed_row, dict):
            detailed_rows.append(detailed_row)

        image_row_simpson = payload.get("image_summary_row_simpson")
        if isinstance(image_row_simpson, dict):
            image_summary_simpson_rows.append(image_row_simpson)

        detailed_row_simpson = payload.get("detailed_summary_row_simpson")
        if isinstance(detailed_row_simpson, dict):
            detailed_simpson_rows.append(detailed_row_simpson)

        per_crypt_rows.extend(payload.get("per_crypt_records", []) or [])

    image_summary_df = pd.DataFrame(image_summary_rows)
    if not image_summary_df.empty:
        image_summary_df = image_summary_df.reindex(columns=[c for c in IMAGE_SUMMARY_COLS if c in image_summary_df.columns])

    detailed_df = pd.DataFrame(detailed_rows)
    if not detailed_df.empty:
        ordered_cols = [c for c in DETAILED_SUMMARY_BASE_COLS if c in detailed_df.columns]
        remaining = [c for c in detailed_df.columns if c not in ordered_cols]
        detailed_df = detailed_df.reindex(columns=ordered_cols + remaining)

    image_summary_simpson_df = pd.DataFrame(image_summary_simpson_rows)
    if not image_summary_simpson_df.empty:
        image_summary_simpson_df = image_summary_simpson_df.reindex(
            columns=[c for c in IMAGE_SUMMARY_SIMPSON_COLS if c in image_summary_simpson_df.columns]
        )

    detailed_simpson_df = pd.DataFrame(detailed_simpson_rows)
    if not detailed_simpson_df.empty:
        ordered_cols = [c for c in DETAILED_SUMMARY_BASE_COLS if c in detailed_simpson_df.columns]
        remaining = [c for c in detailed_simpson_df.columns if c not in ordered_cols]
        detailed_simpson_df = detailed_simpson_df.reindex(columns=ordered_cols + remaining)

    per_crypt_df = pd.DataFrame(per_crypt_rows)
    failures_df = pd.DataFrame(failures)

    image_summary_path = results_dir / "simple_spark_image_summary.csv"
    detailed_path = results_dir / "simple_spark_image_summary_detailed.csv"
    simpson_image_path = results_dir / "simple_spark_image_summary_simpson.csv"
    simpson_detailed_path = results_dir / "simple_spark_image_summary_detailed_simpson.csv"
    per_crypt_path = results_dir / "simple_spark_per_crypt.csv"
    errors_path = results_dir / "simple_spark_errors.csv"

    if not image_summary_df.empty:
        image_summary_df.to_csv(image_summary_path, index=False)
    if not detailed_df.empty:
        detailed_df.to_csv(detailed_path, index=False)
    if not image_summary_simpson_df.empty:
        image_summary_simpson_df.to_csv(simpson_image_path, index=False)
    if not detailed_simpson_df.empty:
        detailed_simpson_df.to_csv(simpson_detailed_path, index=False)
    if not per_crypt_df.empty:
        per_crypt_df.to_csv(per_crypt_path, index=False)
    if not failures_df.empty:
        failures_df.to_csv(errors_path, index=False)

    summary = {
        "results_dir": str(results_dir),
        "processed_subjects": len(successes),
        "failed_subjects": len(failures),
        "image_summary_csv": str(image_summary_path) if not image_summary_df.empty else "",
        "detailed_summary_csv": str(detailed_path) if not detailed_df.empty else "",
        "image_summary_simpson_csv": str(simpson_image_path) if not image_summary_simpson_df.empty else "",
        "detailed_summary_simpson_csv": str(simpson_detailed_path) if not detailed_simpson_df.empty else "",
        "per_crypt_csv": str(per_crypt_path) if not per_crypt_df.empty else "",
        "errors_csv": str(errors_path) if not failures_df.empty else "",
    }
    if debug:
        print(
            "[spark] Completed run: "
            f"processed={summary['processed_subjects']}, failed={summary['failed_subjects']}, "
            f"results_dir={summary['results_dir']}"
        )
    return summary
