from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from .pipeline import run_spark_pipeline
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_implementations.spark_implementation.pipeline import run_spark_pipeline


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    return data


def _resolve_config_path(*, config_arg: Optional[Path], work_dir_arg: Optional[Path]) -> Path:
    if config_arg is not None and work_dir_arg is not None:
        raise ValueError("Provide either --config or --work-dir, not both.")
    if work_dir_arg is not None:
        return (work_dir_arg.expanduser().resolve() / "lysozyme_pipeline_config.yaml").resolve()
    if config_arg is not None:
        return config_arg.expanduser().resolve()
    return (Path.cwd() / "lysozyme_pipeline_config.yaml").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spark-based lysozyme pipeline from YAML config.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory containing lysozyme_pipeline_config.yaml.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Override max subjects from config.")
    parser.add_argument("--partitions", type=int, default=None, help="Spark RDD partitions override.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose spark pipeline logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(config_arg=args.config, work_dir_arg=args.work_dir)
    cfg = _load_yaml(config_path)

    dataset_cfg = cfg.get("dataset_config", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset_config must be a mapping.")
    pipeline_cfg = cfg.get("pipeline_config", {})
    if not isinstance(pipeline_cfg, dict):
        raise ValueError("pipeline_config must be a mapping.")
    spark_cfg = pipeline_cfg.get("spark", {})
    if not isinstance(spark_cfg, dict):
        spark_cfg = {}

    input_csv = Path(str(dataset_cfg.get("input_csv", "lysozyme_input_data.csv")))
    if not input_csv.is_absolute():
        input_csv = (config_path.parent / input_csv).resolve()

    results_root = Path(str(pipeline_cfg.get("results_root", config_path.parent))).expanduser()
    if not results_root.is_absolute():
        results_root = (config_path.parent / results_root).resolve()

    scale_lookup = dataset_cfg.get("scale_lookup", {})
    if not isinstance(scale_lookup, dict):
        scale_lookup = {}
    default_mpp = scale_lookup.get("default_value", None)

    summary = run_spark_pipeline(
        input_csv=input_csv,
        results_root=results_root,
        exp_name=str(dataset_cfg.get("exp_name", "lysozyme_spark_run")),
        blob_size_um=float(dataset_cfg.get("blob_size_um", 22.38)),
        max_regions_per_image=int(dataset_cfg.get("max_regions_per_image", 5)),
        scoring_weights=dataset_cfg.get("scoring_weights", None),
        effective_count_scoring_weights=dataset_cfg.get("effective_count_scoring_weights", None),
        default_microns_per_pixel=float(default_mpp) if default_mpp is not None else None,
        rfp_channel_index=int(dataset_cfg.get("rfp_channel_index", 0)),
        dapi_channel_index=int(dataset_cfg.get("dapi_channel_index", 2)),
        rfp_gt_threshold=int(dataset_cfg.get("rfp_gt_threshold", 71)),
        spark_master=spark_cfg.get("master", None),
        spark_app_name=str(spark_cfg.get("app_name", "lysozyme-spark-pipeline")),
        spark_config=spark_cfg.get("config", None),
        n_workers=pipeline_cfg.get("n_workers", spark_cfg.get("n_workers", None)),
        max_subjects=args.max_subjects if args.max_subjects is not None else pipeline_cfg.get("max_subjects", None),
        partitions=args.partitions if args.partitions is not None else spark_cfg.get("partitions", None),
        save_images=bool(pipeline_cfg.get("save_images", True)),
        save_effective_count_debug=bool(pipeline_cfg.get("save_effective_count_debug", False)),
        log_level=str(spark_cfg.get("log_level", "WARN")),
        debug=bool(args.debug or pipeline_cfg.get("debug", False)),
    )

    print("Spark pipeline completed.")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
