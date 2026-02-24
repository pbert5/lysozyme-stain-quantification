from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.lysozyme_pipelines import DatasetConfig, ScaleLookup, run_dask_pipeline
from src.lysozyme_pipelines.cli import compute_debug_whitelist
from src.lysozyme_stain_quantification.utils.debug_image_saver import (
    DEFAULT_DEBUG_STAGE_WHITELIST,
)


def _resolve_config_path(*, config_arg: Optional[Path], work_dir_arg: Optional[Path]) -> Path:
    if config_arg is not None and work_dir_arg is not None:
        raise ValueError("Provide either --config or --work-dir, not both.")
    if work_dir_arg is not None:
        return (work_dir_arg.expanduser().resolve() / "lysozyme_pipeline_config.yaml").resolve()
    if config_arg is not None:
        return config_arg.expanduser().resolve()
    return (Path.cwd() / "lysozyme_pipeline_config.yaml").resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at top-level")
    return data


def _dataset_config_from_yaml(cfg: Dict[str, Any], config_path: Path) -> tuple[DatasetConfig, Path]:
    dataset_cfg = cfg.get("dataset_config", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset_config must be a mapping")

    discovery = dataset_cfg.get("discovery", {})
    datasets = discovery.get("datasets", []) if isinstance(discovery, dict) else []
    first_root = None
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            first_root = first.get("root_dir")
    image_base_dir = Path(str(first_root or config_path.parent)).expanduser()

    scale_cfg = dataset_cfg.get("scale_lookup", {}) if isinstance(dataset_cfg.get("scale_lookup", {}), dict) else {}
    scale_lookup = ScaleLookup(
        default_value=float(scale_cfg.get("default_value", 0.4476)),
        keys=tuple(scale_cfg.get("keys", [])),
        values=tuple(float(v) for v in scale_cfg.get("values", [])),
    )

    dcfg = DatasetConfig(
        image_base_dir=image_base_dir,
        exp_name=str(dataset_cfg.get("exp_name", "lysozyme_from_config")),
        blob_size_um=float(dataset_cfg.get("blob_size_um", 22.38)),
        max_regions_per_image=int(dataset_cfg.get("max_regions_per_image", 5)),
        scoring_weights=dict(dataset_cfg.get("scoring_weights", {})),
        effective_count_scoring_weights=dict(dataset_cfg.get("effective_count_scoring_weights", {})),
        scale_lookup=scale_lookup,
        channel_keys=tuple(dataset_cfg.get("channel_keys", ["_CH2", "_CH4"])),
        rfp_gt_threshold=int(dataset_cfg.get("rfp_gt_threshold", 71)),
    )

    input_csv = Path(str(dataset_cfg.get("input_csv", "lysozyme_input_data.csv")))
    if not input_csv.is_absolute():
        input_csv = (config_path.parent / input_csv).resolve()
    return dcfg, input_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lysozyme pipeline from YAML config + discovered input CSV.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory containing lysozyme_pipeline_config.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(config_arg=args.config, work_dir_arg=args.work_dir)
    cfg = _load_yaml(config_path)

    dataset_cfg, input_csv = _dataset_config_from_yaml(cfg, config_path)
    pipeline_cfg = cfg.get("pipeline_config", {})
    if not isinstance(pipeline_cfg, dict):
        raise ValueError("pipeline_config must be a mapping")

    debug_whitelist = compute_debug_whitelist(
        pipeline_cfg.get("debug_stage", []),
        base_whitelist=DEFAULT_DEBUG_STAGE_WHITELIST,
    )

    results_root = Path(str(pipeline_cfg.get("results_root", config_path.parent))).expanduser()
    if not results_root.is_absolute():
        results_root = (config_path.parent / results_root).resolve()

    print(f"Using config: {config_path}")
    print(f"Using input CSV: {input_csv}")

    run_dask_pipeline(
        dataset_cfg=dataset_cfg,
        results_root=results_root,
        input_csv=input_csv,
        use_cluster=bool(pipeline_cfg.get("use_cluster", True)),
        force_respawn_cluster=bool(pipeline_cfg.get("force_respawn_cluster", False)),
        n_workers=pipeline_cfg.get("n_workers", None),
        threads_per_worker=pipeline_cfg.get("threads_per_worker", None),
        save_images=bool(pipeline_cfg.get("save_images", True)),
        debug=bool(pipeline_cfg.get("debug", False)),
        max_subjects=pipeline_cfg.get("max_subjects", None),
        connect_to_existing_cluster=bool(pipeline_cfg.get("connect_to_existing_cluster", False)),
        use_timestamps=bool(pipeline_cfg.get("use_timestamps", False)),
        debug_image_capture=bool(pipeline_cfg.get("debug_image_capture", True)),
        debug_image_whitelist=debug_whitelist,
        debug_subject_limit=pipeline_cfg.get("debug_subject_count", None),
        debug_subject_whitelist=pipeline_cfg.get("debug_subject_whitelist", None),
    )


if __name__ == "__main__":
    main()
