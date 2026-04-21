from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from image_utils.discover_lysozyme_images import (
    confirm_overwrite,
    csv_path_from_config,
    discover_rows,
    load_yaml,
    validate_existing_csv,
    write_csv,
)
from reporting_tools.statistical_validation.basic_stats import run_basic_stats_analysis

CONFIG_FILENAME = "lysozyme_pipeline_config.yaml"


def _default_config(work_dir: Path) -> Dict[str, Any]:
    return {
        "dataset_config": {
            "exp_name": "lysozyme_spark_run",
            "input_csv": "lysozyme_input_data.csv",
            "blob_size_um": 22.38,
            "max_regions_per_image": 5,
            "rfp_gt_threshold": 71,
            "rfp_channel_index": 0,
            "dapi_channel_index": 2,
            "channel_keys": ["_RFP", "_DAPI"],
            "scoring_weights": {
                "circularity": 0.15,
                "area": 0.25,
                "line_fit": 0.35,
                "red_intensity": 0.85,
                "com_consistency": 0.05,
            },
            "effective_count_scoring_weights": {
                "circularity": 0.35,
                "area": 0.15,
                "line_fit": 0.45,
                "red_intensity": 0.25,
            },
            "scale_lookup": {
                "default_value": 0.4476,
                "keys": ["40x"],
                "values": [0.2253],
            },
            "discovery": {
                "datasets": [
                    {
                        "name": "default_dataset",
                        "mode": "token_match",
                        "recursive": True,
                        "root_dir": str((work_dir / "images").resolve()),
                        "include_extensions": [".tif", ".tiff", ".jpg", ".jpeg", ".png"],
                        "exclude_name_tokens": ["overlay"],
                        "allow_combined_single_file": True,
                        "microns_per_pixel": 0.4476,
                        "channel_tokens": {
                            "lysozyme": ["_RFP", "_CH2", "c2"],
                            "tissue": ["_DAPI", "_CH4", "c1"],
                        },
                    }
                ]
            },
        },
        "pipeline_config": {
            "backend": "spark",
            "results_root": str(work_dir.resolve()),
            "use_cluster": True,
            "force_respawn_cluster": False,
            "connect_to_existing_cluster": False,
            "n_workers": None,
            "threads_per_worker": None,
            "save_images": True,
            "debug": False,
            "max_subjects": None,
            "use_timestamps": False,
            "debug_image_capture": True,
            "debug_subject_count": 1,
            "debug_subject_whitelist": [],
            "debug_stage": [],
            "discovery_datasets_to_use": [],
            "spark": {
                "master": None,
                "app_name": "lysozyme-spark-pipeline",
                "partitions": None,
                "log_level": "WARN",
                "config": {},
            },
        },
    }


def _resolve_config_path(*, config_arg: Optional[Path], work_dir_arg: Optional[Path]) -> Path:
    if config_arg is not None and work_dir_arg is not None:
        raise ValueError("Provide either --config or --work-dir, not both.")
    if work_dir_arg is not None:
        return (work_dir_arg.expanduser().resolve() / CONFIG_FILENAME).resolve()
    if config_arg is not None:
        return config_arg.expanduser().resolve()
    return (Path.cwd() / CONFIG_FILENAME).resolve()


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _ensure_config(config_path: Path, *, assume_yes: bool) -> Tuple[Dict[str, Any], bool]:
    created = False
    if not config_path.exists():
        template = _default_config(config_path.parent)
        _write_yaml(config_path, template)
        created = True
        print(f"Created config template: {config_path}")

    cfg = load_yaml(config_path)
    if created and not assume_yes:
        if sys.stdin.isatty():
            print("")
            print("Review the generated config, then press Enter to continue.")
            print("Use Ctrl+C to stop, edit config, and rerun when ready.")
            input()
        else:
            raise RuntimeError(
                f"Created {config_path} but cannot pause for confirmation in a non-interactive shell. "
                "Rerun with --yes after updating the config."
            )
    return cfg, created


def _run_discovery_step(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    rewrite_policy: str,
    validate_csv: bool,
) -> Path:
    csv_path = csv_path_from_config(cfg, config_path)
    existing_stats = validate_existing_csv(csv_path) if validate_csv else {"rows": 0, "missing_rows": 0}
    if validate_csv and existing_stats.get("rows", 0) > 0:
        print(
            "Existing CSV check: "
            f"rows={existing_stats['rows']}, "
            f"rows_with_missing_paths={existing_stats['missing_rows']}, "
            f"duplicate_subject_ids={existing_stats.get('duplicate_subject_ids', 0)}"
        )

    should_rebuild = False
    if not csv_path.exists():
        should_rebuild = True
    elif rewrite_policy == "always":
        should_rebuild = True
    elif rewrite_policy == "ask":
        should_rebuild = confirm_overwrite(csv_path=csv_path, policy="ask")
    elif rewrite_policy == "never":
        should_rebuild = False
        if validate_csv and existing_stats.get("missing_rows", 0) > 0:
            raise ValueError(
                f"Existing CSV has {existing_stats['missing_rows']} rows with missing files. "
                "Use --rewrite-csv always (or ask) to rebuild."
            )

    if not should_rebuild:
        print(f"Discovery skipped; using existing CSV: {csv_path}")
        return csv_path

    rows = discover_rows(cfg)
    write_csv(csv_path, rows)
    print(f"Discovery completed: {len(rows)} rows written to {csv_path}")
    return csv_path


def _dataset_config_from_yaml(cfg: Dict[str, Any], config_path: Path):
    from pipeline_implementations.dask_implementation import DatasetConfig, ScaleLookup

    dataset_cfg = cfg.get("dataset_config", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset_config must be a mapping.")

    discovery = dataset_cfg.get("discovery", {})
    datasets = discovery.get("datasets", []) if isinstance(discovery, dict) else []
    first_root = None
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            first_root = first.get("root_dir")
    image_base_dir = Path(str(first_root or config_path.parent)).expanduser()

    scale_cfg = dataset_cfg.get("scale_lookup", {})
    if not isinstance(scale_cfg, dict):
        scale_cfg = {}
    scale_lookup = ScaleLookup(
        default_value=float(scale_cfg.get("default_value", 0.4476)),
        keys=tuple(scale_cfg.get("keys", [])),
        values=tuple(float(v) for v in scale_cfg.get("values", [])),
    )

    scoring_weights = dataset_cfg.get("scoring_weights", {})
    if not isinstance(scoring_weights, dict):
        scoring_weights = {}
    effective_count_weights = dataset_cfg.get("effective_count_scoring_weights", None)
    if effective_count_weights is not None and not isinstance(effective_count_weights, dict):
        effective_count_weights = None

    return DatasetConfig(
        image_base_dir=image_base_dir,
        exp_name=str(dataset_cfg.get("exp_name", "lysozyme_from_config")),
        blob_size_um=float(dataset_cfg.get("blob_size_um", 22.38)),
        max_regions_per_image=int(dataset_cfg.get("max_regions_per_image", 5)),
        scoring_weights=dict(scoring_weights),
        effective_count_scoring_weights=(
            dict(effective_count_weights)
            if effective_count_weights is not None
            else None
        ),
        scale_lookup=scale_lookup,
        channel_keys=tuple(dataset_cfg.get("channel_keys", ["_RFP", "_DAPI"])),
        include_unmatched_combined=bool(dataset_cfg.get("include_unmatched_combined", True)),
        rfp_channel_index=int(dataset_cfg.get("rfp_channel_index", 0)),
        dapi_channel_index=int(dataset_cfg.get("dapi_channel_index", 2)),
        rfp_gt_threshold=int(dataset_cfg.get("rfp_gt_threshold", 71)),
    )


def _run_dask_backend(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    input_csv: Path,
    results_root_override: Optional[Path],
    max_subjects_override: Optional[int],
    debug_override: Optional[bool],
) -> Dict[str, str]:
    from crypt_detection_code.lysozyme_stain_quantification.utils.debug_image_saver import (
        DEFAULT_DEBUG_STAGE_WHITELIST,
    )
    from pipeline_implementations.dask_implementation.cli import compute_debug_whitelist
    from pipeline_implementations.dask_implementation.pipeline import run_dask_pipeline

    dataset_cfg = _dataset_config_from_yaml(cfg, config_path)
    pipeline_cfg = cfg.get("pipeline_config", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}

    debug_whitelist = compute_debug_whitelist(
        pipeline_cfg.get("debug_stage", []),
        base_whitelist=DEFAULT_DEBUG_STAGE_WHITELIST,
    )

    results_root = results_root_override or Path(str(pipeline_cfg.get("results_root", config_path.parent))).expanduser()
    if not results_root.is_absolute():
        results_root = (config_path.parent / results_root).resolve()

    run_dask_pipeline(
        dataset_cfg=dataset_cfg,
        results_root=results_root,
        input_csv=input_csv,
        use_cluster=bool(pipeline_cfg.get("use_cluster", True)),
        force_respawn_cluster=bool(pipeline_cfg.get("force_respawn_cluster", False)),
        n_workers=pipeline_cfg.get("n_workers", None),
        threads_per_worker=pipeline_cfg.get("threads_per_worker", None),
        save_images=bool(pipeline_cfg.get("save_images", True)),
        debug=bool(pipeline_cfg.get("debug", False) if debug_override is None else debug_override),
        max_subjects=max_subjects_override if max_subjects_override is not None else pipeline_cfg.get("max_subjects", None),
        connect_to_existing_cluster=bool(pipeline_cfg.get("connect_to_existing_cluster", False)),
        use_timestamps=bool(pipeline_cfg.get("use_timestamps", False)),
        debug_image_capture=bool(pipeline_cfg.get("debug_image_capture", True)),
        debug_image_whitelist=debug_whitelist,
        debug_subject_limit=pipeline_cfg.get("debug_subject_count", None),
        debug_subject_whitelist=pipeline_cfg.get("debug_subject_whitelist", None),
    )

    results_dir = results_root / "results" / str(dataset_cfg.exp_name)
    image_summary_csv = results_dir / "simple_dask_image_summary.csv"
    return {
        "backend": "dask",
        "results_dir": str(results_dir),
        "image_summary_csv": str(image_summary_csv) if image_summary_csv.exists() else "",
    }


def _run_spark_backend(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    input_csv: Path,
    results_root_override: Optional[Path],
    max_subjects_override: Optional[int],
    spark_partitions_override: Optional[int],
    debug_override: Optional[bool],
) -> Dict[str, str]:
    from pipeline_implementations.spark_implementation.pipeline import run_spark_pipeline

    dataset_cfg = cfg.get("dataset_config", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset_config must be a mapping.")
    pipeline_cfg = cfg.get("pipeline_config", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}
    spark_cfg = pipeline_cfg.get("spark", {})
    if not isinstance(spark_cfg, dict):
        spark_cfg = {}

    results_root = results_root_override or Path(str(pipeline_cfg.get("results_root", config_path.parent))).expanduser()
    if not results_root.is_absolute():
        results_root = (config_path.parent / results_root).resolve()

    scale_cfg = dataset_cfg.get("scale_lookup", {})
    if not isinstance(scale_cfg, dict):
        scale_cfg = {}
    default_mpp = scale_cfg.get("default_value", None)

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
        max_subjects=max_subjects_override if max_subjects_override is not None else pipeline_cfg.get("max_subjects", None),
        partitions=spark_partitions_override if spark_partitions_override is not None else spark_cfg.get("partitions", None),
        save_images=bool(pipeline_cfg.get("save_images", True)),
        save_effective_count_debug=bool(pipeline_cfg.get("save_effective_count_debug", False)),
        debug_image_capture=bool(pipeline_cfg.get("debug_image_capture", False)),
        debug_stage_whitelist=(
            list(pipeline_cfg.get("debug_stage", []))
            if pipeline_cfg.get("debug_stage", [])
            else None
        ),
        log_level=str(spark_cfg.get("log_level", "WARN")),
        debug=bool(pipeline_cfg.get("debug", False) if debug_override is None else debug_override),
    )
    summary["backend"] = "spark"
    return {key: str(value) for key, value in summary.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified lysozyme pipeline manager (discovery + analysis + stats).")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML.")
    parser.add_argument("--work-dir", type=Path, default=None, help=f"Directory containing {CONFIG_FILENAME}.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Override results root directory.")
    parser.add_argument("--backend", choices=("auto", "dask", "spark"), default="auto", help="Analysis backend.")
    parser.add_argument("--rewrite-csv", choices=("never", "always", "ask"), default="never", help="Discovery CSV rewrite policy.")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip image discovery/CSV generation.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip backend analysis.")
    parser.add_argument("--skip-stats", action="store_true", help="Skip stats analysis step.")
    parser.add_argument("--no-validate-existing-csv", action="store_true", help="Disable validation of an existing CSV.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Override configured max subjects.")
    parser.add_argument("--spark-partitions", type=int, default=None, help="Override spark partitions.")
    parser.add_argument("--debug", action="store_true", help="Force backend debug logging on.")
    parser.add_argument("--yes", action="store_true", help="Auto-continue after creating a new config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(config_arg=args.config, work_dir_arg=args.work_dir)
    cfg, created = _ensure_config(config_path, assume_yes=bool(args.yes))
    if created:
        print(f"Config initialized at: {config_path}")

    if args.results_dir is not None:
        cfg.setdefault("pipeline_config", {})
        if not isinstance(cfg["pipeline_config"], dict):
            cfg["pipeline_config"] = {}
        cfg["pipeline_config"]["results_root"] = str(args.results_dir.expanduser().resolve())

    csv_path = csv_path_from_config(cfg, config_path)
    if not args.skip_discovery:
        csv_path = _run_discovery_step(
            cfg=cfg,
            config_path=config_path,
            rewrite_policy=args.rewrite_csv,
            validate_csv=not bool(args.no_validate_existing_csv),
        )
    else:
        print("Discovery step skipped.")
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found while discovery is skipped: {csv_path}")

    backend = args.backend
    if backend == "auto":
        pipeline_cfg = cfg.get("pipeline_config", {})
        if not isinstance(pipeline_cfg, dict):
            pipeline_cfg = {}
        backend = str(pipeline_cfg.get("backend", "dask")).strip().lower()
        if backend not in {"dask", "spark"}:
            backend = "dask"

    analysis_summary: Dict[str, str] = {}
    if not args.skip_analysis:
        print(f"Running analysis backend: {backend}")
        if backend == "dask":
            analysis_summary = _run_dask_backend(
                cfg=cfg,
                config_path=config_path,
                input_csv=csv_path,
                results_root_override=args.results_dir.expanduser().resolve() if args.results_dir else None,
                max_subjects_override=args.max_subjects,
                debug_override=True if args.debug else None,
            )
        elif backend == "spark":
            analysis_summary = _run_spark_backend(
                cfg=cfg,
                config_path=config_path,
                input_csv=csv_path,
                results_root_override=args.results_dir.expanduser().resolve() if args.results_dir else None,
                max_subjects_override=args.max_subjects,
                spark_partitions_override=args.spark_partitions,
                debug_override=True if args.debug else None,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    else:
        print("Analysis step skipped.")

    if not args.skip_stats:
        image_summary_csv = analysis_summary.get("image_summary_csv", "")
        results_dir_text = analysis_summary.get("results_dir", "")
        if not image_summary_csv:
            pipeline_cfg = cfg.get("pipeline_config", {})
            if not isinstance(pipeline_cfg, dict):
                pipeline_cfg = {}
            dataset_cfg = cfg.get("dataset_config", {})
            if not isinstance(dataset_cfg, dict):
                dataset_cfg = {}
            inferred_results_root = (
                args.results_dir.expanduser().resolve()
                if args.results_dir
                else Path(str(pipeline_cfg.get("results_root", config_path.parent))).expanduser()
            )
            if not inferred_results_root.is_absolute():
                inferred_results_root = (config_path.parent / inferred_results_root).resolve()
            inferred_results_dir = inferred_results_root / "results" / str(dataset_cfg.get("exp_name", "lysozyme_run"))
            inferred_csv = inferred_results_dir / (
                "simple_dask_image_summary.csv" if backend == "dask" else "simple_spark_image_summary.csv"
            )
            if inferred_csv.exists():
                image_summary_csv = str(inferred_csv)
                results_dir_text = str(inferred_results_dir)
        if not image_summary_csv and results_dir_text:
            candidate_name = "simple_dask_image_summary.csv" if backend == "dask" else "simple_spark_image_summary.csv"
            candidate = Path(results_dir_text) / candidate_name
            if candidate.exists():
                image_summary_csv = str(candidate)
        if not image_summary_csv:
            print("Stats step skipped: no image summary CSV available.")
        else:
            stats_dir = Path(results_dir_text or Path(image_summary_csv).parent) / "stats"
            stats_outputs = run_basic_stats_analysis(
                image_summary_csv=Path(image_summary_csv),
                output_dir=stats_dir,
            )
            print("Stats analysis completed.")
            for key, value in stats_outputs.items():
                print(f"  {key}: {value}")
    else:
        print("Stats step skipped.")

    if analysis_summary:
        print("Analysis outputs:")
        for key, value in analysis_summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
