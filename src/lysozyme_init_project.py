from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import yaml

CONFIG_FILENAME = "lysozyme_pipeline_config.yaml"
INPUT_CSV_FILENAME = "lysozyme_input_data.csv"
DISCOVERY_SCRIPT_FILENAME = "discover_lysozyme_images.py"
RUN_HINT_FILENAME = "lysozyme_next_steps.txt"


def _default_config(work_dir: Path) -> Dict:
    return {
        "dataset_config": {
            "exp_name": "higher_quality_images_karen",
            "input_csv": INPUT_CSV_FILENAME,
            "blob_size_um": 22.38,
            "max_regions_per_image": 5,
            "rfp_gt_threshold": 71,
            "channel_keys": ["_CH2", "_CH4"],
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
                        "name": "keyence_new",
                        "mode": "structured_dir",
                        "recursive": True,
                        "root_dir": "/home/ash/documents/data/inputs/karen/lysozyme/new/Ileum Lysozyme - stt3 (Keyence)",
                        "subject_from": "two_level_dir",
                        "microns_per_pixel": 0.4476,
                        "channel_file_names": {
                            "lysozyme": "ileum_CH2.jpg",
                            "tissue": "ileum_CH4.jpg",
                        },
                    },
                    {
                        "name": "legacy_originals",
                        "mode": "token_match",
                        "recursive": True,
                        "root_dir": "/home/ash/documents/data/inputs/karen/lysozyme/originals",
                        "include_extensions": [".tif", ".tiff", ".jpg", ".jpeg"],
                        "exclude_name_tokens": ["overlay", "(red)"],
                        "allow_combined_single_file": True,
                        "microns_per_pixel": 0.4476,
                        "channel_tokens": {
                            "lysozyme": ["_RFP", " CH2", "_CH2"],
                            "tissue": ["_DAPI", " CH4", "_CH4"],
                        },
                    },
                ]
            },
        },
        "pipeline_config": {
            "results_root": str(work_dir),
            "use_cluster": True,
            "force_respawn_cluster": False,
            "connect_to_existing_cluster": False,
            "n_workers": None,
            "threads_per_worker": None,
            "save_images": True,
            "debug": False,
            "max_subjects": 100,
            "use_timestamps": False,
            "debug_image_capture": True,
            "debug_subject_count": 1,
            "debug_subject_whitelist": [],
            "debug_stage": [],
            "discovery_datasets_to_use": [],
        },
    }


def _write_config(config_path: Path, config: Dict, force: bool) -> None:
    if config_path.exists() and not force:
        return
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _write_template_csv(csv_path: Path, force: bool) -> None:
    if csv_path.exists() and not force:
        return

    fieldnames: List[str] = [
        "subject_id",
        "lysozyme_path",
        "tissue_path",
        "microns_per_pixel",
        "source_dataset",
        "source_label",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": "example_subject",
                "lysozyme_path": "/abs/path/to/lysozyme_or_rfp_image.tif",
                "tissue_path": "/abs/path/to/tissue_or_dapi_image.tif",
                "microns_per_pixel": "0.4476",
                "source_dataset": "keyence_new",
                "source_label": "template",
                "notes": "optional_metadata",
            }
        )


def _copy_discovery_script(target_path: Path, force: bool) -> None:
    source = Path(__file__).with_name("lysozyme_discovery_template.py")
    if target_path.exists() and not force:
        return
    shutil.copyfile(source, target_path)


def _write_run_hint(hint_path: Path, work_dir: Path) -> str:
    discover_cmd = (
        f"python3 {work_dir / DISCOVERY_SCRIPT_FILENAME} "
        f"--config {work_dir / CONFIG_FILENAME}"
    )
    lines = [
        "Suggested command:",
        discover_cmd,
        "",
        "Then run your pipeline entrypoint after reviewing CSV:",
        f"python3 /home/ash/documents/code/lysozyme/src/dask_lysozyme_pipeline.py",
    ]
    hint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return discover_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a staged lysozyme workspace (YAML config + input CSV + discovery script)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where config/csv/discovery script will be created (default: current working directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scaffold files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.output_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / CONFIG_FILENAME
    csv_path = work_dir / INPUT_CSV_FILENAME
    discovery_path = work_dir / DISCOVERY_SCRIPT_FILENAME
    hint_path = work_dir / RUN_HINT_FILENAME

    config = _default_config(work_dir)

    _write_config(config_path, config, force=args.force)
    _write_template_csv(csv_path, force=args.force)
    _copy_discovery_script(discovery_path, force=args.force)
    discover_cmd = _write_run_hint(hint_path, work_dir)

    print("Initialized lysozyme workspace scaffold:")
    print(f"  Config:          {config_path}")
    print(f"  Input CSV:       {csv_path}")
    print(f"  Discovery script:{discovery_path}")
    print(f"  Run hint:        {hint_path}")
    print("")
    print("Suggested run command:")
    print(f"  {discover_cmd}")


if __name__ == "__main__":
    main()
