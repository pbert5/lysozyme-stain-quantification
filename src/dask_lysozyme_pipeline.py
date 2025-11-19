from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.lysozyme_pipelines import DatasetConfig, ScaleLookup, run_dask_pipeline
from src.lysozyme_pipelines.cli import build_debug_parser, compute_debug_whitelist
from src.lysozyme_stain_quantification.utils.debug_image_saver import (
    DEFAULT_DEBUG_STAGE_WHITELIST,
)

KAREN_DATASET = DatasetConfig(
    image_base_dir=Path("/home/phillip/documents/experimental_data/inputs/karen/lysozyme"),
    exp_name="rendering_test_run",
    blob_size_um=50.0 * 0.4476,
    max_regions_per_image=5,
    scoring_weights={
        "circularity": 0.15,
        "area": 0.25,
        "line_fit": 0.35,
        "red_intensity": 0.85,
        "com_consistency": 0.050,
    },
    scale_lookup=ScaleLookup(
        default_value=0.4476,
        keys=("40x",),
        values=(0.2253,),
    ),
    channel_keys=("_RFP", "_DAPI"),
)


def main() -> None:
    parser = build_debug_parser("Dask-based lysozyme crypt detection for Karen dataset.")
    args = parser.parse_args()
    whitelist = compute_debug_whitelist(args.debug_stage, base_whitelist=DEFAULT_DEBUG_STAGE_WHITELIST)

    run_dask_pipeline(
        dataset_cfg=KAREN_DATASET,
        results_root=REPO_ROOT,
        use_cluster=True,
        force_respawn_cluster=False,
        n_workers=None,
        threads_per_worker=None,
        save_images=True,
        debug=False,
        max_subjects=args.max_subjects,
        connect_to_existing_cluster=False,
        use_timestamps=False,
        debug_image_capture=args.capture_debug_images,
        debug_image_whitelist=whitelist,
        debug_subject_limit=args.debug_subject_count,
        debug_subject_whitelist=args.debug_subject if args.debug_subject is not None else None,
    )


if __name__ == "__main__":
    main()
