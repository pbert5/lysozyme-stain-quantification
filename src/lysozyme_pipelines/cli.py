from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_debug_parser(
    description: str,
    *,
    default_max_subjects: int = 20,
    default_debug_subject_limit: int = 1,
) -> argparse.ArgumentParser:
    """Create a CLI parser with common debug/capture options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--capture-debug-images",
        dest="capture_debug_images",
        action="store_true",
        help="Persist whitelisted intermediate images for animation/debugging.",
    )
    parser.add_argument(
        "--no-capture-debug-images",
        dest="capture_debug_images",
        action="store_false",
        help="Disable intermediate image capture even if globally enabled.",
    )
    parser.add_argument(
        "--debug-stage",
        dest="debug_stage",
        action="append",
        default=None,
        help="Add a stage to the debug whitelist (can be provided multiple times).",
    )
    parser.add_argument(
        "--debug-subject-count",
        dest="debug_subject_count",
        type=int,
        default=default_debug_subject_limit,
        help=f"Capture debug intermediates for the first N subjects (default: {default_debug_subject_limit}).",
    )
    parser.add_argument(
        "--max-subjects",
        dest="max_subjects",
        type=int,
        default=default_max_subjects,
        help=f"Limit the number of subjects processed (default: {default_max_subjects}).",
    )
    parser.add_argument(
        "--debug-subject",
        dest="debug_subject",
        action="append",
        default=None,
        help="Limit intermediate image capture to specific subject names (repeat for multiple).",
    )
    parser.set_defaults(capture_debug_images=True)
    return parser


def compute_debug_whitelist(
    stages: Optional[Sequence[str]],
    *,
    base_whitelist: Optional[Sequence[str]] = None,
) -> Optional[Sequence[str]]:
    """Merge CLI-provided debug stages with a default whitelist."""
    if not stages:
        return None
    whitelist = list(base_whitelist or [])
    whitelist.extend(stages)
    return whitelist
