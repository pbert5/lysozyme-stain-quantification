#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist
from skimage.io import imread
from skimage.transform import resize


REPO_ROOT = Path("/home/ash/documents/code/lysozyme")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codeBase.crypt_detection_code.lysozyme_stain_quantification.crypts.crypt_detection_solutions.crypt_identification_methodologies import (  # noqa: E402
    DEFAULT_MORPHOLOGY_PARAMS,
    preprocess_for_caps,
    to_float01,
)
from codeBase.crypt_detection_code.lysozyme_stain_quantification.single_subject import _to_2d_channel  # noqa: E402
from codeBase.crypt_detection_code.lysozyme_stain_quantification.utils.debug_image_saver import (  # noqa: E402
    _prepare_image_for_save,
)


DEFAULT_INPUT_CSV = REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis_ch3_ch4_combo" / "lysozyme_input_data.csv"
DEFAULT_OUT_PATH = (
    REPO_ROOT / "method_development" / "figure_out_merging_channels" / "merged_tissue_preproc_columns.png"
)


def _load_tissue_channels(row: pd.Series, dapi_channel_index: int) -> tuple[np.ndarray, np.ndarray]:
    ch3_path = Path(str(row["tissue_path"]))
    ch4_path = Path(str(row["tissue_aux_path"]))
    ch3 = _to_2d_channel(imread(str(ch3_path)), preferred_index=dapi_channel_index)
    ch4 = _to_2d_channel(imread(str(ch4_path)), preferred_index=dapi_channel_index)
    if ch3.shape != ch4.shape:
        raise ValueError(f"Shape mismatch for {ch3_path} and {ch4_path}: {ch3.shape} vs {ch4.shape}")
    return ch3, ch4


def _tissue_preprocessed(image: np.ndarray) -> np.ndarray:
    params = DEFAULT_MORPHOLOGY_PARAMS
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        # CLAHE rejects float images outside [-1, 1]. Preserve the original
        # uint8 behavior when possible, and normalize only for float inputs.
        arr = to_float01(arr)
    return preprocess_for_caps(equalize_adapthist(arr), params.salt_and_pepper_noise_size)


def _display_ready(image: np.ndarray, stage: str) -> np.ndarray:
    prepared = _prepare_image_for_save(image, stage=stage)
    if prepared is None:
        raise ValueError(f"Could not prepare image for display at stage '{stage}'.")
    return prepared


def _downsample_for_display(image: np.ndarray, max_width: int = 480) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        return arr
    height, width = arr.shape
    if width <= max_width:
        return arr
    scale = max_width / float(width)
    new_shape = (max(1, int(round(height * scale))), max_width)
    return resize(arr, new_shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a per-subject tissue comparison grid with columns for CH4 original, "
            "CH4 preprocessed, CH3 original, and CH3 preprocessed."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dapi-channel-index", type=int, default=2)
    parser.add_argument("--row-height", type=float, default=1.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"subject_id", "tissue_path", "tissue_aux_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df = df[df["tissue_aux_path"].notna() & df["tissue_aux_path"].astype(str).str.strip().ne("")]
    df = df.sort_values("subject_id").reset_index(drop=True)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()
    if df.empty:
        raise ValueError(f"No rows with auxiliary tissue channels found in {args.input_csv}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(df),
        4,
        figsize=(12, max(3, len(df) * float(args.row_height))),
        dpi=90,
    )
    axes_arr = np.atleast_2d(axes)
    col_titles = [
        "CH4 Original",
        "CH4 Preproc",
        "CH3 Original",
        "CH3 Preproc",
    ]

    for row_idx, (_, row) in enumerate(df.iterrows()):
        ch3, ch4 = _load_tissue_channels(row, dapi_channel_index=int(args.dapi_channel_index))
        ch4_preproc = _tissue_preprocessed(ch4)
        ch3_preproc = _tissue_preprocessed(ch3)

        panels = [
            _downsample_for_display(_display_ready(ch4, stage="dapi_input")),
            _downsample_for_display(_display_ready(ch4_preproc, stage="tissue_preprocessed")),
            _downsample_for_display(_display_ready(ch3, stage="dapi_input")),
            _downsample_for_display(_display_ready(ch3_preproc, stage="tissue_preprocessed")),
        ]

        for col_idx, panel in enumerate(panels):
            ax = axes_arr[row_idx, col_idx]
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(str(row["subject_id"]), fontsize=7, rotation=0, ha="right", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Tissue Preprocessing Comparison (CH4 vs CH3)", fontsize=14)
    fig.tight_layout(rect=(0.08, 0, 1, 0.99))
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
