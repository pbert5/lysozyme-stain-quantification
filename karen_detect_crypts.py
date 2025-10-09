"""
Primary script to run crypt segmentation on lysozyme stain images and save plots.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from image_ops_framework.analysis_stack_xr import AnalysisStackXR
from src.scientific_image_finder.finder import find_subject_image_sets
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts


DEBUG = False


def setup_results_dir(script_dir: Path) -> Path:
    """Create the results directory for this script."""
    results_dir = script_dir / "results" / "karen"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_all_crypts(out, *, lab_name="crypts", ncols=5, figsize_per=(3, 3), subjects=None):
    """
    Plot label data for every subject in a Matplotlib grid.

    Supports:
      - xarray: expects out.label with dims including "lab","subject","y","x".
      - npz_dict (dict): expects key "label:{lab_name}" -> array shaped (subject, y, x).
    Optionally accepts an explicit `subjects` list for dict inputs.
    """
    if isinstance(out, dict):
        key = f"label:{lab_name}"
        if key not in out:
            raise KeyError(f"{key} not found in out dict. Available keys: {list(out.keys())}")
        arr = out[key]
        if subjects is None or len(subjects) != arr.shape[0]:
            subjects = list(range(arr.shape[0]))
        da_like = arr
    elif isinstance(out, xr.Dataset) or isinstance(out, xr.DataArray):
        if isinstance(out, xr.Dataset):
            if "label" not in out:
                raise KeyError("xarray Dataset has no 'label' variable")
            da = out["label"]
        else:
            da = out
        da = da.sel(lab=lab_name)
        subjects = list(da.coords["subject"].values)
        da_like = da.compute() if hasattr(da.data, "compute") else da
    else:
        raise TypeError("out must be an xarray object or an npz_dict (dict)")

    n = len(subjects)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per[0], nrows * figsize_per[1]))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for i, subj in enumerate(subjects):
        ax = axes[i // ncols, i % ncols]
        if isinstance(da_like, np.ndarray):
            img = da_like[i]
        else:
            img = da_like.sel(subject=subj).data
            if hasattr(img, "compute"):
                img = img.compute()
        ax.imshow(img, interpolation="nearest", cmap="viridis")
        ax.set_title(f"subject={subj}", fontsize=9)
        ax.axis("off")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(f'"{lab_name}" labels across subjects', y=0.98)
    fig.tight_layout()
    return fig


def main() -> None:
    results_dir = setup_results_dir(SCRIPT_DIR)

    img_dir = Path("/home/phillip/documents/lysozyme/lysozyme images")
    lysozyme_channel = "rfp"
    dapi_channel = "dapi"

    subject_names, images_by_source, source_names = find_subject_image_sets(
        img_dir=img_dir,
        sources=[("rfp", lysozyme_channel, "r"), ("dapi", dapi_channel, "b")],
        max_subjects=1000,
    )

    if DEBUG:
        print(f"Sources: {source_names}")
        print(f"Found {len(subject_names)} subjects with images in both channels.")
        for subject, red_img, blue_img in zip(subject_names, images_by_source[0], images_by_source[1]):
            print(f"Subject: {subject}, Red shape: {red_img.shape}, Blue shape: {blue_img.shape}")

    stk = (
        AnalysisStackXR()
        .add_sources(subject=subject_names, sources=images_by_source, sourcenames=source_names)
        .run(segment_crypts, channels=["rfp", "dapi"], output_name="crypts", use_dask=True, blob_size_px=40)
    )

    print(stk)
    ds = stk.output(format="dataset")

    fig = plot_all_crypts(ds, lab_name="crypts", ncols=6, figsize_per=(3, 3))

    output_path = results_dir / "karen_detect_crypts.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if DEBUG:
        print(f"Saved visualization to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
