import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import math
from pathlib import Path
def setup_results_dir(script_dir: Path, exp_name: str) -> Path:
    """Create the results directory for this script."""
    results_dir = script_dir / "results" / exp_name
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