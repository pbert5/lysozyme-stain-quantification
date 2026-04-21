from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.io import imread

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEBASE_ROOT = REPO_ROOT / "codeBase"
CRYPT_CODE_ROOT = CODEBASE_ROOT / "crypt_detection_code"
for path in (CODEBASE_ROOT, CRYPT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lysozyme_stain_quantification.crypts.scoring_selector_mod import (  # type: ignore  # noqa: E402
    fit_centroid_curve_from_labels,
    sample_centroid_curve_points,
)
from lysozyme_stain_quantification.segment_crypts import segment_crypts_dual  # type: ignore  # noqa: E402
from lysozyme_stain_quantification.single_subject import (  # type: ignore  # noqa: E402
    _to_2d_channel,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "scratch_space" / "karends_keyance_data_analysis" / "lysozyme_pipeline_config.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "method_development" / "fix line fit" / "out"


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return data


def _discover_subject_dirs(root_dir: Path, lysozyme_name: str, tissue_name: str) -> list[Path]:
    subject_dirs: list[Path] = []
    for path in sorted(root_dir.rglob(lysozyme_name)):
        subject_dir = path.parent
        if (subject_dir / tissue_name).exists():
            subject_dirs.append(subject_dir)
    return subject_dirs


def _normalize_gray(image: np.ndarray, low_q: float = 1.0, high_q: float = 99.5) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, low_q))
    hi = float(np.percentile(arr, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")
    if arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected channels-last RGB/RGBA image, got shape {arr.shape}")
    rgb = arr[..., :3]
    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(3):
        out[..., idx] = _normalize_gray(rgb[..., idx])
    return out


def _find_overlay_path(subject_dir: Path) -> Path | None:
    overlays = sorted(subject_dir.glob("*Overlay.*"))
    return overlays[0] if overlays else None


def _curve_points(labels: np.ndarray, rfp: np.ndarray) -> np.ndarray:
    curve_model = fit_centroid_curve_from_labels(labels, rfp, max_degree=2)
    if curve_model is None:
        return np.empty((0, 2), dtype=np.float64)
    return sample_centroid_curve_points(
        curve_model,
        labels.shape,
        num_samples=max(128, int(np.hypot(labels.shape[0], labels.shape[1])) * 2),
    )


def _render_debug_figure(
    *,
    overlay_rgb: np.ndarray,
    rfp_norm: np.ndarray,
    curve_pts: np.ndarray,
    subject_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    panels = (
        (overlay_rgb, "Multichannel Overlay"),
        (rfp_norm, "RFP"),
    )

    for ax, (image, title) in zip(axes, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
        if curve_pts.shape[0] >= 2:
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], color="#21FFE1", linewidth=2.5)
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(f"Keyence line-fit debug: {subject_label}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the current Keyence line-fit on multichannel and RFP images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Keyence pipeline config yaml.",
    )
    parser.add_argument(
        "--subject-dir",
        type=Path,
        default=None,
        help="Specific subject directory. Defaults to the first discovered subject.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the rendered figure.",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    dataset_cfg = config.get("dataset_config", {})
    discovery = dataset_cfg.get("discovery", {})
    datasets = discovery.get("datasets", [])
    if not datasets:
        raise ValueError("No discovery datasets found in config.")

    keyence_cfg = datasets[0]
    root_dir = Path(keyence_cfg["root_dir"]).expanduser().resolve()
    lysozyme_name = str(keyence_cfg["channel_file_names"]["lysozyme"])
    tissue_name = str(keyence_cfg["channel_file_names"]["tissue"])
    subject_dir = args.subject_dir.expanduser().resolve() if args.subject_dir else None
    if subject_dir is None:
        subject_dirs = _discover_subject_dirs(root_dir, lysozyme_name, tissue_name)
        if not subject_dirs:
            raise FileNotFoundError(f"No subject dirs found under {root_dir}")
        subject_dir = subject_dirs[0]

    rfp_path = subject_dir / lysozyme_name
    dapi_path = subject_dir / tissue_name
    overlay_path = _find_overlay_path(subject_dir)
    if overlay_path is None:
        raise FileNotFoundError(f"No overlay image found in {subject_dir}")
    if not rfp_path.exists():
        raise FileNotFoundError(f"Missing RFP image: {rfp_path}")
    if not dapi_path.exists():
        raise FileNotFoundError(f"Missing DAPI image: {dapi_path}")

    rfp = _to_2d_channel(imread(str(rfp_path)), preferred_index=int(dataset_cfg.get("rfp_channel_index", 0)))
    dapi = _to_2d_channel(imread(str(dapi_path)), preferred_index=int(dataset_cfg.get("dapi_channel_index", 2)))
    if rfp.shape != dapi.shape:
        raise ValueError(f"RFP and DAPI shape mismatch: {rfp.shape} vs {dapi.shape}")

    _, best_labels = segment_crypts_dual(
        channels=(rfp, dapi),
        blob_size_um=float(dataset_cfg.get("blob_size_um", 22.38)),
        microns_per_px=float(keyence_cfg.get("microns_per_pixel", 0.4476)),
        debug=False,
        max_regions_best=int(dataset_cfg.get("max_regions_per_image", 5)),
        scoring_weights=dict(dataset_cfg.get("scoring_weights", {})),
    )

    curve_pts = _curve_points(best_labels, rfp)
    overlay_rgb = _normalize_rgb(imread(str(overlay_path)))
    rfp_norm = _normalize_gray(rfp)

    subject_label = str(subject_dir.relative_to(root_dir))
    output_path = args.output_dir / f"{subject_dir.parent.name}_{subject_dir.name}_linefit_debug.png"
    _render_debug_figure(
        overlay_rgb=overlay_rgb,
        rfp_norm=rfp_norm,
        curve_pts=curve_pts,
        subject_label=subject_label,
        output_path=output_path,
    )

    print(output_path)


if __name__ == "__main__":
    main()
