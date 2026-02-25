#!/usr/bin/env python3
"""
Build poster-ready method assets for the lysozyme pipeline.

Run interface: yes.
Inputs:
  - Existing debug/render PNG assets
  - Existing project config for scoring weights
Outputs:
  - Curated assets under method_development/explaining/assets
  - Generated boards under method_development/explaining/generated
  - methods_text.md, figure_map.md, figure_text/*.txt, asset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from graphviz import Digraph
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage as ndi
from skimage.measure import regionprops


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLAINING_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXPLAINING_DIR / "assets"
GENERATED_DIR = EXPLAINING_DIR / "generated"
FIGURE_TEXT_DIR = EXPLAINING_DIR / "figure_text"
DEFAULT_POSTER_DIALS_PATH = EXPLAINING_DIR / "poster_dials.yaml"

PLANNED_ANIMATION_ROOT = Path(
    "/home/ash/documents/code/morphological_animation_toolkit/planned_animation"
)
LYSOZYME_ROOT = Path("/home/ash/documents/code/lysozyme")
ORIGINAL_OVERLAY_PATH = Path(
    "/home/ash/documents/data/inputs/karen/lysozyme/new/Ileum Lysozyme - stt3 (Keyence)/G2/G2EL/G2EL_ileum_Overlay.jpg"
)
N3_CARD_WIDTH_IN = 4.2
N3_CARD_HEIGHT_IN = 3.2
N3_CARD_IMAGE_HEIGHT_FRAC = 0.82
N3_CARD_IMAGE_ASPECT = N3_CARD_WIDTH_IN / (N3_CARD_HEIGHT_IN * N3_CARD_IMAGE_HEIGHT_FRAC)
POSTER_UNIVERSAL_SCORING_WEIGHTS = {
    "circularity": 0.20,
    "area": 0.20,
    "line_fit": 0.35,
    "red_intensity": 0.85,
}
N4_EXPONENTIAL_QUALITY_STRENGTH = 3.0
N4_ROW_CROP_WIDTH_MULTIPLIER = 2.0


@dataclass(frozen=True)
class CuratedAsset:
    panel_id: str
    filename: str
    source_path: Path
    use_case: str


@dataclass(frozen=True)
class SubjectConfig:
    key: str
    subject_label: str
    curated_assets: tuple[CuratedAsset, ...]
    paths_for_generation: dict[str, Path]
    scoring_config_path: Path


def _build_subject_configs() -> dict[str, SubjectConfig]:
    debug_root = (
        LYSOZYME_ROOT
        / "karens_data/results/higher_quality_images_karen/debug_intermediates/ileum_CH2_7e0c8b"
    )
    curated = (
        CuratedAsset(
            panel_id="C01",
            filename="C01_ileum_ch2_rfp_input.png",
            source_path=debug_root / "segment_crypts_dual/000_rfp_input.png",
            use_case="Raw/working RFP input channel.",
        ),
        CuratedAsset(
            panel_id="C02",
            filename="C02_ileum_ch2_dapi_input.png",
            source_path=debug_root / "segment_crypts_dual/001_dapi_input.png",
            use_case="Raw/working DAPI input channel.",
        ),
        CuratedAsset(
            panel_id="C03",
            filename="C03_ileum_ch2_crypt_preprocessed.png",
            source_path=debug_root / "identify_crypt_seeds_new/002_crypt_preprocessed.png",
            use_case="RFP channel after intensity standardization/preprocessing.",
        ),
        CuratedAsset(
            panel_id="C04",
            filename="C04_ileum_ch2_tissue_preprocessed.png",
            source_path=debug_root / "identify_crypt_seeds_new/003_tissue_preprocessed.png",
            use_case="DAPI channel after intensity standardization/preprocessing.",
        ),
        CuratedAsset(
            panel_id="C05",
            filename="C05_ileum_ch2_tissue_caps_troughs.png",
            source_path=debug_root / "identify_crypt_seeds_new/007_tissue_caps_troughs.png",
            use_case="DAPI-derived tissue border/cavity morphology map.",
        ),
        CuratedAsset(
            panel_id="C06",
            filename="C06_ileum_ch2_good_crypts.png",
            source_path=debug_root / "identify_crypt_seeds_new/020_good_crypts.png",
            use_case="RFP-derived high-likelihood crypt regions.",
        ),
        CuratedAsset(
            panel_id="C07",
            filename="C07_ileum_ch2_distance_image.png",
            source_path=debug_root / "identify_crypt_seeds_new/021_distance_image.png",
            use_case="Combined distance/likelihood map from DAPI and RFP morphology.",
        ),
        CuratedAsset(
            panel_id="C08",
            filename="C08_ileum_ch2_seed_labels.png",
            source_path=debug_root / "identify_crypt_seeds_new/025_seed_labels.png",
            use_case="Seed labels generated from morphology overlap.",
        ),
        CuratedAsset(
            panel_id="C09",
            filename="C09_ileum_ch2_base_labels.png",
            source_path=debug_root / "segment_crypts_dual/038_base_labels.png",
            use_case="Base labels after seeded growth.",
        ),
        CuratedAsset(
            panel_id="C10",
            filename="C10_ileum_ch2_final_crypt_labels.png",
            source_path=debug_root / "segment_crypts_dual/039_final_crypt_labels.png",
            use_case="Final selected crypt labels.",
        ),
        CuratedAsset(
            panel_id="C11",
            filename="C11_roi_mt_quality_fixed_hue.png",
            source_path=PLANNED_ANIMATION_ROOT / "resources/roi_mt_quality_fixed_hue.png",
            use_case="Quality map reference for scoring breakdown figure.",
        ),
    )

    paths_for_generation = {
        "rfp_input": curated[0].source_path,
        "dapi_input": curated[1].source_path,
        "crypt_preprocessed": curated[2].source_path,
        "tissue_preprocessed": curated[3].source_path,
        "tissue_caps_troughs": curated[4].source_path,
        "good_crypts": curated[5].source_path,
        "distance_image": curated[6].source_path,
        "seed_labels": curated[7].source_path,
        "base_labels": curated[8].source_path,
        "final_labels": curated[9].source_path,
        "quality_hue_reference": curated[10].source_path,
        "crypt_clean": debug_root / "identify_crypt_seeds_new/014_crypt_clean.png",
        "opened_split_times_thinned": debug_root / "identify_crypt_seeds_new/019_opened_split_times_thinned.png",
        "paired_overlay_original": ORIGINAL_OVERLAY_PATH,
    }

    cfg = SubjectConfig(
        key="ileum_ch2_debug_story",
        subject_label="ileum_CH2_7e0c8b",
        curated_assets=curated,
        paths_for_generation=paths_for_generation,
        scoring_config_path=LYSOZYME_ROOT / "karens_data/lysozyme_pipeline_config.yaml",
    )
    return {cfg.key: cfg}


SUBJECT_CONFIGS = _build_subject_configs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build poster-ready static method assets for lysozyme pipeline.",
    )
    parser.add_argument(
        "--subject-key",
        default="ileum_ch2_debug_story",
        choices=sorted(SUBJECT_CONFIGS.keys()),
        help="Subject configuration key to use for curation and generated boards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended actions without writing files.",
    )
    parser.add_argument(
        "--poster-dials-yaml",
        default=str(DEFAULT_POSTER_DIALS_PATH),
        help="YAML file that controls poster dials (weights, N3 crop geometry, figure text, N4 exp strength).",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[build_poster_assets] {message}")


def _as_float(value: Any, default: float, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    if min_value is not None and out < min_value:
        return float(default)
    return out


def _as_int(value: Any, default: int, *, min_value: int | None = None) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    if min_value is not None and out < min_value:
        return int(default)
    return out


def _dial(dials: dict[str, Any], path: str, default: Any) -> Any:
    node: Any = dials
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_poster_dials(dials_path: Path) -> dict[str, Any]:
    if not dials_path.exists():
        _log(f"Poster dials YAML not found at {dials_path}; using in-code defaults.")
        return {}
    with dials_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Poster dials YAML must be a mapping: {dials_path}")
    _log(f"Loaded poster dials from {dials_path}")
    return dict(loaded)


def _to_float_rgb(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    if np.nanmax(image) > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _load_rgb(path: Path) -> np.ndarray:
    return _to_float_rgb(plt.imread(path))


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gray = np.tensordot(rgb[..., :3], coeffs, axes=([-1], [0]))
    gray = gray.astype(np.float32)
    lo, hi = float(np.nanmin(gray)), float(np.nanmax(gray))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray - lo) / (hi - lo)


def _nonblack_mask(rgb: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb > threshold
    return np.max(rgb[..., :3], axis=-1) > threshold


def _merge_channels(dapi_gray: np.ndarray, rfp_gray: np.ndarray) -> np.ndarray:
    r = np.clip(rfp_gray, 0.0, 1.0)
    b = np.clip(dapi_gray, 0.0, 1.0)
    g = np.clip(0.20 * r + 0.20 * b, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _overlay(base_rgb: np.ndarray, overlay_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = base_rgb.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * overlay_rgb[mask]
    return np.clip(out, 0.0, 1.0)


def _save_graphviz_card(image_rgb: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(N3_CARD_WIDTH_IN, N3_CARD_HEIGHT_IN), dpi=240)
    gs = fig.add_gridspec(2, 1, height_ratios=(0.18, 0.82), hspace=0.0)
    ax_title = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[1, 0])

    ax_title.set_facecolor("#1f4368")
    ax_title.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        weight="bold",
    )
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    for spine in ax_title.spines.values():
        spine.set_visible(False)

    ax_img.imshow(image_rgb)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#1f4368")
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(out_path, dpi=240, facecolor="white")
    plt.close(fig)


def _normalize_for_display(rgb: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    arr = _to_float_rgb(rgb)
    scale = float(np.quantile(arr, quantile))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    scaled = np.clip(arr / scale, 0.0, 1.0)
    return np.power(scaled, 0.82)


def _labels_from_color_components(
    label_rgb: np.ndarray,
    threshold: float = 0.08,
) -> np.ndarray:
    rgb_float = _to_float_rgb(label_rgb)
    active_mask = np.max(rgb_float, axis=2) > threshold
    rgb_u8 = np.round(rgb_float * 255.0).astype(np.uint8)
    unique_colors = np.unique(rgb_u8.reshape(-1, 3), axis=0)

    label_img = np.zeros(rgb_u8.shape[:2], dtype=np.int32)
    next_id = 1
    for color in unique_colors:
        if np.all(color == 0):
            continue
        color_mask = np.all(rgb_u8 == color, axis=2) & active_mask
        if not np.any(color_mask):
            continue
        comp, count = ndi.label(color_mask)
        for comp_id in range(1, count + 1):
            label_img[comp == comp_id] = next_id
            next_id += 1
    return label_img


def _label_boundary_mask(label_img: np.ndarray) -> np.ndarray:
    padded = np.pad(label_img.astype(np.int32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    positive = center > 0
    edge_mask = positive & (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )
    return edge_mask


def _weighted_vertical_crop(
    image_rgb: np.ndarray,
    *,
    target_aspect: float,
    center_y: float,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h <= 1 or w <= 1 or not np.isfinite(target_aspect) or target_aspect <= 0.0:
        return image_rgb
    target_h = int(round(w / target_aspect))
    target_h = max(1, min(h, target_h))
    if target_h >= h:
        return image_rgb

    if not np.isfinite(center_y):
        center_y = float(h) / 2.0
    y0 = int(round(center_y - target_h / 2.0))
    y0 = max(0, min(h - target_h, y0))
    y1 = y0 + target_h
    return image_rgb[y0:y1, :]


def _horizontal_crop_within_bounds(
    image_rgb: np.ndarray,
    *,
    target_aspect: float,
    center_x: float,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h <= 1 or w <= 1 or not np.isfinite(target_aspect) or target_aspect <= 0.0:
        return image_rgb
    target_w = int(round(float(h) * float(target_aspect)))
    target_w = max(1, min(w, target_w))
    if target_w >= w:
        return image_rgb
    if not np.isfinite(center_x):
        center_x = float(w) / 2.0
    x0 = int(round(center_x - target_w / 2.0))
    x0 = max(0, min(w - target_w, x0))
    x1 = x0 + target_w
    return image_rgb[:, x0:x1]


def _compute_analysis_window_context(
    *,
    base_labels_rgb: np.ndarray,
    rfp_std_rgb: np.ndarray,
    scoring_weights: dict[str, float],
    n3_box_dials: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int]]:
    label_img = _labels_from_color_components(base_labels_rgb, threshold=0.08)
    scored_regions = _score_label_regions(
        label_img=label_img,
        intensity_gray=_grayscale(rfp_std_rgb),
        scoring_weights=scoring_weights,
    )
    crop_box = _crop_box_from_top_region(
        scored_regions,
        image_shape=label_img.shape,
        target_aspect=N3_CARD_IMAGE_ASPECT,
        n3_box_dials=n3_box_dials,
    )
    return label_img, scored_regions, crop_box


def _score_label_regions(
    label_img: np.ndarray,
    intensity_gray: np.ndarray,
    scoring_weights: dict[str, float],
) -> pd.DataFrame:
    regions = regionprops(label_img.astype(np.int32), intensity_image=intensity_gray.astype(np.float32))
    rows: list[dict[str, float]] = []
    for region in regions:
        area = float(region.area)
        perimeter = float(region.perimeter)
        circularity = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        if hasattr(region, "intensity_mean"):
            intensity_mean = float(region.intensity_mean)
        elif hasattr(region, "mean_intensity"):
            intensity_mean = float(region.mean_intensity)
        else:
            intensity_mean = 0.0
        if hasattr(region, "equivalent_diameter_area"):
            equivalent_diameter = float(region.equivalent_diameter_area)
        elif hasattr(region, "equivalent_diameter"):
            equivalent_diameter = float(region.equivalent_diameter)
        else:
            equivalent_diameter = float(np.sqrt(max(4.0 * area / np.pi, 0.0)))
        rows.append(
            {
                "label_id": float(region.label),
                "area": area,
                "com_row": float(region.centroid[0]),
                "com_col": float(region.centroid[1]),
                "total_red_intensity": float(intensity_mean * area),
                "red_intensity_per_area": intensity_mean,
                "circularity": circularity,
                "equivalent_diameter": equivalent_diameter,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if len(df) < 2:
        df["normalized_line_distance"] = 0.0
    else:
        x_coords = df["com_col"].to_numpy(dtype=np.float64)
        y_coords = df["com_row"].to_numpy(dtype=np.float64)
        weights = df["total_red_intensity"].to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.ones_like(weights)
            weight_sum = float(np.sum(weights))

        x_mean = float(np.sum(weights * x_coords) / weight_sum)
        y_mean = float(np.sum(weights * y_coords) / weight_sum)
        x_centered = x_coords - x_mean
        y_centered = y_coords - y_mean
        cov = np.array(
            [
                [float(np.sum(weights * x_centered * x_centered) / weight_sum), float(np.sum(weights * x_centered * y_centered) / weight_sum)],
                [float(np.sum(weights * x_centered * y_centered) / weight_sum), float(np.sum(weights * y_centered * y_centered) / weight_sum)],
            ],
            dtype=np.float64,
        )
        eigvals, eigvecs = np.linalg.eigh(cov)
        vx = float(eigvecs[0, int(np.argmax(eigvals))])
        vy = float(eigvecs[1, int(np.argmax(eigvals))])
        denom = float(np.hypot(vx, vy))
        if denom <= 1e-12:
            denom = 1.0
        distances = np.abs(vy * (x_coords - x_mean) - vx * (y_coords - y_mean)) / denom
        area_radius = np.sqrt(np.maximum(df["area"].to_numpy(dtype=np.float64), 0.0) / 2.0)
        area_radius = np.where(area_radius > 0.0, area_radius, 1.0)
        df["normalized_line_distance"] = distances / area_radius

    max_circularity = float(df["circularity"].max())
    max_area = float(df["area"].max())
    max_line_distance = float(df["normalized_line_distance"].max())
    max_red_intensity = float(df["red_intensity_per_area"].max())

    df["circularity_score"] = (
        1.0 - (df["circularity"] / max_circularity) if max_circularity > 0.0 else 1.0
    )
    df["area_score"] = 1.0 - (df["area"] / max_area) if max_area > 0.0 else 0.0
    df["line_fit_score"] = (
        df["normalized_line_distance"] / max_line_distance if max_line_distance > 0.0 else 0.0
    )
    df["red_intensity_score"] = (
        1.0 - (df["red_intensity_per_area"] / max_red_intensity) if max_red_intensity > 0.0 else 1.0
    )
    df["quality_score"] = (
        float(scoring_weights.get("circularity", 0.0)) * df["circularity_score"]
        + float(scoring_weights.get("area", 0.0)) * df["area_score"]
        + float(scoring_weights.get("line_fit", 0.0)) * df["line_fit_score"]
        + float(scoring_weights.get("red_intensity", 0.0)) * df["red_intensity_score"]
    )
    return df.sort_values("quality_score", ascending=True).reset_index(drop=True)


def _crop_box_from_top_region(
    scored_df: pd.DataFrame,
    image_shape: tuple[int, int],
    *,
    target_aspect: float = N3_CARD_IMAGE_ASPECT,
    n3_box_dials: dict[str, Any] | None = None,
) -> tuple[int, int, int, int]:
    n3_box_dials = n3_box_dials or {}
    center_to_edge_scale = _as_float(
        n3_box_dials.get("center_to_edge_equivalent_diameter_scale"),
        0.5,
        min_value=0.01,
    )
    vertical_up_multiplier = _as_float(
        n3_box_dials.get("vertical_up_multiplier"),
        3.0,
        min_value=0.0,
    )
    vertical_down_multiplier = _as_float(
        n3_box_dials.get("vertical_down_multiplier"),
        5.0,
        min_value=0.0,
    )
    crypt_length_min_px = _as_float(
        n3_box_dials.get("crypt_length_min_px"),
        20.0,
        min_value=1.0,
    )
    crypt_length_max_fraction = _as_float(
        n3_box_dials.get("crypt_length_max_fraction_of_min_image_dim"),
        1.0 / 3.0,
        min_value=0.01,
    )
    min_span = _as_int(
        n3_box_dials.get("min_span_px"),
        160,
        min_value=1,
    )

    height, width = int(image_shape[0]), int(image_shape[1])
    if scored_df.empty:
        return (0, height, 0, width)

    top = scored_df.iloc[0]
    center_y = float(top["com_row"])
    center_x = float(top["com_col"])
    equivalent_diameter = float(top.get("equivalent_diameter", np.nan))
    area = float(top.get("area", np.nan))
    fallback_diameter = np.sqrt(max(4.0 * area / np.pi, 1.0)) if np.isfinite(area) else 1.0
    if not np.isfinite(equivalent_diameter) or equivalent_diameter <= 0.0:
        equivalent_diameter = fallback_diameter

    # Treat "length" as center-to-edge distance so requested multipliers remain zoom-friendly.
    crypt_length = np.clip(
        center_to_edge_scale * equivalent_diameter,
        crypt_length_min_px,
        float(min(height, width)) * crypt_length_max_fraction,
    )

    y0 = int(np.floor(center_y - vertical_up_multiplier * crypt_length))
    y1 = int(np.ceil(center_y + vertical_down_multiplier * crypt_length))
    target_height = max(1, y1 - y0)
    target_width = int(round(float(target_aspect) * float(target_height)))
    half_width = max(1, int(round(target_width / 2.0)))
    x0 = int(np.floor(center_x)) - half_width
    x1 = x0 + target_width

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(height, y1)
    x1 = min(width, x1)

    if (y1 - y0) < min_span:
        pad = min_span - (y1 - y0)
        y0 = max(0, y0 - pad // 2)
        y1 = min(height, y1 + (pad - pad // 2))
    target_height = max(1, y1 - y0)
    target_width = min(width, max(min_span, int(round(float(target_aspect) * float(target_height)))))
    center_x_i = int(round(center_x))
    x0 = center_x_i - target_width // 2
    x1 = x0 + target_width
    if x0 < 0:
        x1 = min(width, x1 - x0)
        x0 = 0
    if x1 > width:
        shift = x1 - width
        x0 = max(0, x0 - shift)
        x1 = width

    if y1 <= y0 or x1 <= x0:
        return (0, height, 0, width)
    return (y0, y1, x0, x1)


def _crop_rgb(rgb: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = crop_box
    return rgb[y0:y1, x0:x1]


def _draw_crop_box(
    rgb: np.ndarray,
    crop_box: tuple[int, int, int, int],
    color: tuple[float, float, float] = (1.0, 0.92, 0.20),
    thickness: int = 5,
) -> np.ndarray:
    out = rgb.copy()
    y0, y1, x0, x1 = crop_box
    y0 = max(0, min(int(y0), out.shape[0] - 1))
    y1 = max(1, min(int(y1), out.shape[0]))
    x0 = max(0, min(int(x0), out.shape[1] - 1))
    x1 = max(1, min(int(x1), out.shape[1]))

    out[y0 : min(y0 + thickness, y1), x0:x1] = color
    out[max(y1 - thickness, y0) : y1, x0:x1] = color
    out[y0:y1, x0 : min(x0 + thickness, x1)] = color
    out[y0:y1, max(x1 - thickness, x0) : x1] = color
    return out


def _series_to_quality(values: pd.Series) -> dict[int, float]:
    arr = values.to_numpy(dtype=np.float64)
    if arr.size == 0:
        return {}
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        norm = np.full(arr.shape, 0.5, dtype=np.float64)
    elif hi <= lo:
        norm = np.full(arr.shape, 0.5, dtype=np.float64)
    else:
        norm = (arr - lo) / (hi - lo)
    quality = 1.0 - np.clip(norm, 0.0, 1.0)
    return {int(label): float(q) for label, q in zip(values.index.to_numpy(), quality)}


def _apply_exponential_quality_scale(
    label_to_quality: dict[int, float],
    strength: float = N4_EXPONENTIAL_QUALITY_STRENGTH,
) -> dict[int, float]:
    if not label_to_quality:
        return {}
    s = float(strength)
    if not np.isfinite(s) or s <= 0.0:
        return {int(label): float(np.clip(q, 0.0, 1.0)) for label, q in label_to_quality.items()}
    denom = float(np.expm1(s))
    if denom <= 1e-12:
        return {int(label): float(np.clip(q, 0.0, 1.0)) for label, q in label_to_quality.items()}
    out: dict[int, float] = {}
    for label, q in label_to_quality.items():
        q_clip = float(np.clip(q, 0.0, 1.0))
        out[int(label)] = float(np.expm1(s * q_clip) / denom)
    return out


def _render_quality_overlay(
    context_rgb: np.ndarray,
    label_img: np.ndarray,
    label_to_quality: dict[int, float],
    alpha: float = 0.72,
    mode: str = "saturation",
) -> np.ndarray:
    out = context_rgb.copy()
    for label_id, quality in label_to_quality.items():
        mask = label_img == int(label_id)
        if not np.any(mask):
            continue
        q = float(np.clip(quality, 0.0, 1.0))
        if mode == "hue":
            color = np.asarray(plt.get_cmap("turbo")(q)[:3], dtype=np.float32)
        else:
            # Linear saturation ramp (constant hue/value) for perceptually steady progression.
            hue = 0.02
            sat = 0.05 + 0.95 * q
            val = 1.0
            i = int(np.floor(hue * 6.0))
            f = hue * 6.0 - i
            p = val * (1.0 - sat)
            qh = val * (1.0 - f * sat)
            t = val * (1.0 - (1.0 - f) * sat)
            i = i % 6
            if i == 0:
                color = np.array([val, t, p], dtype=np.float32)
            elif i == 1:
                color = np.array([qh, val, p], dtype=np.float32)
            elif i == 2:
                color = np.array([p, val, t], dtype=np.float32)
            elif i == 3:
                color = np.array([p, qh, val], dtype=np.float32)
            elif i == 4:
                color = np.array([t, p, val], dtype=np.float32)
            else:
                color = np.array([val, p, qh], dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    edges = _label_boundary_mask(label_img)
    edges = ndi.binary_dilation(edges, iterations=1)
    out[edges] = np.clip(0.20 * out[edges] + 0.80 * np.array([1.0, 1.0, 1.0]), 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def _verify_exists(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required source files:\n{msg}")


def _ensure_dirs(dry_run: bool) -> None:
    for path in (ASSETS_DIR, GENERATED_DIR, FIGURE_TEXT_DIR):
        if dry_run:
            _log(f"DRY RUN: would create directory {path}")
            continue
        path.mkdir(parents=True, exist_ok=True)


def _copy_curated_assets(cfg: SubjectConfig, dry_run: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    _verify_exists(asset.source_path for asset in cfg.curated_assets)
    for asset in cfg.curated_assets:
        dest = ASSETS_DIR / asset.filename
        if dry_run:
            _log(f"DRY RUN: would copy {asset.source_path} -> {dest}")
        else:
            shutil.copy2(asset.source_path, dest)
        rows.append(
            {
                "panel_id": asset.panel_id,
                "filename": asset.filename,
                "source_path": str(asset.source_path),
                "use_case": asset.use_case,
                "status": "copied",
            }
        )
    return rows


def _load_scoring_weights(config_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    if not config_path.exists():
        return (
            {
                "circularity": 0.15,
                "area": 0.25,
                "line_fit": 0.35,
                "red_intensity": 0.85,
                "com_consistency": 0.05,
            },
            {
                "circularity": 0.35,
                "area": 0.15,
                "line_fit": 0.45,
                "red_intensity": 0.25,
            },
        )

    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    dataset_cfg = loaded.get("dataset_config", {})
    scoring = dict(dataset_cfg.get("scoring_weights", {}) or {})
    effective = dict(dataset_cfg.get("effective_count_scoring_weights", {}) or {})
    return scoring, effective


def _apply_poster_weight_overrides(
    scoring_weights: dict[str, float],
    effective_weights: dict[str, float],
    poster_weights: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    poster_scoring = dict(scoring_weights)
    poster_effective = dict(effective_weights)
    for metric, fallback in POSTER_UNIVERSAL_SCORING_WEIGHTS.items():
        value = _as_float(poster_weights.get(metric), fallback)
        poster_scoring[metric] = float(value)
        poster_effective[metric] = float(value)
    return poster_scoring, poster_effective


def _format_weight(weight: float | int | str | None) -> str:
    if weight is None:
        return "NA"
    try:
        return f"{float(weight):.2f}"
    except (TypeError, ValueError):
        return str(weight)


def _add_arrow_between_axes(
    fig: plt.Figure,
    ax_from: plt.Axes,
    ax_to: plt.Axes,
    *,
    start_y_frac: float = 0.5,
    end_y_frac: float = 0.5,
) -> None:
    from_box = ax_from.get_position()
    to_box = ax_to.get_position()
    arrow = FancyArrowPatch(
        (from_box.x1, from_box.y0 + from_box.height * start_y_frac),
        (to_box.x0, to_box.y0 + to_box.height * end_y_frac),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        lw=2.0,
        color="#123a66",
    )
    fig.add_artist(arrow)


def _generate_n1_pipeline_flowchart(
    output_path: Path,
    dry_run: bool,
    poster_dials: dict[str, Any],
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    fig, ax = plt.subplots(figsize=(18, 8.8), dpi=320)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    default_title = "Lysozyme Pipeline Overview (Poster Flow)"
    default_step_cards = [
        "Input field\n(DAPI + RFP)",
        "Split channels\nand standardize\nintensity",
        "Build morphology\nmaps for DAPI\nand RFP",
        "Combine channel\nevidence into\noverlap map",
        "Create seeds\nthen grow\nbase labels",
        "Apply weighted\nquality scoring\nand selection",
    ]
    default_step_label_prefix = "Step"
    default_footer = (
        "Start with paired channels, build morphology-informed overlap evidence, "
        "then score and keep the strongest crypt regions."
    )

    title = str(_dial(poster_dials, "text.n1.title", default_title))
    step_cards_cfg = _dial(poster_dials, "text.n1.step_cards", default_step_cards)
    if not isinstance(step_cards_cfg, list):
        step_cards_cfg = default_step_cards
    step_cards_txt = []
    for idx, fallback_txt in enumerate(default_step_cards):
        if idx < len(step_cards_cfg) and isinstance(step_cards_cfg[idx], str):
            step_cards_txt.append(step_cards_cfg[idx])
        else:
            step_cards_txt.append(fallback_txt)
    step_label_prefix = str(_dial(poster_dials, "text.n1.step_label_prefix", default_step_label_prefix))
    footer_text = str(_dial(poster_dials, "text.n1.footer_text", default_footer))

    ax.set_title(title, fontsize=28, weight="bold", pad=18)

    step_cards = [(idx + 1, txt) for idx, txt in enumerate(step_cards_txt)]
    step_positions = {
        1: (0.06, 0.59),
        2: (0.38, 0.59),
        3: (0.70, 0.59),
        4: (0.70, 0.20),
        5: (0.38, 0.20),
        6: (0.06, 0.20),
    }
    box_colors = ["#1d3557", "#27527e", "#2f6b96", "#3b84ab", "#4b9dc1", "#65b8d6"]
    box_width = 0.24
    box_height = 0.25

    for (idx, text), color in zip(step_cards, box_colors):
        x, y = step_positions[idx]
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            linewidth=2.3,
            edgecolor="#0f2138",
            facecolor=color,
            alpha=0.98,
        )
        ax.add_patch(box)
        ax.text(
            x + box_width / 2,
            y + box_height / 2,
            text,
            color="white",
            fontsize=14,
            ha="center",
            va="center",
            weight="bold",
        )
        ax.text(
            x + box_width / 2,
            y - 0.05,
            f"{step_label_prefix} {idx}",
            color="#16345b",
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
        )

    def _arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", lw=3.0, color="#10325b"),
        )

    step1 = step_positions[1]
    step2 = step_positions[2]
    step3 = step_positions[3]
    step4 = step_positions[4]
    step5 = step_positions[5]
    step6 = step_positions[6]

    _arrow(
        (step1[0] + box_width + 0.01, step1[1] + box_height * 0.50),
        (step2[0] - 0.01, step2[1] + box_height * 0.50),
    )
    _arrow(
        (step2[0] + box_width + 0.01, step2[1] + box_height * 0.50),
        (step3[0] - 0.01, step3[1] + box_height * 0.50),
    )
    _arrow(
        (step3[0] + box_width * 0.50, step3[1] - 0.01),
        (step4[0] + box_width * 0.50, step4[1] + box_height + 0.01),
    )
    _arrow(
        (step4[0] - 0.01, step4[1] + box_height * 0.50),
        (step5[0] + box_width + 0.01, step5[1] + box_height * 0.50),
    )
    _arrow(
        (step5[0] - 0.01, step5[1] + box_height * 0.50),
        (step6[0] + box_width + 0.01, step6[1] + box_height * 0.50),
    )

    ax.text(
        0.5,
        0.07,
        footer_text,
        ha="center",
        va="center",
        fontsize=14,
        color="#10263f",
        weight="bold",
    )

    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    _log(f"Generated {output_path}")


def _generate_n2_channel_split_standardization(
    cfg: SubjectConfig,
    output_path: Path,
    dry_run: bool,
    poster_dials: dict[str, Any],
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    original_overlay = _load_rgb(cfg.paths_for_generation["paired_overlay_original"])
    rfp_std = _load_rgb(cfg.paths_for_generation["crypt_preprocessed"])
    dapi_std = _load_rgb(cfg.paths_for_generation["tissue_preprocessed"])

    dapi_gray = _grayscale(dapi_std)
    rfp_gray = _grayscale(rfp_std)

    dapi_vis = np.stack(
        [0.12 * dapi_gray, 0.30 * dapi_gray, np.clip(1.05 * dapi_gray, 0.0, 1.0)], axis=-1
    )
    rfp_vis = np.stack(
        [np.clip(1.05 * rfp_gray, 0.0, 1.0), 0.20 * rfp_gray, 0.18 * rfp_gray], axis=-1
    )

    fig = plt.figure(figsize=(16, 9), dpi=320)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0), hspace=0.08, wspace=0.08)

    ax_input = fig.add_subplot(grid[:, 0])
    ax_dapi = fig.add_subplot(grid[0, 1])
    ax_rfp = fig.add_subplot(grid[1, 1])

    input_title = str(
        _dial(
            poster_dials,
            "text.n2.input_panel_title",
            "Original paired field (precombined overlay)",
        )
    )
    dapi_title = str(_dial(poster_dials, "text.n2.dapi_panel_title", "DAPI channel (standardized)"))
    rfp_title = str(
        _dial(
            poster_dials,
            "text.n2.rfp_panel_title",
            "RFP anti-LYZ channel (standardized)",
        )
    )
    figure_title = str(
        _dial(
            poster_dials,
            "text.n2.figure_title",
            "Channel Split and Intensity Standardization",
        )
    )
    footer_text = str(
        _dial(
            poster_dials,
            "text.n2.footer_text",
            (
                "The raw precombined field is split into channels, then each channel is normalized "
                "independently to align contrast for morphology filters."
            ),
        )
    )

    ax_input.imshow(original_overlay)
    ax_input.set_title(input_title, fontsize=16, weight="bold")
    ax_input.axis("off")

    ax_dapi.imshow(dapi_vis)
    ax_dapi.set_title(dapi_title, fontsize=14, weight="bold")
    ax_dapi.axis("off")

    ax_rfp.imshow(rfp_vis)
    ax_rfp.set_title(rfp_title, fontsize=14, weight="bold")
    ax_rfp.axis("off")

    _add_arrow_between_axes(fig, ax_input, ax_dapi, start_y_frac=0.66, end_y_frac=0.50)
    _add_arrow_between_axes(fig, ax_input, ax_rfp, start_y_frac=0.34, end_y_frac=0.50)

    fig.suptitle(figure_title, fontsize=21, weight="bold", y=0.98)

    fig.text(
        0.5,
        0.02,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#16345b",
    )

    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    _log(f"Generated {output_path}")


def _generate_n3_morphology_seed_flow(
    cfg: SubjectConfig,
    output_path: Path,
    scoring_weights: dict[str, float],
    dry_run: bool,
    poster_dials: dict[str, Any],
    n3_box_dials: dict[str, Any],
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    original_overlay = _load_rgb(cfg.paths_for_generation["paired_overlay_original"])
    dapi_std = _load_rgb(cfg.paths_for_generation["tissue_preprocessed"])
    rfp_std = _load_rgb(cfg.paths_for_generation["crypt_preprocessed"])
    distance_img = _load_rgb(cfg.paths_for_generation["distance_image"])
    tissue_caps = _load_rgb(cfg.paths_for_generation["tissue_caps_troughs"])
    crypt_clean = _load_rgb(cfg.paths_for_generation["crypt_clean"])
    opened_split_times_thinned = _load_rgb(cfg.paths_for_generation["opened_split_times_thinned"])
    seed_labels = _load_rgb(cfg.paths_for_generation["seed_labels"])
    base_labels = _load_rgb(cfg.paths_for_generation["base_labels"])
    rfp_gray = _grayscale(rfp_std)
    base_label_img, _scored_regions, crop_box = _compute_analysis_window_context(
        base_labels_rgb=base_labels,
        rfp_std_rgb=rfp_std,
        scoring_weights=scoring_weights,
        n3_box_dials=n3_box_dials,
    )

    original_display = _normalize_for_display(original_overlay)
    source_with_box = _draw_crop_box(original_display, crop_box)
    zoom_source = _crop_rgb(original_display, crop_box)

    dapi_gray = _grayscale(dapi_std)
    dapi_input_vis = np.stack(
        [0.12 * dapi_gray, 0.30 * dapi_gray, np.clip(1.05 * dapi_gray, 0.0, 1.0)],
        axis=-1,
    )
    rfp_input_vis = np.stack(
        [np.clip(1.05 * rfp_gray, 0.0, 1.0), 0.20 * rfp_gray, 0.18 * rfp_gray],
        axis=-1,
    )

    distance_gray_rgb = np.repeat(_grayscale(distance_img)[..., None], 3, axis=2)
    crypt_clean_gray = np.repeat(_grayscale(crypt_clean)[..., None], 3, axis=2)
    dapi_gray_rgb = np.repeat(dapi_gray[..., None], 3, axis=2)
    rfp_gray_rgb = np.repeat(rfp_gray[..., None], 3, axis=2)

    tissue_mask = _nonblack_mask(tissue_caps, threshold=0.08)
    opened_mask = _nonblack_mask(opened_split_times_thinned, threshold=0.08)
    seed_mask = _nonblack_mask(seed_labels, threshold=0.08)
    base_mask = base_label_img > 0
    base_boundary = _label_boundary_mask(base_label_img)

    dapi_morph = np.clip(0.60 * dapi_gray_rgb, 0.0, 1.0)
    dapi_morph[tissue_mask] = np.clip(dapi_morph[tissue_mask] + np.array([0.08, 0.28, 0.92]), 0.0, 1.0)

    opened_red = np.zeros_like(crypt_clean_gray)
    opened_red[..., 0] = 1.0
    rfp_morph = np.clip(0.60 * rfp_gray_rgb, 0.0, 1.0)
    rfp_morph = _overlay(rfp_morph, opened_red, opened_mask, alpha=0.96)

    overlap = dapi_morph.copy()
    opened_top_mask = ndi.binary_dilation(opened_mask, iterations=1)
    overlap = _overlay(overlap, opened_red, opened_top_mask, alpha=0.98)

    seeds_on_distance = _overlay(distance_gray_rgb, seed_labels, seed_mask, alpha=1.0)

    zoom_gray = _grayscale(zoom_source)
    zoom_gray_rgb = np.repeat(zoom_gray[..., None], 3, axis=2)
    base_crop_rgb = _crop_rgb(base_labels, crop_box)
    base_mask_crop = _crop_rgb(base_mask[..., None].astype(np.float32), crop_box)[..., 0] > 0.5
    base_boundary_crop = _crop_rgb(base_boundary[..., None].astype(np.float32), crop_box)[..., 0] > 0.5
    base_on_zoom = _overlay(zoom_gray_rgb, base_crop_rgb, base_mask_crop, alpha=0.52)
    base_on_zoom[base_boundary_crop] = 1.0

    default_node_titles = {
        "original_with_box": "Original field + zoom box",
        "zoom_source": "Zoomed analysis window",
        "dapi_input": "DAPI input (standardized)",
        "rfp_input": "RFP input (standardized)",
        "dapi_morph": "DAPI morphology",
        "rfp_morph": "RFP morphology",
        "overlap": "Channel overlap",
        "seed": "Seed labels on grayscale distance",
        "base": "Base labels on zoomed grayscale + boundaries",
    }
    node_titles_cfg = _dial(poster_dials, "text.n3.node_titles", {})
    if isinstance(node_titles_cfg, dict):
        for key in default_node_titles:
            if isinstance(node_titles_cfg.get(key), str):
                default_node_titles[key] = node_titles_cfg[key]

    node_images = {
        "original_with_box": (source_with_box, default_node_titles["original_with_box"]),
        "zoom_source": (zoom_source, default_node_titles["zoom_source"]),
        "dapi_input": (_crop_rgb(dapi_input_vis, crop_box), default_node_titles["dapi_input"]),
        "rfp_input": (_crop_rgb(rfp_input_vis, crop_box), default_node_titles["rfp_input"]),
        "dapi_morph": (_crop_rgb(dapi_morph, crop_box), default_node_titles["dapi_morph"]),
        "rfp_morph": (
            _crop_rgb(rfp_morph, crop_box),
            default_node_titles["rfp_morph"],
        ),
        "overlap": (_crop_rgb(overlap, crop_box), default_node_titles["overlap"]),
        "seed": (_crop_rgb(seeds_on_distance, crop_box), default_node_titles["seed"]),
        "base": (base_on_zoom, default_node_titles["base"]),
    }

    with tempfile.TemporaryDirectory(prefix="n3_graphviz_cards_") as temp_dir:
        temp_root = Path(temp_dir)
        node_card_paths: dict[str, Path] = {}
        for node_id, (img, title) in node_images.items():
            card_path = temp_root / f"{node_id}.png"
            _save_graphviz_card(img, title, card_path)
            node_card_paths[node_id] = card_path

        graph = Digraph(name="N3", engine="dot", format="png")
        graph_title = str(
            _dial(
                poster_dials,
                "text.n3.figure_title",
                "Zoomed Morphology-Guided Overlap and Seed Progression",
            )
        )
        graph.attr(
            rankdir="LR",
            splines="spline",
            nodesep="0.78",
            ranksep="1.00",
            bgcolor="white",
            pad="0.20",
            dpi="320",
            label=graph_title,
            labelloc="t",
            fontsize="27",
            fontname="Helvetica-Bold",
        )
        graph.attr(
            "node",
            shape="box",
            style="rounded",
            color="#1a3f65",
            penwidth="1.8",
            width="3.00",
            height="2.20",
            fixedsize="true",
            imagescale="both",
            label="",
        )
        graph.attr("edge", color="#123a66", penwidth="2.2", arrowsize="0.95")

        for node_id in node_images:
            graph.node(node_id, image=str(node_card_paths[node_id]))

        with graph.subgraph() as rank:
            rank.attr(rank="same")
            rank.node("original_with_box")
            rank.node("zoom_source")
        with graph.subgraph() as rank:
            rank.attr(rank="same")
            rank.node("dapi_input")
            rank.node("rfp_input")
        with graph.subgraph() as rank:
            rank.attr(rank="same")
            rank.node("dapi_morph")
            rank.node("rfp_morph")

        graph.edge("original_with_box", "zoom_source")
        graph.edge("zoom_source", "dapi_input")
        graph.edge("zoom_source", "rfp_input")
        graph.edge("dapi_input", "dapi_morph")
        graph.edge("rfp_input", "rfp_morph")
        graph.edge("dapi_morph", "overlap")
        graph.edge("rfp_morph", "overlap")
        graph.edge("overlap", "seed")
        graph.edge("seed", "base")

        output_path.write_bytes(graph.pipe(format="png"))

    _log(f"Generated {output_path}")


def _generate_n4_quality_scoring_breakdown(
    cfg: SubjectConfig,
    output_path: Path,
    scoring_weights: dict[str, float],
    effective_weights: dict[str, float],
    dry_run: bool,
    poster_dials: dict[str, Any],
    n3_box_dials: dict[str, Any],
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    source_overlay = _load_rgb(cfg.paths_for_generation["paired_overlay_original"])
    rfp_std = _load_rgb(cfg.paths_for_generation["crypt_preprocessed"])
    base_labels = _load_rgb(cfg.paths_for_generation["base_labels"])
    source_display = _normalize_for_display(source_overlay)
    exp_strength = _as_float(
        _dial(poster_dials, "n4.exp_strength", N4_EXPONENTIAL_QUALITY_STRENGTH),
        N4_EXPONENTIAL_QUALITY_STRENGTH,
        min_value=0.0,
    )
    row_crop_width_multiplier = _as_float(
        _dial(
            poster_dials,
            "n4.row_crop_width_multiplier",
            N4_ROW_CROP_WIDTH_MULTIPLIER,
        ),
        N4_ROW_CROP_WIDTH_MULTIPLIER,
        min_value=0.01,
    )
    n4_text_defaults = {
        "cumulative_linear_title": "Cumulative Quality Saturation Reference (Linear)",
        "cumulative_linear_note": "Baseline global reference.",
        "cumulative_exponential_title": "Cumulative Quality Saturation Reference (Exponential)",
        "cumulative_exponential_note": "Expanded top-end separation (exp strength={exp_strength:.1f}).",
        "table_title": "Weighted Quality Criteria",
        "header_labels": ["Metric", "Weight", "Quality Saturation Ref", "Interpretation"],
        "figure_title": "Scoring and Selection",
        "effective_prefix": "Effective-count weights:",
        "metric_descriptions": {
            "circularity": "Saturation ranks detections by circularity quality (higher saturation is better).",
            "area": "Saturation ranks detections by area quality (higher saturation is better).",
            "line_fit": "Saturation ranks detections by axis-alignment quality (higher saturation is better).",
            "red_intensity": "Saturation ranks detections by red-intensity quality (higher saturation is better).",
        },
    }
    n4_text_cfg = _dial(poster_dials, "text.n4", {})
    if not isinstance(n4_text_cfg, dict):
        n4_text_cfg = {}
    metric_descriptions_cfg = n4_text_cfg.get("metric_descriptions", {})
    metric_descriptions = dict(n4_text_defaults["metric_descriptions"])
    if isinstance(metric_descriptions_cfg, dict):
        for metric in metric_descriptions:
            if isinstance(metric_descriptions_cfg.get(metric), str):
                metric_descriptions[metric] = metric_descriptions_cfg[metric]
    header_labels_cfg = n4_text_cfg.get("header_labels", n4_text_defaults["header_labels"])
    if not isinstance(header_labels_cfg, list):
        header_labels_cfg = n4_text_defaults["header_labels"]
    header_labels = []
    for idx, fallback in enumerate(n4_text_defaults["header_labels"]):
        if idx < len(header_labels_cfg) and isinstance(header_labels_cfg[idx], str):
            header_labels.append(header_labels_cfg[idx])
        else:
            header_labels.append(fallback)
    cumulative_linear_title = str(n4_text_cfg.get("cumulative_linear_title", n4_text_defaults["cumulative_linear_title"]))
    cumulative_linear_note = str(n4_text_cfg.get("cumulative_linear_note", n4_text_defaults["cumulative_linear_note"]))
    cumulative_exponential_title = str(
        n4_text_cfg.get("cumulative_exponential_title", n4_text_defaults["cumulative_exponential_title"])
    )
    cumulative_exponential_note_template = str(
        n4_text_cfg.get("cumulative_exponential_note", n4_text_defaults["cumulative_exponential_note"])
    )
    table_title = str(n4_text_cfg.get("table_title", n4_text_defaults["table_title"]))
    figure_title = str(n4_text_cfg.get("figure_title", n4_text_defaults["figure_title"]))
    effective_prefix = str(n4_text_cfg.get("effective_prefix", n4_text_defaults["effective_prefix"]))
    try:
        cumulative_exponential_note = cumulative_exponential_note_template.format(exp_strength=exp_strength)
    except Exception:
        cumulative_exponential_note = cumulative_exponential_note_template

    label_img, scored_regions, crop_box = _compute_analysis_window_context(
        base_labels_rgb=base_labels,
        rfp_std_rgb=rfp_std,
        scoring_weights=scoring_weights,
        n3_box_dials=n3_box_dials,
    )
    y0_max, y1_max, x0_max, x1_max = crop_box

    metric_to_score_col = {
        "circularity": "circularity_score",
        "area": "area_score",
        "line_fit": "line_fit_score",
        "red_intensity": "red_intensity_score",
    }
    ordered_metrics = ["circularity", "area", "line_fit", "red_intensity"]
    rows: list[tuple[str, str, str, str]] = []
    for metric in ordered_metrics:
        if metric == "com_consistency":
            continue
        if metric in scoring_weights:
            score_col = metric_to_score_col.get(metric, "")
            rows.append(
                (
                    metric,
                    _format_weight(scoring_weights.get(metric)),
                    metric_descriptions.get(metric, "Saturation ranks detections by this metric."),
                    score_col,
                )
            )

    if not scored_regions.empty:
        scored_indexed = scored_regions.copy()
        scored_indexed["label_id"] = scored_indexed["label_id"].astype(int)
        scored_indexed = scored_indexed.set_index("label_id")
        cumulative_quality_linear = _series_to_quality(scored_indexed["quality_score"])
        cumulative_quality_exponential = _apply_exponential_quality_scale(
            cumulative_quality_linear,
            strength=exp_strength,
        )
        cumulative_overlay_linear = _render_quality_overlay(
            context_rgb=source_display,
            label_img=label_img,
            label_to_quality=cumulative_quality_linear,
            alpha=0.74,
        )
        cumulative_overlay_exponential = _render_quality_overlay(
            context_rgb=source_display,
            label_img=label_img,
            label_to_quality=cumulative_quality_exponential,
            alpha=0.74,
        )
    else:
        scored_indexed = pd.DataFrame()
        cumulative_overlay_linear = source_display
        cumulative_overlay_exponential = source_display

    metric_overlays: dict[str, np.ndarray] = {}
    for metric, _weight, _description, score_col in rows:
        if not scored_regions.empty and score_col in scored_indexed.columns:
            metric_quality = _series_to_quality(scored_indexed[score_col])
            metric_overlays[metric] = _render_quality_overlay(
                context_rgb=source_display,
                label_img=label_img,
                label_to_quality=metric_quality,
                alpha=0.74,
            )
        else:
            metric_overlays[metric] = source_display.copy()

    max_crop_overlays: dict[str, np.ndarray] = {
        metric: _crop_rgb(overlay, crop_box) for metric, overlay in metric_overlays.items()
    }

    fig = plt.figure(figsize=(22, 10.8), dpi=320)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.92, 1.50), hspace=0.08, wspace=0.06)
    ax_cumulative_linear = fig.add_subplot(gs[0, 0])
    ax_cumulative_exponential = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[:, 1])

    ax_cumulative_linear.imshow(cumulative_overlay_linear)
    ax_cumulative_linear.set_title(cumulative_linear_title, fontsize=15, weight="bold")
    ax_cumulative_linear.text(
        0.02,
        0.03,
        cumulative_linear_note,
        transform=ax_cumulative_linear.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=(0.02, 0.07, 0.15, 0.70), edgecolor="none"),
    )
    ax_cumulative_linear.axis("off")

    ax_cumulative_exponential.imshow(cumulative_overlay_exponential)
    ax_cumulative_exponential.set_title(
        cumulative_exponential_title,
        fontsize=15,
        weight="bold",
    )
    ax_cumulative_exponential.text(
        0.02,
        0.03,
        cumulative_exponential_note,
        transform=ax_cumulative_exponential.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=(0.02, 0.07, 0.15, 0.70), edgecolor="none"),
    )
    ax_cumulative_exponential.axis("off")

    ax_tbl.set_title(table_title, fontsize=18, weight="bold", pad=10)
    ax_tbl.set_xlim(0.0, 1.0)
    ax_tbl.set_ylim(0.0, 1.0)
    ax_tbl.axis("off")

    col_edges = [0.02, 0.17, 0.28, 0.66, 0.98]
    header_top = 0.94
    header_h = 0.10
    rows_count = max(len(rows), 1)
    row_h = min(0.16, (header_top - 0.20 - header_h) / rows_count)
    table_bottom = header_top - header_h - rows_count * row_h
    for idx, label in enumerate(header_labels):
        x0 = col_edges[idx]
        width = col_edges[idx + 1] - col_edges[idx]
        ax_tbl.add_patch(
            FancyBboxPatch(
                (x0, header_top - header_h),
                width,
                header_h,
                boxstyle="square,pad=0.0",
                facecolor="#1f4368",
                edgecolor="#16344f",
                linewidth=1.4,
            )
        )
        ax_tbl.text(
            x0 + width * 0.03,
            header_top - header_h / 2,
            label,
            color="white",
            fontsize=11,
            weight="bold",
            ha="left",
            va="center",
        )

    for idx, (metric, weight, interpretation, _score_col) in enumerate(rows):
        y1 = header_top - header_h - idx * row_h
        y0 = y1 - row_h
        row_color = "#f5f9ff" if idx % 2 == 0 else "white"
        ax_tbl.add_patch(
            FancyBboxPatch(
                (col_edges[0], y0),
                col_edges[-1] - col_edges[0],
                row_h,
                boxstyle="square,pad=0.0",
                facecolor=row_color,
                edgecolor="#c4d4e7",
                linewidth=1.0,
            )
        )
        ax_tbl.text(col_edges[0] + 0.012, y0 + row_h * 0.50, metric, ha="left", va="center", fontsize=10, color="#10263f")
        ax_tbl.text(col_edges[1] + 0.012, y0 + row_h * 0.50, weight, ha="left", va="center", fontsize=10, color="#10263f")

        hue_x0 = col_edges[2] + 0.020
        hue_x1 = col_edges[3] - 0.020
        hue_y0 = y0 + row_h * 0.12
        hue_y1 = y1 - row_h * 0.12
        target_aspect = (
            (hue_x1 - hue_x0) / max(hue_y1 - hue_y0, 1e-6)
        ) * float(row_crop_width_multiplier)

        row_overlay = max_crop_overlays.get(metric, _crop_rgb(source_display, crop_box))
        if not scored_indexed.empty and metric in metric_overlays:
            score_col = metric_to_score_col.get(metric)
            if score_col and score_col in scored_indexed.columns:
                metric_quality = _series_to_quality(scored_indexed[score_col])
                in_window = (
                    (scored_indexed["com_row"] >= y0_max)
                    & (scored_indexed["com_row"] < y1_max)
                    & (scored_indexed["com_col"] >= x0_max)
                    & (scored_indexed["com_col"] < x1_max)
                )
                subset = scored_indexed.loc[in_window]
                if not subset.empty:
                    weights = np.asarray(
                        [
                            float(metric_quality.get(int(label_id), 0.0))
                            for label_id in subset.index.to_numpy()
                        ],
                        dtype=np.float64,
                    )
                    coords = subset["com_col"].to_numpy(dtype=np.float64) - float(x0_max)
                    weight_sum = float(np.sum(weights))
                    if weight_sum > 1e-8:
                        center_x = float(np.sum(coords * weights) / weight_sum)
                    else:
                        center_x = row_overlay.shape[1] * 0.5
                else:
                    center_x = row_overlay.shape[1] * 0.5
            else:
                center_x = row_overlay.shape[1] * 0.5
        else:
            center_x = row_overlay.shape[1] * 0.5

        row_overlay = _horizontal_crop_within_bounds(
            row_overlay,
            target_aspect=target_aspect,
            center_x=center_x,
        )
        overlay_h = max(row_overlay.shape[0], 1)
        overlay_w = max(row_overlay.shape[1], 1)
        overlay_aspect = float(overlay_w) / float(overlay_h)
        cell_w = hue_x1 - hue_x0
        cell_h = hue_y1 - hue_y0
        draw_w = min(cell_w, cell_h * overlay_aspect)
        draw_h = draw_w / max(overlay_aspect, 1e-6)
        draw_x0 = hue_x0 + 0.5 * (cell_w - draw_w)
        draw_x1 = draw_x0 + draw_w
        draw_y0 = hue_y0 + 0.5 * (cell_h - draw_h)
        draw_y1 = draw_y0 + draw_h
        ax_tbl.imshow(
            row_overlay,
            extent=[draw_x0, draw_x1, draw_y0, draw_y1],
            aspect="auto",
            zorder=2,
        )
        ax_tbl.add_patch(
            FancyBboxPatch(
                (draw_x0, draw_y0),
                draw_x1 - draw_x0,
                draw_y1 - draw_y0,
                boxstyle="square,pad=0.0",
                facecolor="none",
                edgecolor="#1f4368",
                linewidth=0.9,
                zorder=3,
            )
        )

        ax_tbl.text(
            col_edges[3] + 0.010,
            y0 + row_h * 0.50,
            textwrap.fill(interpretation, width=40),
            ha="left",
            va="center",
            fontsize=9,
            color="#10263f",
        )

    for x in col_edges:
        ax_tbl.plot([x, x], [table_bottom, header_top], color="#b3c6db", lw=1.0)
    for idx in range(rows_count + 1):
        y = header_top - header_h - idx * row_h
        ax_tbl.plot([col_edges[0], col_edges[-1]], [y, y], color="#b3c6db", lw=1.0)

    formula_terms = [f"{_format_weight(scoring_weights.get(metric))}*{metric}_score" for metric, _, _, _ in rows]
    formula = "quality_score = " + " + ".join(formula_terms) if formula_terms else "quality_score = configured weighted sum"
    ax_tbl.text(
        0.02,
        table_bottom - 0.05,
        textwrap.fill(formula, width=95),
        ha="left",
        va="top",
        fontsize=10,
        color="#16345a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef5ff", edgecolor="#b7c8de"),
    )

    eff_text = f"{effective_prefix} " + ", ".join(
        f"{k}={_format_weight(v)}" for k, v in effective_weights.items()
    )
    ax_tbl.text(0.02, 0.05, eff_text, ha="left", va="bottom", fontsize=9, color="#27496e")

    fig.suptitle(figure_title, fontsize=22, weight="bold", y=0.98)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    _log(f"Generated {output_path}")


def _copy_generated_to_assets(
    generated_files: dict[str, Path],
    dry_run: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for panel_id, src in generated_files.items():
        dest = ASSETS_DIR / src.name
        if dry_run:
            _log(f"DRY RUN: would copy generated {src} -> {dest}")
        else:
            shutil.copy2(src, dest)
        rows.append(
            {
                "panel_id": panel_id,
                "filename": src.name,
                "source_path": str(src),
                "use_case": f"Generated poster board {panel_id}.",
                "status": "generated",
            }
        )
    return rows


def _write_methods_text(dry_run: bool) -> None:
    target = EXPLAINING_DIR / "methods_text.md"
    content = textwrap.dedent(
        """
        # Methods Text (Poster-Ready, Plain Language)

        ## One-line goal
        Detect lysozyme-producing intestinal crypt regions from paired DAPI and RFP images, then rank and select the most representative crypts per field.

        ## 1) Pipeline framing
        The pipeline starts from paired fluorescence channels, standardizes each channel independently, and then builds morphology-informed maps that encode where crypt-like structures are most likely to exist.

        ## 2) Morphology-informed likelihood
        In DAPI, we identify tissue borders and cavity-like structures; in RFP, we identify strong lysozyme-positive regions in expected size/shape ranges. These maps are combined into a distance/likelihood image where high values better match the target crypt profile.

        ## 3) Seed to region progression
        For figure communication, we select a zoom window around the highest-quality candidate and show the morphology flow locally: channel overlap, seed labels over grayscale distance image, and base labels over grayscale zoomed context with explicit label boundaries.

        ## 4) Weighted quality scoring
        Candidate regions are scored using circularity, area, line-fit alignment, and red-intensity features. Each metric row uses a tissue overlay saturation map for that single metric, while a separate cumulative saturation map shows the weighted total quality.
        """
    ).strip() + "\n"

    if dry_run:
        _log(f"DRY RUN: would write {target}")
        return
    target.write_text(content, encoding="utf-8")
    _log(f"Wrote {target}")


def _write_figure_map(dry_run: bool) -> None:
    target = EXPLAINING_DIR / "figure_map.md"
    content = textwrap.dedent(
        """
        # Figure Map (Updated N-series)

        ## Core source assets (C01-C11)
        - `C01` -> `assets/C01_ileum_ch2_rfp_input.png`: RFP channel input.
        - `C02` -> `assets/C02_ileum_ch2_dapi_input.png`: DAPI channel input.
        - `C03` -> `assets/C03_ileum_ch2_crypt_preprocessed.png`: RFP standardized/preprocessed.
        - `C04` -> `assets/C04_ileum_ch2_tissue_preprocessed.png`: DAPI standardized/preprocessed.
        - `C05` -> `assets/C05_ileum_ch2_tissue_caps_troughs.png`: DAPI morphology (caps/troughs).
        - `C06` -> `assets/C06_ileum_ch2_good_crypts.png`: RFP morphology-derived candidate regions.
        - `C07` -> `assets/C07_ileum_ch2_distance_image.png`: Combined likelihood/distance map.
        - `C08` -> `assets/C08_ileum_ch2_seed_labels.png`: Seed labels.
        - `C09` -> `assets/C09_ileum_ch2_base_labels.png`: Base labels after seeded growth.
        - `C10` -> `assets/C10_ileum_ch2_final_crypt_labels.png`: Final selected labels.
        - `C11` -> `assets/C11_roi_mt_quality_fixed_hue.png`: Quality hue reference.

        ## Generated poster boards (N1-N4)
        - `N1` -> `assets/N1_pipeline_flowchart.png`: High-level pipeline flowchart used to guide figure order.
        - `N2` -> `assets/N2_channel_split_standardization.png`: Original field split into standardized DAPI/RFP channels.
        - `N3` -> `assets/N3_morphology_seed_flowchart.png`: Morphology-based likelihood and seed-to-label flow.
        - `N4` -> `assets/N4_quality_scoring_breakdown.png`: Scoring criteria, weights, and quality-saturation interpretation.

        ## Figure text files
        - `figure_text/N1_pipeline_flowchart.txt`
        - `figure_text/N2_channel_split_standardization.txt`
        - `figure_text/N3_morphology_seed_flowchart.txt`
        - `figure_text/N4_quality_scoring_breakdown.txt`
        """
    ).strip() + "\n"

    if dry_run:
        _log(f"DRY RUN: would write {target}")
        return
    target.write_text(content, encoding="utf-8")
    _log(f"Wrote {target}")


def _write_figure_text_files(
    scoring_weights: dict[str, float],
    effective_weights: dict[str, float],
    dry_run: bool,
) -> list[Path]:
    files: dict[str, str] = {}

    files["N1_pipeline_flowchart.txt"] = textwrap.dedent(
        """
        Subtitle
        End-to-end pipeline overview from paired fluorescence channels to final crypt-level outputs.

        Large Text Box
        Start with paired DAPI and RFP images, split and standardize each channel, then build morphology-driven maps that emphasize crypt-like patterns. Combine channel evidence into a likelihood/distance representation, derive seed labels, expand to base and final crypt labels, and apply weighted quality scoring to keep the strongest crypt regions for downstream quantification.
        """
    ).strip() + "\n"

    files["N2_channel_split_standardization.txt"] = textwrap.dedent(
        """
        Subtitle
        What we start with: one field that branches into standardized DAPI and RFP channels.

        Large Text Box
        The input field is split into DAPI (tissue/cell context) and RFP anti-LYZ (lysozyme signal) channels. We standardize intensity per channel so both channels are on consistent contrast scales before morphology filtering. This makes downstream operations less sensitive to raw brightness variation across images.
        """
    ).strip() + "\n"

    files["N3_morphology_seed_flowchart.txt"] = textwrap.dedent(
        """
        Subtitle
        Zoomed morphology-guided overlap from standardized channels, then seed-to-region progression.

        Large Text Box
        We first mark a zoom window around the highest-quality detection candidate, then run the visual flow on that local window so crypt-scale morphology is easy to read.

        In RFP, crypt-like signal is modeled as local peaks that are relatively large, locally stable in intensity, and approximately round, with strong transitions near borders. In DAPI, we estimate tissue boundaries and cavity-like spaces where crypt lumens are expected to have low signal. Overlap between these maps highlights high-likelihood crypt centers.

        In this panel, seed labels are overlaid on a grayscale distance image, and base labels are overlaid on the grayscale zoomed analysis window with explicit label boundaries.
        """
    ).strip() + "\n"

    files["N4_quality_scoring_breakdown.txt"] = textwrap.dedent(
        f"""
        Subtitle
        Weighted scoring with per-metric tissue saturation references and paired cumulative maps (linear + exponential).

        Large Text Box
        Candidate regions are scored by shape and signal properties. For each metric row, the saturation reference shows the same tissue image with detections colored by that metric-specific quality. The global cumulative reference is shown twice: linear (baseline) and exponential (top-end quality separation).

        Row-level references use the same analysis-window bounds as the zoomed morphology panel: first we match the vertical extent to that window, then crop horizontally to fit the table cell proportion while staying inside the analysis-window limits. Labels come from the same base-label set.

        Selection weights (poster-standardized):
        - circularity: {_format_weight(scoring_weights.get('circularity'))}
        - area: {_format_weight(scoring_weights.get('area'))}
        - line_fit: {_format_weight(scoring_weights.get('line_fit'))}
        - red_intensity: {_format_weight(scoring_weights.get('red_intensity'))}

        Effective-count weights:
        - circularity: {_format_weight(effective_weights.get('circularity'))}
        - area: {_format_weight(effective_weights.get('area'))}
        - line_fit: {_format_weight(effective_weights.get('line_fit'))}
        - red_intensity: {_format_weight(effective_weights.get('red_intensity'))}
        """
    ).strip() + "\n"

    written: list[Path] = []
    for filename, content in files.items():
        target = FIGURE_TEXT_DIR / filename
        written.append(target)
        if dry_run:
            _log(f"DRY RUN: would write {target}")
            continue
        target.write_text(content, encoding="utf-8")
        _log(f"Wrote {target}")

    return written


def _write_manifest(
    curated_rows: list[dict[str, str]],
    generated_rows: list[dict[str, str]],
    figure_text_paths: list[Path],
    dry_run: bool,
) -> list[dict[str, str]]:
    target = EXPLAINING_DIR / "asset_manifest.csv"

    mapped_rows = [
        {
            "panel_id": "N1",
            "filename": "N1_pipeline_flowchart.png",
            "source_path": str(ASSETS_DIR / "N1_pipeline_flowchart.png"),
            "use_case": "Pipeline flow anchor figure.",
            "status": "mapped",
        },
        {
            "panel_id": "N2",
            "filename": "N2_channel_split_standardization.png",
            "source_path": str(ASSETS_DIR / "N2_channel_split_standardization.png"),
            "use_case": "Channel split and intensity standardization figure.",
            "status": "mapped",
        },
        {
            "panel_id": "N3",
            "filename": "N3_morphology_seed_flowchart.png",
            "source_path": str(ASSETS_DIR / "N3_morphology_seed_flowchart.png"),
            "use_case": "Morphology and seed-progression flowchart.",
            "status": "mapped",
        },
        {
            "panel_id": "N4",
            "filename": "N4_quality_scoring_breakdown.png",
            "source_path": str(ASSETS_DIR / "N4_quality_scoring_breakdown.png"),
            "use_case": "Quality scoring and weight breakdown figure.",
            "status": "mapped",
        },
    ]

    text_rows = []
    for path in figure_text_paths:
        panel_id = f"T{len(text_rows) + 1}"
        text_rows.append(
            {
                "panel_id": panel_id,
                "filename": str(path.relative_to(EXPLAINING_DIR)),
                "source_path": str(path),
                "use_case": "Figure copy file with Subtitle and Large Text Box sections.",
                "status": "mapped",
            }
        )

    all_rows = [*curated_rows, *generated_rows, *mapped_rows, *text_rows]
    fieldnames = ["panel_id", "filename", "source_path", "use_case", "status"]

    if dry_run:
        _log(f"DRY RUN: would write {target} with {len(all_rows)} rows")
        return all_rows

    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    _log(f"Wrote {target}")
    return all_rows


def _validate_references(dry_run: bool) -> None:
    if dry_run:
        _log("DRY RUN: skipping validation checks")
        return

    required_docs = [
        EXPLAINING_DIR / "methods_text.md",
        EXPLAINING_DIR / "figure_map.md",
        EXPLAINING_DIR / "asset_manifest.csv",
        FIGURE_TEXT_DIR / "N1_pipeline_flowchart.txt",
        FIGURE_TEXT_DIR / "N2_channel_split_standardization.txt",
        FIGURE_TEXT_DIR / "N3_morphology_seed_flowchart.txt",
        FIGURE_TEXT_DIR / "N4_quality_scoring_breakdown.txt",
    ]
    _verify_exists(required_docs)

    figure_map_text = (EXPLAINING_DIR / "figure_map.md").read_text(encoding="utf-8")
    rel_paths = sorted(
        set(re.findall(r"(?:assets|figure_text)/[A-Za-z0-9_.\-/]+(?:\.png|\.txt)", figure_map_text))
    )
    if not rel_paths:
        raise RuntimeError("No asset/text paths detected in figure_map.md.")

    for rel in rel_paths:
        abs_path = EXPLAINING_DIR / rel
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing figure_map reference: {abs_path}")

    manifest = pd.read_csv(EXPLAINING_DIR / "asset_manifest.csv")
    for source in manifest["source_path"].astype(str):
        for part in [p.strip() for p in source.split("|")]:
            if not part:
                continue
            if not Path(part).exists():
                raise FileNotFoundError(f"Manifest source path does not exist: {part}")

    _log(
        f"Validation complete: {len(rel_paths)} figure_map references verified and {len(manifest)} manifest rows checked."
    )


def build_bundle(cfg: SubjectConfig, dry_run: bool, poster_dials_yaml: Path) -> None:
    required = [cfg.scoring_config_path, *cfg.paths_for_generation.values()]
    _verify_exists(required)
    _ensure_dirs(dry_run=dry_run)
    poster_dials = _load_poster_dials(poster_dials_yaml)
    poster_weights = _dial(poster_dials, "weights.poster_universal", {})
    if not isinstance(poster_weights, dict):
        poster_weights = {}
    n3_box_dials = _dial(poster_dials, "n3_box_relative_dims", {})
    if not isinstance(n3_box_dials, dict):
        n3_box_dials = {}

    curated_rows = _copy_curated_assets(cfg=cfg, dry_run=dry_run)
    loaded_scoring_weights, loaded_effective_weights = _load_scoring_weights(cfg.scoring_config_path)
    scoring_weights, effective_weights = _apply_poster_weight_overrides(
        loaded_scoring_weights,
        loaded_effective_weights,
        poster_weights=poster_weights,
    )

    n1_path = GENERATED_DIR / "N1_pipeline_flowchart.png"
    n2_path = GENERATED_DIR / "N2_channel_split_standardization.png"
    n3_path = GENERATED_DIR / "N3_morphology_seed_flowchart.png"
    n4_path = GENERATED_DIR / "N4_quality_scoring_breakdown.png"

    _generate_n1_pipeline_flowchart(
        output_path=n1_path,
        dry_run=dry_run,
        poster_dials=poster_dials,
    )
    _generate_n2_channel_split_standardization(
        cfg=cfg,
        output_path=n2_path,
        dry_run=dry_run,
        poster_dials=poster_dials,
    )
    _generate_n3_morphology_seed_flow(
        cfg=cfg,
        output_path=n3_path,
        scoring_weights=scoring_weights,
        dry_run=dry_run,
        poster_dials=poster_dials,
        n3_box_dials=n3_box_dials,
    )
    _generate_n4_quality_scoring_breakdown(
        cfg=cfg,
        output_path=n4_path,
        scoring_weights=scoring_weights,
        effective_weights=effective_weights,
        dry_run=dry_run,
        poster_dials=poster_dials,
        n3_box_dials=n3_box_dials,
    )

    generated_rows = _copy_generated_to_assets(
        generated_files={"N1": n1_path, "N2": n2_path, "N3": n3_path, "N4": n4_path},
        dry_run=dry_run,
    )

    _write_methods_text(dry_run=dry_run)
    _write_figure_map(dry_run=dry_run)
    figure_text_paths = _write_figure_text_files(
        scoring_weights=scoring_weights,
        effective_weights=effective_weights,
        dry_run=dry_run,
    )

    _write_manifest(
        curated_rows=curated_rows,
        generated_rows=generated_rows,
        figure_text_paths=figure_text_paths,
        dry_run=dry_run,
    )

    _validate_references(dry_run=dry_run)

    if dry_run:
        _log("Dry run complete. No files were written.")
    else:
        _log(f"Poster asset bundle complete at: {EXPLAINING_DIR}")


def main() -> None:
    args = parse_args()
    cfg = SUBJECT_CONFIGS[args.subject_key]
    build_bundle(
        cfg=cfg,
        dry_run=bool(args.dry_run),
        poster_dials_yaml=Path(args.poster_dials_yaml),
    )


if __name__ == "__main__":
    main()
