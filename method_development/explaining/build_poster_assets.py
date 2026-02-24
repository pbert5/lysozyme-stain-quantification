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
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from graphviz import Digraph
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage as ndi


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLAINING_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXPLAINING_DIR / "assets"
GENERATED_DIR = EXPLAINING_DIR / "generated"
FIGURE_TEXT_DIR = EXPLAINING_DIR / "figure_text"

PLANNED_ANIMATION_ROOT = Path(
    "/home/ash/documents/code/morphological_animation_toolkit/planned_animation"
)
LYSOZYME_ROOT = Path("/home/ash/documents/code/lysozyme")
ORIGINAL_OVERLAY_PATH = Path(
    "/home/ash/documents/data/inputs/karen/lysozyme/new/Ileum Lysozyme - stt3 (Keyence)/G2/G2EL/G2EL_ileum_Overlay.jpg"
)


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
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[build_poster_assets] {message}")


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


def _fit_primary_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points: np.ndarray | None = None
    labels, count = ndi.label(mask)
    if count >= 2:
        component_ids = np.arange(1, count + 1)
        component_sizes = ndi.sum(mask.astype(np.float32), labels, index=component_ids)
        keep_ids = component_ids[np.asarray(component_sizes) >= 8]
        if keep_ids.size >= 2:
            centers = np.asarray(
                ndi.center_of_mass(mask.astype(np.float32), labels, index=keep_ids),
                dtype=np.float64,
            )
            centers = centers[~np.isnan(centers).any(axis=1)]
            if centers.shape[0] >= 2:
                points = np.stack([centers[:, 1], centers[:, 0]], axis=1)

    if points is None:
        ys, xs = np.where(mask)
        if xs.size < 2:
            return None
        points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return center, (direction / norm)


def _line_segment_in_image(
    width: int,
    height: int,
    center_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx, dy = float(direction_xy[0]), float(direction_xy[1])

    intersections: list[tuple[float, float, float]] = []
    eps = 1e-12
    if abs(dx) > eps:
        for x in (0.0, float(width - 1)):
            t = (x - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= float(height - 1):
                intersections.append((t, x, y))
    if abs(dy) > eps:
        for y in (0.0, float(height - 1)):
            t = (y - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= float(width - 1):
                intersections.append((t, x, y))

    if len(intersections) < 2:
        return None

    intersections.sort(key=lambda item: item[0])
    _, x0, y0 = intersections[0]
    _, x1, y1 = intersections[-1]
    return (x0, y0), (x1, y1)


def _save_graphviz_card(image_rgb: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(4.2, 3.2), dpi=240)
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


def _generate_n1_pipeline_flowchart(output_path: Path, dry_run: bool) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    fig, ax = plt.subplots(figsize=(18, 8.8), dpi=320)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "Lysozyme Pipeline Overview (Poster Flow)",
        fontsize=28,
        weight="bold",
        pad=18,
    )

    step_cards = [
        (1, "Input field\n(DAPI + RFP)"),
        (2, "Split channels\nand standardize\nintensity"),
        (3, "Build morphology\nmaps for DAPI\nand RFP"),
        (4, "Combine channel\nevidence into\noverlap map"),
        (5, "Create seeds\nthen grow\nbase labels"),
        (6, "Apply weighted\nquality scoring\nand selection"),
    ]
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
            f"Step {idx}",
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
        "Start with paired channels, build morphology-informed overlap evidence, then score and keep the strongest crypt regions.",
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

    ax_input.imshow(original_overlay)
    ax_input.set_title("Original paired field (precombined overlay)", fontsize=16, weight="bold")
    ax_input.axis("off")

    ax_dapi.imshow(dapi_vis)
    ax_dapi.set_title("DAPI channel (standardized)", fontsize=14, weight="bold")
    ax_dapi.axis("off")

    ax_rfp.imshow(rfp_vis)
    ax_rfp.set_title("RFP anti-LYZ channel (standardized)", fontsize=14, weight="bold")
    ax_rfp.axis("off")

    _add_arrow_between_axes(fig, ax_input, ax_dapi, start_y_frac=0.66, end_y_frac=0.50)
    _add_arrow_between_axes(fig, ax_input, ax_rfp, start_y_frac=0.34, end_y_frac=0.50)

    fig.suptitle(
        "Channel Split and Intensity Standardization",
        fontsize=21,
        weight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        "The raw precombined field is split into channels, then each channel is normalized independently to align contrast for morphology filters.",
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
    dry_run: bool,
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    dapi_input = _load_rgb(cfg.paths_for_generation["dapi_input"])
    rfp_input = _load_rgb(cfg.paths_for_generation["rfp_input"])
    tissue_caps = _load_rgb(cfg.paths_for_generation["tissue_caps_troughs"])
    crypt_clean = _load_rgb(cfg.paths_for_generation["crypt_clean"])
    opened_split_times_thinned = _load_rgb(cfg.paths_for_generation["opened_split_times_thinned"])
    seed_labels = _load_rgb(cfg.paths_for_generation["seed_labels"])
    base_labels = _load_rgb(cfg.paths_for_generation["base_labels"])
    crypt_clean_gray = np.repeat(_grayscale(crypt_clean)[..., None], 3, axis=2)

    tissue_mask = _nonblack_mask(tissue_caps, threshold=0.08)
    opened_mask = _nonblack_mask(opened_split_times_thinned, threshold=0.08)
    seed_mask = _nonblack_mask(seed_labels, threshold=0.08)
    base_mask = _nonblack_mask(base_labels, threshold=0.08)

    dapi_morph = crypt_clean_gray * 0.30
    dapi_morph[tissue_mask] = np.clip(dapi_morph[tissue_mask] + np.array([0.08, 0.28, 0.92]), 0.0, 1.0)

    opened_red = np.zeros_like(crypt_clean_gray)
    opened_red[..., 0] = 1.0
    rfp_morph = _overlay(crypt_clean_gray, opened_red, opened_mask, alpha=0.90)

    overlap = crypt_clean_gray * 0.30
    overlap[tissue_mask] = np.clip(overlap[tissue_mask] + np.array([0.08, 0.28, 0.92]), 0.0, 1.0)
    overlap[opened_mask] = np.clip(overlap[opened_mask] + np.array([0.92, 0.06, 0.06]), 0.0, 1.0)

    seeds_on_gray = _overlay(crypt_clean_gray, seed_labels, seed_mask, alpha=0.92)
    base_on_gray = _overlay(crypt_clean_gray, base_labels, base_mask, alpha=0.55)

    node_images = {
        "dapi_input": (dapi_input, "DAPI input"),
        "rfp_input": (rfp_input, "RFP input"),
        "dapi_morph": (dapi_morph, "DAPI morphology"),
        "rfp_morph": (rfp_morph, "Crypt clean + opened split times thinned"),
        "overlap": (overlap, "Channel overlap"),
        "seed": (seeds_on_gray, "Seed labels on grayscale"),
        "base": (base_on_gray, "Base labels on grayscale"),
    }

    with tempfile.TemporaryDirectory(prefix="n3_graphviz_cards_") as temp_dir:
        temp_root = Path(temp_dir)
        node_card_paths: dict[str, Path] = {}
        for node_id, (img, title) in node_images.items():
            card_path = temp_root / f"{node_id}.png"
            _save_graphviz_card(img, title, card_path)
            node_card_paths[node_id] = card_path

        graph = Digraph(name="N3", engine="dot", format="png")
        graph.attr(
            rankdir="LR",
            splines="spline",
            nodesep="0.72",
            ranksep="1.05",
            bgcolor="white",
            pad="0.20",
            dpi="320",
            label="Morphology-Guided Overlap and Seed Progression",
            labelloc="t",
            fontsize="28",
            fontname="Helvetica-Bold",
        )
        graph.attr(
            "node",
            shape="box",
            style="rounded",
            color="#1a3f65",
            penwidth="1.8",
            width="2.95",
            height="2.25",
            fixedsize="true",
            imagescale="both",
            label="",
        )
        graph.attr("edge", color="#123a66", penwidth="2.2", arrowsize="0.95")

        for node_id in node_images:
            graph.node(node_id, image=str(node_card_paths[node_id]))

        with graph.subgraph() as rank:
            rank.attr(rank="same")
            rank.node("dapi_input")
            rank.node("rfp_input")
        with graph.subgraph() as rank:
            rank.attr(rank="same")
            rank.node("dapi_morph")
            rank.node("rfp_morph")

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
) -> None:
    if dry_run:
        _log(f"DRY RUN: would generate {output_path}")
        return

    quality_ref = _load_rgb(cfg.paths_for_generation["quality_hue_reference"])
    source_overlay = _load_rgb(cfg.paths_for_generation["paired_overlay_original"])
    good_crypts = _load_rgb(cfg.paths_for_generation["good_crypts"])
    good_mask = _nonblack_mask(good_crypts, threshold=0.08)
    scale = float(np.quantile(source_overlay, 0.995))
    if scale <= 1e-6:
        scale = 1.0
    source_display = np.clip(source_overlay / scale, 0.0, 1.0)

    fig = plt.figure(figsize=(18, 10), dpi=320)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.38), height_ratios=(1.0, 0.86), wspace=0.07, hspace=0.08)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_cumulative = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[:, 1])

    ax_line.imshow(source_display)
    if np.any(good_mask):
        edge_only = good_mask & (
            ~(
                np.roll(good_mask, 1, axis=0)
                & np.roll(good_mask, -1, axis=0)
                & np.roll(good_mask, 1, axis=1)
                & np.roll(good_mask, -1, axis=1)
            )
        )
        edge_only = ndi.binary_dilation(edge_only, iterations=1)
        line_overlay = np.zeros_like(source_overlay)
        line_overlay[good_mask] = np.array([0.90, 0.10, 0.10])
        ax_line.imshow(line_overlay, alpha=0.32)
        edge_overlay = np.zeros_like(source_overlay)
        edge_overlay[edge_only] = np.array([0.35, 0.98, 0.30])
        ax_line.imshow(edge_overlay, alpha=0.95)

        primary_axis = _fit_primary_axis(good_mask)
        if primary_axis is not None:
            center, direction = primary_axis
            segment = _line_segment_in_image(
                width=good_mask.shape[1],
                height=good_mask.shape[0],
                center_xy=center,
                direction_xy=direction,
            )
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                ax_line.plot([x0, x1], [y0, y1], color="#ffd75a", lw=3.0)
    ax_line.set_title("Line-Fit Cue Through Good Crypt Regions", fontsize=15, weight="bold")
    ax_line.axis("off")

    ax_cumulative.imshow(quality_ref)
    ax_cumulative.set_title("Cumulative Quality Hue Reference", fontsize=15, weight="bold")
    ax_cumulative.text(
        0.02,
        0.03,
        "Separate global reference: cumulative weighted quality map.",
        transform=ax_cumulative.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=(0.02, 0.07, 0.15, 0.70), edgecolor="none"),
    )
    ax_cumulative.axis("off")

    ax_tbl.set_title("Weighted Quality Criteria", fontsize=18, weight="bold", pad=10)
    ax_tbl.set_xlim(0.0, 1.0)
    ax_tbl.set_ylim(0.0, 1.0)
    ax_tbl.axis("off")

    metric_descriptions = {
        "circularity": "Lower subscore when shape better matches the target crypt roundness profile.",
        "area": "Lower subscore when region size remains in expected crypt area range.",
        "line_fit": "Lower subscore when region centroid aligns with the fitted crypt axis.",
        "red_intensity": "Lower subscore when per-area lysozyme intensity is stronger.",
    }
    ordered_metrics = ["circularity", "area", "line_fit", "red_intensity"]
    rows: list[tuple[str, str, str]] = []
    for metric in ordered_metrics:
        if metric in scoring_weights:
            rows.append(
                (
                    metric,
                    _format_weight(scoring_weights.get(metric)),
                    metric_descriptions.get(metric, "Lower subscore indicates better match."),
                )
            )
    for metric in sorted(scoring_weights.keys()):
        if metric in {"com_consistency", *ordered_metrics}:
            continue
        rows.append(
            (
                metric,
                _format_weight(scoring_weights.get(metric)),
                metric_descriptions.get(metric, "Lower subscore indicates better match."),
            )
        )

    col_edges = [0.02, 0.19, 0.30, 0.60, 0.98]
    header_top = 0.94
    header_h = 0.09
    rows_count = max(len(rows), 1)
    row_h = min(0.14, (header_top - 0.20 - header_h) / rows_count)
    table_bottom = header_top - header_h - rows_count * row_h
    header_labels = ["Metric", "Weight", "Quality Hue Ref", "Interpretation"]

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

    gradient = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    hue_strip = plt.get_cmap("turbo")(gradient)[..., :3]

    for idx, (metric, weight, interpretation) in enumerate(rows):
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
        hue_y0 = y0 + row_h * 0.24
        hue_y1 = y1 - row_h * 0.24
        ax_tbl.imshow(hue_strip, extent=[hue_x0, hue_x1, hue_y0, hue_y1], aspect="auto", zorder=2)
        ax_tbl.text(hue_x0 - 0.005, y0 + row_h * 0.50, "low", ha="right", va="center", fontsize=8, color="#25496d")
        ax_tbl.text(hue_x1 + 0.005, y0 + row_h * 0.50, "high", ha="left", va="center", fontsize=8, color="#25496d")

        ax_tbl.text(
            col_edges[3] + 0.010,
            y0 + row_h * 0.50,
            textwrap.fill(interpretation, width=42),
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

    formula_terms = [f"{_format_weight(scoring_weights.get(metric))}*{metric}_score" for metric, _, _ in rows]
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

    eff_text = "Effective-count weights: " + ", ".join(
        f"{k}={_format_weight(v)}" for k, v in effective_weights.items()
    )
    ax_tbl.text(0.02, 0.05, eff_text, ha="left", va="bottom", fontsize=9, color="#27496e")

    fig.suptitle("Scoring and Selection", fontsize=22, weight="bold", y=0.98)
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
        We extract seeds from continuous high-likelihood areas, grow these into base labels, and then derive final crypt labels for downstream measurements and overlays.

        ## 4) Weighted quality scoring
        Candidate regions are scored using circularity, area, line-fit alignment, and red-intensity features. Lower weighted score means better match; highest-ranked regions are retained for final reporting.
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
        - `N4` -> `assets/N4_quality_scoring_breakdown.png`: Scoring criteria, weights, and quality-hue interpretation.

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
        Morphology-guided crypt likelihood from multi-channel overlap, then seed-to-region progression.

        Large Text Box
        We combined information across multiple fluorescence channels and applied morphology based filtering to emphasize structures consistent with the expected crypt appearance. This produced a likelihood map where higher values indicate locations whose intensity and local spatial pattern best match the target profile, even when diffuse staining is present.

        In RFP, crypt-like signal is modeled as local peaks that are relatively large, locally stable in intensity, and approximately round, with strong transitions near borders. In DAPI, we estimate tissue boundaries and cavity-like spaces where crypt lumens are expected to have low signal. Overlap between these maps highlights high-likelihood crypt centers. We then extract seed labels and grow base labels on the same grayscale context image.
        """
    ).strip() + "\n"

    files["N4_quality_scoring_breakdown.txt"] = textwrap.dedent(
        f"""
        Subtitle
        Weighted scoring of candidate crypt regions with explicit feature weights.

        Large Text Box
        Candidate regions are scored by shape and signal properties, then ranked so lower total score indicates a better crypt match.

        Selection weights (from config):
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


def build_bundle(cfg: SubjectConfig, dry_run: bool) -> None:
    required = [cfg.scoring_config_path, *cfg.paths_for_generation.values()]
    _verify_exists(required)
    _ensure_dirs(dry_run=dry_run)

    curated_rows = _copy_curated_assets(cfg=cfg, dry_run=dry_run)
    scoring_weights, effective_weights = _load_scoring_weights(cfg.scoring_config_path)

    n1_path = GENERATED_DIR / "N1_pipeline_flowchart.png"
    n2_path = GENERATED_DIR / "N2_channel_split_standardization.png"
    n3_path = GENERATED_DIR / "N3_morphology_seed_flowchart.png"
    n4_path = GENERATED_DIR / "N4_quality_scoring_breakdown.png"

    _generate_n1_pipeline_flowchart(output_path=n1_path, dry_run=dry_run)
    _generate_n2_channel_split_standardization(
        cfg=cfg,
        output_path=n2_path,
        dry_run=dry_run,
    )
    _generate_n3_morphology_seed_flow(
        cfg=cfg,
        output_path=n3_path,
        dry_run=dry_run,
    )
    _generate_n4_quality_scoring_breakdown(
        cfg=cfg,
        output_path=n4_path,
        scoring_weights=scoring_weights,
        effective_weights=effective_weights,
        dry_run=dry_run,
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
    build_bundle(cfg=cfg, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
