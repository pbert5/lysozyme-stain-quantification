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
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLAINING_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXPLAINING_DIR / "assets"
GENERATED_DIR = EXPLAINING_DIR / "generated"
FIGURE_TEXT_DIR = EXPLAINING_DIR / "figure_text"

PLANNED_ANIMATION_ROOT = Path(
    "/home/ash/documents/code/morphological_animation_toolkit/planned_animation"
)
LYSOZYME_ROOT = Path("/home/ash/documents/code/lysozyme")


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

    fig, ax = plt.subplots(figsize=(18, 5.1), dpi=320)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "Lysozyme Pipeline Overview (Poster Flow)",
        fontsize=23,
        weight="bold",
        pad=20,
    )

    box_texts = [
        "Input field\n(DAPI + RFP)",
        "Split channels\nand standardize\nintensity",
        "Morphology maps:\nDAPI borders +\nRFP peaks",
        "Channel overlap\n-> distance /\nlikelihood map",
        "Seed labels\n-> base labels\n-> final regions",
        "Weighted scoring\nfor final\ncrypt selection",
    ]
    box_colors = ["#1d3557", "#274b74", "#2f628f", "#3b7ca8", "#4d97bf", "#67b4d4"]
    x_positions = np.linspace(0.02, 0.84, num=6)
    width, height = 0.135, 0.38
    y = 0.27

    for idx, (x, text, color) in enumerate(zip(x_positions, box_texts, box_colors), start=1):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=2,
            edgecolor="#0f2138",
            facecolor=color,
            alpha=0.98,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            color="white",
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
        )
        ax.text(
            x + width / 2,
            y - 0.06,
            f"Step {idx}",
            color="#16345b",
            fontsize=11,
            ha="center",
            va="center",
            weight="bold",
        )

    for i in range(5):
        x0 = x_positions[i] + width + 0.006
        x1 = x_positions[i + 1] - 0.006
        ax.annotate(
            "",
            xy=(x1, y + height / 2),
            xytext=(x0, y + height / 2),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#10325b"),
        )

    ax.text(
        0.5,
        0.08,
        "Start with paired channels, build a morphology-informed likelihood map, then score and keep the strongest crypt regions.",
        ha="center",
        va="center",
        fontsize=12,
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

    rfp_raw = _load_rgb(cfg.paths_for_generation["rfp_input"])
    dapi_raw = _load_rgb(cfg.paths_for_generation["dapi_input"])
    rfp_std = _load_rgb(cfg.paths_for_generation["crypt_preprocessed"])
    dapi_std = _load_rgb(cfg.paths_for_generation["tissue_preprocessed"])

    merged_raw = _merge_channels(_grayscale(dapi_raw), _grayscale(rfp_raw))

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

    ax_input.imshow(merged_raw)
    ax_input.set_title("Original paired field (DAPI + RFP)", fontsize=16, weight="bold")
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
        "Each channel is normalized independently so morphology filters operate on comparable contrast ranges.",
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

    tissue_pre = _load_rgb(cfg.paths_for_generation["tissue_preprocessed"])
    crypt_pre = _load_rgb(cfg.paths_for_generation["crypt_preprocessed"])
    tissue_caps = _load_rgb(cfg.paths_for_generation["tissue_caps_troughs"])
    good_crypts = _load_rgb(cfg.paths_for_generation["good_crypts"])
    distance_img = _load_rgb(cfg.paths_for_generation["distance_image"])
    seed_labels = _load_rgb(cfg.paths_for_generation["seed_labels"])
    base_labels = _load_rgb(cfg.paths_for_generation["base_labels"])
    final_labels = _load_rgb(cfg.paths_for_generation["final_labels"])

    dapi_gray = _grayscale(tissue_pre)
    rfp_gray = _grayscale(crypt_pre)

    dapi_vis = np.stack([0.12 * dapi_gray, 0.25 * dapi_gray, np.clip(1.05 * dapi_gray, 0.0, 1.0)], axis=-1)
    rfp_vis = np.stack([np.clip(1.05 * rfp_gray, 0.0, 1.0), 0.18 * rfp_gray, 0.16 * rfp_gray], axis=-1)

    context = np.repeat(dapi_gray[..., None], 3, axis=2) * 0.50
    tissue_mask = _nonblack_mask(tissue_caps, threshold=0.08)
    crypt_mask = _nonblack_mask(good_crypts, threshold=0.08)
    overlap = context.copy()
    overlap[tissue_mask] = np.clip(overlap[tissue_mask] + np.array([0.08, 0.20, 0.70]), 0.0, 1.0)
    overlap[crypt_mask] = np.clip(overlap[crypt_mask] + np.array([0.75, 0.10, 0.10]), 0.0, 1.0)

    distance_gray = np.repeat(_grayscale(distance_img)[..., None], 3, axis=2)
    seed_mask = _nonblack_mask(seed_labels, threshold=0.08)
    base_mask = _nonblack_mask(base_labels, threshold=0.08)
    final_mask = _nonblack_mask(final_labels, threshold=0.08)

    seeds_on_distance = _overlay(distance_gray, seed_labels, seed_mask, alpha=0.92)
    base_on_distance = _overlay(distance_gray, base_labels, base_mask, alpha=0.55)

    merged_input = _merge_channels(dapi_gray, rfp_gray)
    final_on_merged = _overlay(merged_input, final_labels, final_mask, alpha=0.62)

    fig, axes = plt.subplots(2, 4, figsize=(20, 11), dpi=320)
    fig.suptitle(
        "Morphology-Guided Likelihood and Seeded Region Building",
        fontsize=21,
        weight="bold",
        y=0.97,
    )

    panels = [
        (dapi_vis, "DAPI input", "Cell/tissue channel after standardization."),
        (tissue_caps, "DAPI morphology", "Tissue edge + cavity map (caps/troughs)."),
        (rfp_vis, "RFP input", "Lysozyme channel after standardization."),
        (overlap, "Channel overlap", "Blue: DAPI cavity context, Red: RFP candidate regions."),
        (distance_gray, "Distance image", "Morphology-driven likelihood surface."),
        (seeds_on_distance, "Seed labels", "Unique seed areas found on the distance map."),
        (base_on_distance, "Base labels", "Seeded growth into base crypt regions."),
        (final_on_merged, "Final labels on input", "Selected crypt regions on merged channels."),
    ]

    for ax, (img, title, subtitle) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#133454",
        )
        ax.axis("off")

    for col in range(3):
        _add_arrow_between_axes(fig, axes[0, col], axes[0, col + 1])
    for col in range(3):
        _add_arrow_between_axes(fig, axes[1, col], axes[1, col + 1])

    top_last = axes[0, 3].get_position()
    bottom_first = axes[1, 0].get_position()
    down_arrow = FancyArrowPatch(
        (top_last.x0 + top_last.width * 0.50, top_last.y0),
        (bottom_first.x0 + bottom_first.width * 0.50, bottom_first.y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        lw=2.0,
        color="#123a66",
    )
    fig.add_artist(down_arrow)

    morph_text = (
        "We combined information across multiple fluorescence channels and applied morphology "
        "based filtering to emphasize structures consistent with the expected crypt appearance. "
        "This produced a likelihood map where higher values indicate locations whose intensity and "
        "local spatial pattern best match the target profile, even when diffuse staining is present."
    )
    fig.text(
        0.5,
        0.02,
        textwrap.fill(morph_text, width=170),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#12324f",
    )

    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
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

    fig = plt.figure(figsize=(16, 8.8), dpi=320)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.06)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    ax_img.imshow(quality_ref)
    ax_img.set_title("Quality Hue Reference", fontsize=16, weight="bold")
    h, w = quality_ref.shape[:2]
    ax_img.add_patch(Circle((w * 0.30, h * 0.55), radius=min(w, h) * 0.09, fill=False, ec="white", lw=2.0))
    ax_img.add_patch(Circle((w * 0.63, h * 0.43), radius=min(w, h) * 0.08, fill=False, ec="#f8d44b", lw=2.0))
    ax_img.plot([w * 0.16, w * 0.86], [h * 0.80, h * 0.22], "--", color="white", lw=2.2)
    ax_img.text(w * 0.18, h * 0.17, "line fit", color="white", fontsize=11, weight="bold")
    ax_img.text(w * 0.22, h * 0.66, "circularity", color="white", fontsize=11, weight="bold")
    ax_img.axis("off")

    ax_tbl.axis("off")
    ax_tbl.set_title("Weighted Quality Criteria", fontsize=16, weight="bold")

    rows = [
        ["circularity", _format_weight(scoring_weights.get("circularity")), "Lower score if shape is closer to target roundness"],
        ["area", _format_weight(scoring_weights.get("area")), "Lower score for larger regions in expected size range"],
        ["line_fit", _format_weight(scoring_weights.get("line_fit")), "Lower score when centroid lies close to main axis"],
        ["red_intensity", _format_weight(scoring_weights.get("red_intensity")), "Lower score for brighter per-area lysozyme signal"],
        ["com_consistency", _format_weight(scoring_weights.get("com_consistency")), "Configured legacy term (compatibility only)"],
    ]

    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Metric", "Weight", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.22, 0.14, 0.62],
        bbox=[0.02, 0.30, 0.96, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.42)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1f4368")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f5f9ff" if row_idx % 2 == 1 else "white")

    c_w = _format_weight(scoring_weights.get("circularity"))
    a_w = _format_weight(scoring_weights.get("area"))
    l_w = _format_weight(scoring_weights.get("line_fit"))
    r_w = _format_weight(scoring_weights.get("red_intensity"))
    formula_text = (
        f"quality_score = {c_w}*circularity_score + {a_w}*area_score "
        f"+ {l_w}*line_fit_score + {r_w}*red_intensity_score\n"
        "(Lower total score ranks higher; top regions are selected.)"
    )
    ax_tbl.text(
        0.03,
        0.20,
        formula_text,
        transform=ax_tbl.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#16345a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef5ff", edgecolor="#b7c8de"),
    )

    eff_text = "Effective-count weights: " + ", ".join(
        f"{k}={_format_weight(v)}" for k, v in effective_weights.items()
    )
    ax_tbl.text(
        0.03,
        0.08,
        eff_text,
        transform=ax_tbl.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#27496e",
    )

    fig.suptitle("Scoring and Selection", fontsize=21, weight="bold", y=0.98)
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

        In RFP, crypt-like signal is modeled as local peaks that are relatively large, locally stable in intensity, and approximately round, with strong transitions near borders. In DAPI, we estimate tissue boundaries and cavity-like spaces where crypt lumens are expected to have low signal. Overlap between these maps highlights high-likelihood crypt centers. We then extract seeds, grow base labels, and map final crypt labels back onto the original channel context.
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
        - com_consistency: {_format_weight(scoring_weights.get('com_consistency'))} (legacy compatibility term)

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
