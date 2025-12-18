#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ManualSource:
    reviewer: str
    csv_path: str


def _normalize_manual_metadata_key(raw_key: str) -> str:
    if not isinstance(raw_key, str):
        return ""

    key = raw_key.strip()

    key = key.replace("+", "")
    key = re.sub(r"_(combined_channels|separate_channels)$", "", key)
    key = re.sub(r"_\(\d+\)", "", key)
    key = re.sub(r"__+", "_", key)
    key = key.strip("_")
    return key


def _subject_name_to_metadata_key(subject_name: str) -> str:
    if not isinstance(subject_name, str):
        return ""

    s = subject_name.strip()
    s = s.replace("+", "")
    s = s.replace(" - ", "_-_")
    s = s.replace(" [", "_")
    s = s.replace("]", "")
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    return s


def _auto_metadata_key_to_base(auto_key: str, manual_keys: list[str]) -> str:
    if not isinstance(auto_key, str) or not auto_key:
        return ""

    matches = [k for k in manual_keys if auto_key.startswith(k)]
    if not matches:
        return auto_key
    return max(matches, key=len)


def _load_manual_source(source: ManualSource) -> pd.DataFrame:
    df = pd.read_csv(source.csv_path)

    required_cols = {"metadata_key", "rating_bool", "Total Area"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manual CSV missing columns {sorted(missing)}: {source.csv_path}")

    df = df.copy()
    df["reviewer"] = source.reviewer
    df["manual_source_file"] = os.path.basename(source.csv_path)
    df["metadata_key_raw"] = df["metadata_key"].astype(str)
    df["metadata_key"] = df["metadata_key_raw"].map(_normalize_manual_metadata_key)
    df["manual_area_px"] = pd.to_numeric(df["Total Area"], errors="coerce")

    df = df[df["rating_bool"] == True]  # noqa: E712 (explicit True is intentional)
    df = df[df["manual_area_px"].notna()]
    df = df[df["metadata_key"].astype(bool)]

    return df[["metadata_key", "reviewer", "manual_area_px", "manual_source_file"]]


def _pivot_manual(manual_long: pd.DataFrame) -> pd.DataFrame:
    manual_area = (
        manual_long.pivot_table(
            index="metadata_key",
            columns="reviewer",
            values="manual_area_px",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    manual_source = (
        manual_long.pivot_table(
            index="metadata_key",
            columns="reviewer",
            values="manual_source_file",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in manual_area.columns:
        if col != "metadata_key":
            manual_area = manual_area.rename(columns={col: f"manual_area_{col}"})
    for col in manual_source.columns:
        if col != "metadata_key":
            manual_source = manual_source.rename(columns={col: f"manual_source_file_{col}"})

    return manual_area.merge(manual_source, on="metadata_key", how="left")


def _plot_relation(
    df: pd.DataFrame,
    *,
    image_source_type: str,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    df = df[df["image_source_type"] == image_source_type].copy()
    df = df[df[y_col].notna() & df[x_col].notna()]

    if df.empty:
        return

    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, alpha=0.75)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1, linestyle="--", color="black", alpha=0.7)

    corr = float(x.corr(y)) if len(df) >= 2 else float("nan")
    ax.set_title(f"{title} ({image_source_type}) | r={corr:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare manual ImageJ areas vs auto pipeline areas (px) for lysozyme."
    )
    parser.add_argument(
        "--auto-csv",
        default="results/normal_out/simple_dask_image_summary_detailed_simpson.csv",
        help="Auto summary CSV (detailed simpson; used for effective count).",
    )
    parser.add_argument(
        "--auto-per-crypt-csv",
        default="results/normal_out/simple_dask_per_crypt.csv",
        help="Auto per-crypt CSV (fallback for pixel areas if summary lacks pixel-area columns).",
    )
    parser.add_argument(
        "--manual-adam-csv",
        default="/home/ash/documents/data/inputs/karen/lysozyme/manual_analysis_results/image_j_quantification_eva_04222025(autoImages - A).csv",
        help="Manual ImageJ CSV from Adam (autoImages - A).",
    )
    parser.add_argument(
        "--manual-hadley-csv",
        default="/home/ash/documents/data/inputs/karen/lysozyme/manual_analysis_results/image_j_quantification_eva_04222025(autoImages - H ).csv",
        help="Manual ImageJ CSV from Hadley (autoImages - H).",
    )
    parser.add_argument(
        "--out-dir",
        default="method_development/match_data_out_to_manaul/out",
        help="Output directory for merged CSV and plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auto = pd.read_csv(args.auto_csv)
    required_auto_cols = {
        "subject_name",
        "image_source_type",
        "simpson_effective_count",
    }
    missing_auto = required_auto_cols - set(auto.columns)
    if missing_auto:
        raise ValueError(f"Auto CSV missing columns {sorted(missing_auto)}: {args.auto_csv}")

    extra_cols = [
        c
        for c in [
            "selected_crypt_area_px_sum",
            "selected_crypt_area_px_std",
            "rfp_gt_threshold",
            "selected_rfp_px_gt_threshold",
        ]
        if c in auto.columns
    ]
    auto = auto[["subject_name", "image_source_type", "simpson_effective_count", *extra_cols]].copy()
    auto["auto_effective_count"] = auto["simpson_effective_count"].astype(float)
    auto = auto.drop(columns=["simpson_effective_count"])

    if "selected_crypt_area_px_sum" in auto.columns:
        auto = auto.rename(
            columns={
                "selected_crypt_area_px_sum": "auto_area_px",
                "selected_crypt_area_px_std": "auto_area_px_std",
                "selected_rfp_px_gt_threshold": "auto_rfp_px_gt_threshold",
            }
        )
    else:
        per_crypt = pd.read_csv(args.auto_per_crypt_csv)
        required_per_crypt_cols = {"subject_name", "image_source_type", "pixel_area"}
        missing_per_crypt = required_per_crypt_cols - set(per_crypt.columns)
        if missing_per_crypt:
            raise ValueError(
                f"Auto per-crypt CSV missing columns {sorted(missing_per_crypt)}: {args.auto_per_crypt_csv}"
            )

        per_crypt = per_crypt.copy()
        per_crypt["pixel_area"] = pd.to_numeric(per_crypt["pixel_area"], errors="coerce")
        per_crypt = per_crypt[per_crypt["pixel_area"].notna()]
        per_crypt_agg = (
            per_crypt.groupby(["subject_name", "image_source_type"], as_index=False)
            .agg(auto_area_px=("pixel_area", "sum"), auto_area_px_std=("pixel_area", "std"))
            .copy()
        )
        per_crypt_agg["auto_area_px_std"] = per_crypt_agg["auto_area_px_std"].fillna(0.0)
        auto = auto.merge(per_crypt_agg, on=["subject_name", "image_source_type"], how="left")

    if "auto_area_px_std" not in auto.columns:
        auto["auto_area_px_std"] = 0.0

    if "auto_rfp_px_gt_threshold" not in auto.columns:
        auto["auto_rfp_px_gt_threshold"] = pd.NA
    if "rfp_gt_threshold" not in auto.columns:
        auto["rfp_gt_threshold"] = pd.NA

    manual_sources = [
        ManualSource(reviewer="Adam", csv_path=args.manual_adam_csv),
        ManualSource(reviewer="Hadley", csv_path=args.manual_hadley_csv),
    ]
    manual_long = pd.concat([_load_manual_source(s) for s in manual_sources], ignore_index=True)
    manual_wide = _pivot_manual(manual_long)

    manual_keys = sorted(manual_wide["metadata_key"].dropna().astype(str).unique().tolist(), key=len, reverse=True)
    auto["metadata_key_guess"] = auto["subject_name"].map(_subject_name_to_metadata_key)
    auto["metadata_key"] = auto["metadata_key_guess"].map(lambda k: _auto_metadata_key_to_base(k, manual_keys))

    merged = auto.merge(manual_wide, on="metadata_key", how="left")
    merged = merged[merged[[c for c in merged.columns if c.startswith("manual_area_")]].notna().any(axis=1)].copy()

    manual_area_cols = [c for c in merged.columns if c.startswith("manual_area_")]
    merged["manual_area_mean"] = merged[manual_area_cols].mean(axis=1, skipna=True)
    merged["auto_area_px_corrected_5"] = (
        merged["auto_area_px"].astype(float) / merged["auto_effective_count"].astype(float) * 5.0
    )
    merged["auto_rfp_px_gt_threshold_corrected_5"] = (
        pd.to_numeric(merged["auto_rfp_px_gt_threshold"], errors="coerce")
        / merged["auto_effective_count"].astype(float)
        * 5.0
    )

    out_cols = [
        "subject_name",
        "metadata_key",
        "image_source_type",
        *sorted(manual_area_cols),
        "manual_area_mean",
        "auto_area_px",
        "auto_area_px_std",
        "auto_effective_count",
        "auto_area_px_corrected_5",
        "rfp_gt_threshold",
        "auto_rfp_px_gt_threshold",
        "auto_rfp_px_gt_threshold_corrected_5",
    ]
    merged[[c for c in out_cols if c in merged.columns]].sort_values(
        ["subject_name", "image_source_type"]
    ).to_csv(
        out_dir / "manual_vs_auto_area_comparison.csv", index=False
    )

    pd.DataFrame(
        [{"reviewer": s.reviewer, "csv_path": s.csv_path} for s in manual_sources]
    ).to_csv(out_dir / "manual_sources.csv", index=False)

    for image_source_type in ["separate_channels", "combined_channels"]:
        _plot_relation(
            merged,
            image_source_type=image_source_type,
            x_col="auto_area_px",
            y_col="manual_area_mean",
            title="Auto vs manual area (uncorrected)",
            xlabel="Auto selected crypt area (px)",
            ylabel="Manual mean area (px)",
            out_path=out_dir / f"auto_vs_manual_area_uncorrected_{image_source_type}.png",
        )
        _plot_relation(
            merged,
            image_source_type=image_source_type,
            x_col="auto_area_px_corrected_5",
            y_col="manual_area_mean",
            title="Auto vs manual area (corrected to 5 crypts)",
            xlabel="Auto selected crypt area corrected to 5 crypts (px)",
            ylabel="Manual mean area (px)",
            out_path=out_dir / f"auto_vs_manual_area_corrected_{image_source_type}.png",
        )
        _plot_relation(
            merged,
            image_source_type=image_source_type,
            x_col="auto_rfp_px_gt_threshold",
            y_col="manual_area_mean",
            title="Auto RFP>threshold pixels vs manual area (uncorrected)",
            xlabel="Auto pixels in crypts with RFP>threshold (px)",
            ylabel="Manual mean area (px)",
            out_path=out_dir / f"auto_gt_threshold_vs_manual_area_uncorrected_{image_source_type}.png",
        )
        _plot_relation(
            merged,
            image_source_type=image_source_type,
            x_col="auto_rfp_px_gt_threshold_corrected_5",
            y_col="manual_area_mean",
            title="Auto RFP>threshold pixels vs manual area (corrected to 5 crypts)",
            xlabel="Auto pixels in crypts with RFP>threshold corrected to 5 crypts (px)",
            ylabel="Manual mean area (px)",
            out_path=out_dir / f"auto_gt_threshold_vs_manual_area_corrected_{image_source_type}.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
