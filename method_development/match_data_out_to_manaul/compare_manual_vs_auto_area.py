#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class CriticalParams:
    rfp_gt_threshold_cut: int = 71
    acceptable_peak_px: int = 20_000
    drop_auto_gt_px: int = 50_000


CRIT = CriticalParams()


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


def _filter_xy(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    max_x: int | None = None,
    max_y: int | None = None,
) -> pd.DataFrame:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if max_x is not None:
        mask &= x.astype(float) <= float(max_x)
    if max_y is not None:
        mask &= y.astype(float) <= float(max_y)
    return df.loc[mask].copy()


def _assoc_stats(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    max_x: int | None = None,
    max_y: int | None = None,
) -> dict[str, float | int]:
    df = _filter_xy(df, x_col=x_col, y_col=y_col, max_x=max_x, max_y=max_y)
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    x = x.astype(float)
    y = y.astype(float)
    n = int(len(df))
    pearson_r = float("nan")
    pearson_p = float("nan")
    spearman_rho = float("nan")
    spearman_p = float("nan")
    brown_forsythe_stat = float("nan")
    brown_forsythe_p = float("nan")
    if n >= 2:
        try:
            pearson_r, pearson_p = sps.pearsonr(x, y)
        except Exception:
            pearson_r, pearson_p = float("nan"), float("nan")
        try:
            spearman_rho, spearman_p = sps.spearmanr(x, y)
        except Exception:
            spearman_rho, spearman_p = float("nan"), float("nan")
        try:
            brown_forsythe_stat, brown_forsythe_p = sps.levene(x, y, center="median")
        except Exception:
            brown_forsythe_stat, brown_forsythe_p = float("nan"), float("nan")

    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "brown_forsythe_stat": float(brown_forsythe_stat),
        "brown_forsythe_p": float(brown_forsythe_p),
    }


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
    max_x: int | None = None,
    max_y: int | None = None,
) -> None:
    df = df[df["image_source_type"] == image_source_type].copy()
    df = _filter_xy(df, x_col=x_col, y_col=y_col, max_x=max_x, max_y=max_y)

    if df.empty:
        return

    stats = _assoc_stats(df, x_col=x_col, y_col=y_col, max_x=max_x, max_y=max_y)
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, alpha=0.75)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1, linestyle="--", color="black", alpha=0.7)

    title_text = textwrap.fill(f"{title} ({image_source_type})", width=54)
    ax.set_title(f"{title_text}\n(n={stats['n']})", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02,
        0.98,
        f"Pearson r={stats['pearson_r']:.3f}\nSpearman ρ={stats['spearman_rho']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return


def _plot_distribution(
    df: pd.DataFrame,
    *,
    image_source_type: str,
    col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    max_x: int | None = None,
) -> None:
    series = pd.to_numeric(
        df.loc[df["image_source_type"] == image_source_type, col],
        errors="coerce",
    ).dropna()
    if max_x is not None:
        series = series[series.astype(float) <= float(max_x)]
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(series.astype(float), bins=30, alpha=0.85, edgecolor="white", linewidth=0.6)
    title_text = textwrap.fill(f"{title} ({image_source_type})", width=54)
    ax.set_title(f"{title_text}\n(n={len(series)})", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
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
        "--rfp-gt-threshold-cut",
        type=int,
        default=CRIT.rfp_gt_threshold_cut,
        help="Fallback RFP>threshold cut used when auto CSV lacks rfp_gt_threshold.",
    )
    parser.add_argument(
        "--acceptable-peak-px",
        type=int,
        default=CRIT.acceptable_peak_px,
        help="Reference value for distribution peak checks (written to CSV).",
    )
    parser.add_argument(
        "--drop-auto-gt-px",
        type=int,
        default=CRIT.drop_auto_gt_px,
        help="Drop rows where auto metrics exceed this pixel count (outlier filter).",
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

    crit = CriticalParams(
        rfp_gt_threshold_cut=int(args.rfp_gt_threshold_cut),
        acceptable_peak_px=int(args.acceptable_peak_px),
        drop_auto_gt_px=int(args.drop_auto_gt_px),
    )

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
        auto["rfp_gt_threshold"] = int(crit.rfp_gt_threshold_cut)

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

    # Drop extreme auto outliers globally (shareable CSV + stats sanity).
    merged = merged[
        pd.to_numeric(merged["auto_rfp_px_gt_threshold"], errors="coerce").le(float(crit.drop_auto_gt_px))
        | pd.to_numeric(merged["auto_rfp_px_gt_threshold"], errors="coerce").isna()
    ].copy()
    merged = merged[
        pd.to_numeric(merged["auto_rfp_px_gt_threshold_corrected_5"], errors="coerce").le(float(crit.drop_auto_gt_px))
        | pd.to_numeric(merged["auto_rfp_px_gt_threshold_corrected_5"], errors="coerce").isna()
    ].copy()

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

    plot_specs = [
        dict(
            x_col="auto_area_px",
            y_col="manual_area_mean",
            title="Auto selected-crypt pixel area vs manual area (uncorrected)",
            xlabel="Auto selected-crypt area (px)",
            ylabel="Manual mean area (px)",
            stem="auto_vs_manual_area_uncorrected",
        ),
        dict(
            x_col="auto_area_px_corrected_5",
            y_col="manual_area_mean",
            title="Auto selected-crypt pixel area vs manual area (corrected to 5 crypts)",
            xlabel="Auto selected-crypt area corrected to 5 crypts (px)",
            ylabel="Manual mean area (px)",
            stem="auto_vs_manual_area_corrected",
        ),
        dict(
            x_col="auto_rfp_px_gt_threshold",
            y_col="manual_area_mean",
            title="Auto pixels in crypts with RFP>threshold vs manual area (uncorrected)",
            xlabel="Auto pixels in crypts with RFP>threshold (px)",
            ylabel="Manual mean area (px)",
            stem="auto_gt_threshold_vs_manual_area_uncorrected",
            max_x=crit.drop_auto_gt_px,
        ),
        dict(
            x_col="auto_rfp_px_gt_threshold_corrected_5",
            y_col="manual_area_mean",
            title="Auto pixels in crypts with RFP>threshold vs manual area (corrected to 5 crypts)",
            xlabel="Auto pixels in crypts with RFP>threshold corrected to 5 crypts (px)",
            ylabel="Manual mean area (px)",
            stem="auto_gt_threshold_vs_manual_area_corrected",
            max_x=crit.drop_auto_gt_px,
        ),
    ]

    assoc_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    for image_source_type in ["separate_channels", "combined_channels"]:
        for spec in plot_specs:
            stats = _assoc_stats(
                merged.loc[merged["image_source_type"] == image_source_type],
                x_col=spec["x_col"],
                y_col=spec["y_col"],
                max_x=spec.get("max_x", None),
            )
            assoc_rows.append(
                {
                    "image_source_type": image_source_type,
                    "x_col": spec["x_col"],
                    "y_col": spec["y_col"],
                    "n": stats["n"],
                    "pearson_r": stats["pearson_r"],
                    "pearson_p": stats["pearson_p"],
                    "spearman_rho": stats["spearman_rho"],
                    "spearman_p": stats["spearman_p"],
                    "brown_forsythe_stat": stats["brown_forsythe_stat"],
                    "brown_forsythe_p": stats["brown_forsythe_p"],
                    "max_x": spec.get("max_x", None),
                }
            )
            _plot_relation(
                merged,
                image_source_type=image_source_type,
                x_col=spec["x_col"],
                y_col=spec["y_col"],
                title=spec["title"],
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                out_path=out_dir / f"{spec['stem']}_{image_source_type}.png",
                max_x=spec.get("max_x", None),
            )

        dist_cols = [
            ("manual_area_mean", "Manual mean area distribution", "Manual mean area (px)"),
            ("auto_area_px", "Auto selected-crypt area distribution", "Auto selected-crypt area (px)"),
            (
                "auto_area_px_corrected_5",
                "Auto selected-crypt area distribution (corrected to 5 crypts)",
                "Auto selected-crypt area corrected to 5 crypts (px)",
            ),
            (
                "auto_rfp_px_gt_threshold",
                "Auto RFP>threshold pixel-count distribution",
                "Auto pixels in crypts with RFP>threshold (px)",
            ),
            (
                "auto_rfp_px_gt_threshold_corrected_5",
                "Auto RFP>threshold pixel-count distribution (corrected to 5 crypts)",
                "Auto pixels in crypts with RFP>threshold corrected to 5 crypts (px)",
            ),
        ]
        for col, dist_title, xlabel in dist_cols:
            if col not in merged.columns:
                continue
            max_x = crit.drop_auto_gt_px if col.startswith("auto_rfp_px_gt_threshold") else None
            _plot_distribution(
                merged,
                image_source_type=image_source_type,
                col=col,
                title=dist_title,
                xlabel=xlabel,
                out_path=out_dir / f"dist_{col}_{image_source_type}.png",
                max_x=max_x,
            )
            s = pd.to_numeric(
                merged.loc[merged["image_source_type"] == image_source_type, col],
                errors="coerce",
            ).dropna()
            if max_x is not None:
                s = s[s.astype(float) <= float(max_x)]
            if not s.empty:
                counts, edges = np.histogram(s.astype(float).to_numpy(), bins=30)
                peak_bin = int(counts.argmax())
                peak_center = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
                peak_rows.append(
                    {
                        "image_source_type": image_source_type,
                        "col": col,
                        "n": int(len(s)),
                        "peak_center_px": peak_center,
                        "acceptable_peak_px": int(crit.acceptable_peak_px),
                        "peak_le_acceptable": bool(peak_center <= float(crit.acceptable_peak_px)),
                        "max_x_applied": max_x,
                    }
                )

    pd.DataFrame(assoc_rows).to_csv(out_dir / "association_stats.csv", index=False)
    if peak_rows:
        pd.DataFrame(peak_rows).to_csv(out_dir / "distribution_peaks.csv", index=False)

    sharable_cols = [
        "subject_name",
        "metadata_key",
        "image_source_type",
        "manual_area_Adam",
        "manual_area_Hadley",
        "manual_area_mean",
        "manual_source_file_Adam",
        "manual_source_file_Hadley",
        "rfp_gt_threshold",
        "auto_effective_count",
        "auto_rfp_px_gt_threshold",
        "auto_rfp_px_gt_threshold_corrected_5",
    ]
    merged[[c for c in sharable_cols if c in merged.columns]].sort_values(
        ["subject_name", "image_source_type"]
    ).to_csv(out_dir / "sharable_manual_vs_auto_gated.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
