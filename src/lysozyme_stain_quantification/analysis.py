"""Lysozyme stain quantification downstream analysis (Python version)

This script replaces the prior R-based `proc.r` workflow. It consumes the
`consolidated_summary.csv` produced by the extraction pipeline and generates:

1. Filtered detection-level dataframe (visual-inspection filtered if available)
2. Per-image summary (top N ROIs per image, variability filtering)
3. Best image per animal (raw + normalized fluorescence)
4. Per-animal averages (combined sources & separated by source)
5. Normalized fluorescence using per-image crypt/background contrast
6. Summary report (missing animals by source, distribution stats)

Outputs are written under results/All/summaries/analysis/ by default.

Normalization logic:
    fluorescence_norm = fluorescence / (average_crypt_intensity / background_tissue_intensity)
                      = fluorescence * background_tissue_intensity / average_crypt_intensity

Group tag classification:
    - If subdir contains "jej lyz" (case insensitive) => Jej_LYZ
    - Else if is_retake is True OR subdir contains 'retake' => Retakes
    - Else => Normal

Usage (from repo root or any dir):
    python -m code.src.analysis

You can also import and call run_analysis() to get the results dict in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
import textwrap
import pandas as pd

# Ensure this file can be executed directly (python code/src/analysis.py) without
# requiring the package-relative imports that trigger other heavy modules.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ---------------------------------------------------------------------------
# Configuration (edit as needed)
# ---------------------------------------------------------------------------

BASE_RESULTS_DIR = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\All")
SUMMARY_DIR = BASE_RESULTS_DIR / "summaries"
CONSOLIDATED_FILE = SUMMARY_DIR / "consolidated_summary.csv"
VISUAL_RATINGS_FILE = SUMMARY_DIR / "visual_inspection_ratings.csv"
ANALYSIS_OUT_DIR = SUMMARY_DIR / "analysis"

TOP_N_ROIS_PER_IMAGE = 5          # Number of top ROIs (by raw fluorescence) to keep per image
SD_RATIO_LIMIT = 0.5              # Filter out images whose within-image SD / mean exceeds this
MIN_ROIS_REQUIRED = 1             # Skip images with fewer than this after top-N selection


@dataclass
class AnalysisOutputs:
    processed_data: pd.DataFrame
    by_image: pd.DataFrame
    best_image_per_animal: pd.DataFrame
    best_image_per_animal_normalized: pd.DataFrame
    by_animal_combined: pd.DataFrame
    by_animal_separated: pd.DataFrame
    summary_report_path: Path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _classify_source(subdir: str, is_retake: bool) -> str:
    if isinstance(subdir, str):
        s = subdir.lower()
    else:
        s = ""
    if "jej lyz" in s:
        return "Jej_LYZ"
    if is_retake or "retake" in s:
        return "Retakes"
    return "Normal"


def _load_consolidated(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Consolidated summary file not found: {path}")
    df = pd.read_csv(path)
    # Ensure object columns are strings to simplify downstream .str usage
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    # Basic sanity columns check
    required = {"image_name", "fluorescence", "background_tissue_intensity", "average_crypt_intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in consolidated file: {missing}")
    return df


def _load_good_images(visual_path: Path) -> set[str] | None:
    if not visual_path.exists():
        return None
    vr = pd.read_csv(visual_path)
    if not {"image_name", "rating"}.issubset(vr.columns):
        return None
    vr["rating"] = vr["rating"].astype(str)
    good = set(vr.loc[vr["rating"].str.lower() == "good", "image_name"].astype(str))
    return good


def _prepare_detection_level(df: pd.DataFrame, good_images: set[str] | None) -> pd.DataFrame:
    if good_images is not None and len(good_images) > 0:
        df = df[df["image_name"].isin(good_images)].copy()
    # Derive animal ID (first 4 chars) robustly
    # Coerce image_name to string to avoid .str errors
    df["image_name"] = df["image_name"].astype(str)
    df["Animal"] = df["image_name"].str.slice(0, 4)
    # Source tag
    df["source"] = [
        _classify_source(subdir, bool(is_r)) for subdir, is_r in zip(df.get("subdir", [""] * len(df)), df.get("is_retake", [False] * len(df)))
    ]
    # Contrast ratio (avoid division by zero)
    df["contrast_ratio"] = df.apply(
        lambda r: (r["average_crypt_intensity"] / r["background_tissue_intensity"]) if r["background_tissue_intensity"] not in (0, None) else pd.NA,
        axis=1,
    )
    # Normalized fluorescence
    df["normalized_fluorescence"] = df.apply(
        lambda r: (r["fluorescence"] * r["background_tissue_intensity"] / r["average_crypt_intensity"]) if r["average_crypt_intensity"] not in (0, None) else pd.NA,
        axis=1,
    )
    # Drop rows missing key values
    df = df.dropna(subset=["fluorescence", "normalized_fluorescence"])
    return df


def _build_by_image(df: pd.DataFrame) -> pd.DataFrame:
    # Select top N ROIs by raw fluorescence per image
    df_sorted = df.sort_values(["image_name", "fluorescence"], ascending=[True, False])
    df_top = df_sorted.groupby("image_name", group_keys=False).head(TOP_N_ROIS_PER_IMAGE)

    agg = (
        df_top.groupby(["image_name", "Animal", "source"], dropna=False)
        .agg(
            mean_fluorescence=("fluorescence", "mean"),
            mean_normalized_fluorescence=("normalized_fluorescence", "mean"),
            sd_fluorescence=("fluorescence", "std"),
            sd_normalized_fluorescence=("normalized_fluorescence", "std"),
            contrast_ratio=("contrast_ratio", "mean"),  # same within image; mean is fine
            background_tissue_intensity=("background_tissue_intensity", "mean"),
            average_crypt_intensity=("average_crypt_intensity", "mean"),
            n_rois=("fluorescence", "count"),
        )
        .reset_index()
    )
    # SD ratios
    agg["sd_ratio"] = agg["sd_fluorescence"] / agg["mean_fluorescence"].replace(0, pd.NA)
    agg["sd_ratio_normalized"] = agg["sd_normalized_fluorescence"] / agg["mean_normalized_fluorescence"].replace(0, pd.NA)
    # Filter variability & minimum ROI count
    filt = (agg["sd_ratio"].fillna(0) < SD_RATIO_LIMIT) & (agg["n_rois"] >= MIN_ROIS_REQUIRED)
    return agg.loc[filt].reset_index(drop=True)


def _best_image_per_animal(by_image: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    metric = "mean_normalized_fluorescence" if normalized else "mean_fluorescence"
    # For deterministic tie-breaking, sort by metric then keep first
    sub = by_image.sort_values(["Animal", "source", metric], ascending=[True, True, False])
    best = sub.groupby(["Animal", "source"], as_index=False).head(1)
    return best.reset_index(drop=True)


def _by_animal(df: pd.DataFrame, separated: bool = False) -> pd.DataFrame:
    # Re-pick top N ROIs per image before collapsing to animal level
    df_sorted = df.sort_values(["image_name", "fluorescence"], ascending=[True, False])
    df_top = df_sorted.groupby("image_name", group_keys=False).head(TOP_N_ROIS_PER_IMAGE)
    group_cols = ["Animal", "source"] if separated else ["Animal"]
    agg = (
        df_top.groupby(group_cols, dropna=False)
        .agg(
            mean_fluorescence_per_animal=("fluorescence", "mean"),
            mean_normalized_fluorescence_per_animal=("normalized_fluorescence", "mean"),
            sd_fluorescence=("fluorescence", "std"),
            sd_normalized_fluorescence=("normalized_fluorescence", "std"),
            n_images=("image_name", pd.Series.nunique),
        )
        .reset_index()
    )
    agg["CV"] = agg["sd_fluorescence"] / agg["mean_fluorescence_per_animal"].replace(0, pd.NA)
    agg["CV_normalized"] = agg["sd_normalized_fluorescence"] / agg["mean_normalized_fluorescence_per_animal"].replace(0, pd.NA)
    return agg


def _write_dataframe(df: pd.DataFrame, path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WRITE] {name}: {path}")


def _build_summary_report(outputs: AnalysisOutputs):
    df = outputs.processed_data
    all_animals = sorted(df["Animal"].unique())
    sources = sorted(df["source"].unique())

    missing_lines = []
    for src in sources:
        animals_src = set(df.loc[df["source"] == src, "Animal"].unique())
        missing = [a for a in all_animals if a not in animals_src]
        if missing:
            missing_lines.append(f"- {src}: Missing {', '.join(missing)}")
        else:
            missing_lines.append(f"- {src}: All animals present")

    report = textwrap.dedent(
        f"""
        Lysozyme Stain Quantification Analysis Summary
        ==============================================

        Parameters:
          TOP_N_ROIS_PER_IMAGE = {TOP_N_ROIS_PER_IMAGE}
          SD_RATIO_LIMIT       = {SD_RATIO_LIMIT}

        Data:
          Detections processed: {len(df)}
          Images processed:     {df['image_name'].nunique()}
          Animals:              {len(all_animals)}

        Source Distribution:
        {df['source'].value_counts().to_string()}

        Mean Contrast Ratio (per detection): {df['contrast_ratio'].mean():.3f}
        SD Contrast Ratio:                   {df['contrast_ratio'].std():.3f}

        Missing Animals by Source:
        {'\n'.join(missing_lines)}

        Output Files:
          - analysis_all_data_processed.csv
          - analysis_by_image.csv
          - analysis_best_image_per_animal.csv
          - analysis_best_image_per_animal_normalized.csv
          - analysis_by_animal_combined.csv
          - analysis_by_animal_separated.csv
          - analysis_summary.txt (this file)
        """
    ).strip() + "\n"

    outputs.summary_report_path.write_text(report, encoding="utf-8")
    print(f"[WRITE] summary report: {outputs.summary_report_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    consolidated_file: Path = CONSOLIDATED_FILE,
    visual_ratings_file: Path = VISUAL_RATINGS_FILE,
    out_dir: Path = ANALYSIS_OUT_DIR,
) -> AnalysisOutputs:
    print("[INFO] Loading consolidated summary...")
    df = _load_consolidated(consolidated_file)
    good_images = _load_good_images(visual_ratings_file)
    if good_images is not None:
        print(f"[INFO] Visual inspection filter: {len(good_images)} good images")
    else:
        print("[INFO] No visual inspection ratings found; using all images")

    print("[INFO] Preparing detection-level data...")
    df_proc = _prepare_detection_level(df, good_images)
    print(f"[INFO] Detections after filtering: {len(df_proc)}")

    print("[INFO] Building per-image summary...")
    by_image = _build_by_image(df_proc)
    print(f"[INFO] Images retained after SD filter: {len(by_image)}")

    print("[INFO] Selecting best images per animal (raw)...")
    best_raw = _best_image_per_animal(by_image, normalized=False)
    print("[INFO] Selecting best images per animal (normalized)...")
    best_norm = _best_image_per_animal(by_image, normalized=True)

    print("[INFO] Building per-animal combined (all sources) summary...")
    by_animal_combined = _by_animal(df_proc, separated=False)
    print("[INFO] Building per-animal separated (by source) summary...")
    by_animal_separated = _by_animal(df_proc, separated=True)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_dataframe(df_proc, out_dir / "analysis_all_data_processed.csv", "processed detections")
    _write_dataframe(by_image, out_dir / "analysis_by_image.csv", "by image")
    _write_dataframe(best_raw, out_dir / "analysis_best_image_per_animal.csv", "best image / animal (raw)")
    _write_dataframe(best_norm, out_dir / "analysis_best_image_per_animal_normalized.csv", "best image / animal (normalized)")
    _write_dataframe(by_animal_combined, out_dir / "analysis_by_animal_combined.csv", "by animal combined")
    _write_dataframe(by_animal_separated, out_dir / "analysis_by_animal_separated.csv", "by animal separated")

    outputs = AnalysisOutputs(
        processed_data=df_proc,
        by_image=by_image,
        best_image_per_animal=best_raw,
        best_image_per_animal_normalized=best_norm,
        by_animal_combined=by_animal_combined,
        by_animal_separated=by_animal_separated,
        summary_report_path=out_dir / "analysis_summary.txt",
    )
    _build_summary_report(outputs)

    return outputs


def main():  # pragma: no cover - CLI entry
    try:
        run_analysis()
        print("[DONE] Analysis complete.")
    except Exception as e:
        import traceback
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
