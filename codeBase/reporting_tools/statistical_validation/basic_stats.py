from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _present_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def run_basic_stats_analysis(*, image_summary_csv: Path, output_dir: Path) -> Dict[str, str]:
    """
    Compute lightweight descriptive stats for image-level results.

    Outputs:
    - basic_stats_overall.json
    - basic_stats_by_source_dataset.csv (if source_dataset is present)
    - basic_stats_by_image_source_type.csv (if image_source_type is present)
    """
    image_summary_csv = Path(image_summary_csv).expanduser().resolve()
    if not image_summary_csv.exists():
        raise FileNotFoundError(f"Image summary CSV not found: {image_summary_csv}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(image_summary_csv)
    if df.empty:
        overall_path = output_dir / "basic_stats_overall.json"
        overall_payload = {"row_count": 0, "notes": "No rows in image summary CSV."}
        overall_path.write_text(json.dumps(overall_payload, indent=2), encoding="utf-8")
        return {"overall_json": str(overall_path)}

    count_col = "Count" if "Count" in df.columns else ("crypt_count" if "crypt_count" in df.columns else None)
    area_col = "Total Area" if "Total Area" in df.columns else ("crypt_area_um2_sum" if "crypt_area_um2_sum" in df.columns else None)
    mean_col = "Mean" if "Mean" in df.columns else ("rfp_sum_mean" if "rfp_sum_mean" in df.columns else None)

    overall: Dict[str, Any] = {"row_count": int(len(df))}
    if count_col is not None:
        overall["crypt_count_column"] = count_col
        overall["crypt_count_mean"] = float(df[count_col].mean(skipna=True))
        overall["crypt_count_median"] = float(df[count_col].median(skipna=True))
        overall["crypt_count_std"] = float(df[count_col].std(skipna=True))
        overall["zero_crypt_images"] = int((df[count_col].fillna(0) <= 0).sum())
    if area_col is not None:
        overall["total_area_column"] = area_col
        overall["total_area_mean"] = float(df[area_col].mean(skipna=True))
        overall["total_area_median"] = float(df[area_col].median(skipna=True))
        overall["total_area_std"] = float(df[area_col].std(skipna=True))
    if mean_col is not None:
        overall["mean_intensity_column"] = mean_col
        overall["mean_intensity_mean"] = float(df[mean_col].mean(skipna=True))
        overall["mean_intensity_median"] = float(df[mean_col].median(skipna=True))
        overall["mean_intensity_std"] = float(df[mean_col].std(skipna=True))

    overall_path = output_dir / "basic_stats_overall.json"
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    outputs = {"overall_json": str(overall_path)}

    metrics = _present_columns(df, [col for col in [count_col, area_col, mean_col] if col is not None])
    if "source_dataset" in df.columns and metrics:
        by_dataset = (
            df.groupby("source_dataset", dropna=False)[metrics]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        by_dataset.columns = [
            "source_dataset" if col[0] == "source_dataset" else f"{col[0]}_{col[1]}"
            for col in by_dataset.columns.to_flat_index()
        ]
        by_dataset_path = output_dir / "basic_stats_by_source_dataset.csv"
        by_dataset.to_csv(by_dataset_path, index=False)
        outputs["by_source_dataset_csv"] = str(by_dataset_path)

    if "image_source_type" in df.columns and metrics:
        by_source_type = (
            df.groupby("image_source_type", dropna=False)[metrics]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        by_source_type.columns = [
            "image_source_type" if col[0] == "image_source_type" else f"{col[0]}_{col[1]}"
            for col in by_source_type.columns.to_flat_index()
        ]
        by_source_type_path = output_dir / "basic_stats_by_image_source_type.csv"
        by_source_type.to_csv(by_source_type_path, index=False)
        outputs["by_image_source_type_csv"] = str(by_source_type_path)

    return outputs
