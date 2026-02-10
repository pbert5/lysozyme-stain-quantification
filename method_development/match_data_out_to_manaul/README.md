# Manual vs Auto Matching (Area)

This folder contains a small workflow to compare **manual ImageJ “Total Area”** outputs against the automated pipeline outputs.

## Why the `RFP>threshold` metric exists

The manual ImageJ workflow typically measures **area after an intensity threshold is applied** (i.e., pixels below the threshold are excluded before “Total Area” is computed).

To mimic that behavior on the automated side, the pipeline exports:

- `selected_rfp_px_gt_threshold`: **number of pixels** inside the *selected crypt masks* where the **raw RFP intensity** is `> rfp_gt_threshold` (0–255 scale)

This is intended to be the closest “apples-to-apples” comparison to manual “Total Area” (which is effectively a thresholded pixel count / area).

Related columns (auto CSVs):
- `rfp_gt_threshold`: the intensity cut used (e.g. `71`)
- `selected_rfp_px_gt_threshold`: thresholded pixel-count inside **selected** crypts
- `detected_rfp_px_gt_threshold`: same idea, but inside **detected (pre-selection)** crypts

## What “corrected to 5 crypts” means

The manual evaluations are treated as if they represent **5 crypts per image** (even if the manual file labels say `combined_channels`, we duplicate manual values across both source types for comparison).

The automated pipeline has a per-image **effective count** (`auto_effective_count`, Simpson-style) that can deviate from exactly 5.

So we create “corrected” versions to put auto values onto the same “5 crypts” scale:

`corrected_to_5 = (auto_value / auto_effective_count) * 5`

In the comparison outputs this appears as:
- `auto_rfp_px_gt_threshold_corrected_5`

## Scripts and outputs

### 1) Run the pipeline (adds the needed columns)

Example (threshold `71`):

`./.venv/bin/python src/dask_lysozyme_pipeline.py --rfp-gt-threshold 71`

This produces CSVs under `results/<exp_name>/`, including columns like `selected_rfp_px_gt_threshold`.

### 2) Build the comparison tables + plots

`./.venv/bin/python method_development/match_data_out_to_manaul/compare_manual_vs_auto_area.py`

Key outputs (written to `method_development/match_data_out_to_manaul/out/`):
- `sharable_manual_vs_auto_gated.csv`: **shareable** table that keeps only the gated auto metric(s) (drops flat auto area columns)
- `association_stats.csv`: Spearman (ρ, p) + Brown–Forsythe (Levene median) + Pearson (r, p) for reference
- `distribution_peaks.csv`: rough histogram peak location checks vs `acceptable_peak_px`
- `*.png`: scatter plots and distributions, split by `image_source_type`

## Critical parameters (comparison script)

At the top of `method_development/match_data_out_to_manaul/compare_manual_vs_auto_area.py`:
- `rfp_gt_threshold_cut`: fallback value if auto CSV lacks `rfp_gt_threshold`
- `acceptable_peak_px`: reference value for distribution “peak” sanity checks
- `drop_auto_gt_px`: filters out extreme auto outliers (e.g. drop any auto gated values `> 50k`)

You can override these via CLI flags:
- `--rfp-gt-threshold-cut`
- `--acceptable-peak-px`
- `--drop-auto-gt-px`

