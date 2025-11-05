# Crypt Fluorescence Summary Output

The `summarize_crypt_fluorescence` function in `src/lysozyme_stain_quantification/quantify/crypt_fluorescence_summary.py` produces a compact, ordered set of metrics that can be saved directly to a spreadsheet. The function follows the Analysis Stack run contract (`func(channels, masks, **kwargs) -> array`) and returns a one-dimensional NumPy array. Consumers can call `.tolist()` on the result to obtain a plain Python list.

## Channel Expectations

- `channels[0]`: normalized RFP intensity image (float array).
- `channels[1]`: crypt label image where `0` denotes background and positive integers denote individual crypts.
- `channels[2]`: scalar microns-per-pixel value (float or scalar array).

All arrays must share the same spatial shape.

## Metric Order and Interpretation

The array elements are ordered according to `SUMMARY_FIELD_ORDER`, reproduced here with units and interpretation:

| Field name | Units | Description |
| --- | --- | --- |
| `crypt_count` | count | Number of crypts with at least one labeled pixel. |
| `crypt_area_um2_sum` | µm² | Total crypt area (sum of per-crypt pixel counts × µm² per pixel). |
| `crypt_area_um2_mean` | µm² | Mean area across all crypts. |
| `crypt_area_um2_std` | µm² | Population standard deviation of crypt areas. |
| `rfp_sum_total` | intensity units | Sum of normalized RFP values across all crypt pixels. |
| `rfp_sum_mean` | intensity units | Mean of per-crypt intensity sums. |
| `rfp_sum_std` | intensity units | Population standard deviation of per-crypt intensity sums. |
| `rfp_intensity_mean` | intensity units | Mean of per-crypt mean intensities (average pixel value per crypt). |
| `rfp_intensity_std` | intensity units | Population standard deviation of per-crypt mean intensities. |
| `rfp_intensity_min` | intensity units | Minimum per-crypt mean intensity. |
| `rfp_intensity_max` | intensity units | Maximum per-crypt mean intensity. |
| `rfp_max_intensity_mean` | intensity units | Mean of the maximum pixel intensity observed within each crypt. |
| `rfp_max_intensity_std` | intensity units | Population standard deviation of the per-crypt maximum intensities. |
| `effective_full_intensity_um2_sum` | µm² | Sum of effective “100 % intensity” area (see below). |
| `effective_full_intensity_um2_mean` | µm² | Mean effective “100 % intensity” area per crypt. |
| `effective_full_intensity_um2_std` | µm² | Population standard deviation of effective “100 % intensity” area per crypt. |

All standard deviations use the population definition (`ddof=0`) so that single-crypt observations report `0.0`.

## Effective 100 % Intensity Area

The effective intensity area rescales fluorescence values so they can be interpreted as if the signal were fully saturated across a region:

```
effective_area_um2 = (sum(max(rfp, 0)) / intensity_upper_bound) * (µm² per pixel)
```

- Intensities are clipped at zero so dim or negative-normalized pixels cannot reduce the area.
- `intensity_upper_bound` defaults to the maximum non-negative normalized RFP value in the image, but callers can override it (e.g., set `intensity_upper_bound=256` for 8-bit data).

This quantity highlights how much area would need to glow at the reference intensity to match the measured fluorescence.

## Export Tips

1. Call the analysis function via `AnalysisStackXR` or directly:
   ```python
   summary = summarize_crypt_fluorescence([normalized_rfp, crypt_labels, microns_per_px])
   row = summary.tolist()
   ```
2. Use `SUMMARY_FIELD_ORDER` as column headers when building a dataframe or spreadsheet.
3. If multiple images share the same microns-per-pixel scale, pass it as a scalar array or reuse the `subject_scale_lookup.subject_scale_from_name` helper to populate `channels[2]`.
4. For downstream tuning, compare `effective_full_intensity_um2_*` metrics against the raw area statistics to understand how tightly the ROIs capture high-intensity regions.
