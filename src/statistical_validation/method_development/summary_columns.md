Summary CSV columns (simple_dask_image_summary.csv)

- subject name
  - Identifier for the image/sample; derived from the file base name and subfolder label where applicable.

- image_source_type
  - Source of the image pair:
    - combined_channels: a single TIFF containing multiple channels.
    - separate_channels: separate RFP and DAPI TIFF files.

- Count
  - Number of detected crypt regions (objects) in the image.
  - Derived from labeled regions after segmentation.

- Total Area
  - Sum of the areas of all detected crypts in square microns (μm²).
  - Computed from label pixel counts using microns-per-pixel scaling.

- Average Size
  - Mean crypt area in square microns (μm²).
  - Equivalent to Total Area divided by Count when Count > 0.

- % Area
  - Percent of the full image area covered by detected crypts.
  - Calculated as (Total Area / Image Area) × 100, where Image Area = (H×W×μm² per pixel).
  - Note: Manual ImageJ summaries may compute %Area relative to a ROI/thresholded area; ours is relative to the full image. This difference should be considered during comparison.

- Mean
  - Uses integrated fluorescence per crypt instead of per-pixel mean.
  - Defined as rfp_sum_mean × (μm per pixel): average per-crypt intensity sum scaled by spatial resolution.
  - Avoids dividing by area; better suited for cross-MPP comparison here.

- effective_full_intensity_um2_mean
  - Mean “effective full-intensity area” per crypt in μm².
  - Interprets positive intensity as a fraction of a nominal saturation bound and converts to an area-equivalent measure.
  - Useful for comparing images with different spatial resolutions.

- rfp_intensity_um2_sum
  - Total RFP sum standardized to μm²: rfp_sum_total × (μm per pixel)².
  - Provides a cross-resolution comparable aggregate intensity measure.

Notes

- Units: All area quantities are in μm²; no pixel-unit metrics are included in the summary.
- Scale: microns-per-pixel is used internally for calculations and is kept in the per-crypt detailed CSV, not in the summary.
- Detailed per-crypt results are saved to simple_dask_per_crypt.csv, which includes pixel-area fields and per-crypt intensity statistics for deeper analysis.

Detailed image summary (simple_dask_image_summary_detailed.csv)

- subject_name, image_source_type, microns_per_px
- All original detailed metrics in this order:
  - crypt_count
  - crypt_area_um2_sum
  - crypt_area_um2_mean
  - crypt_area_um2_std
  - rfp_sum_total
  - rfp_sum_mean
  - rfp_sum_std
  - rfp_intensity_mean
  - rfp_intensity_std
  - rfp_intensity_min
  - rfp_intensity_max
  - rfp_max_intensity_mean
  - rfp_max_intensity_std
  - effective_full_intensity_um2_sum
  - effective_full_intensity_um2_mean
  - effective_full_intensity_um2_std
