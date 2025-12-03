# Eva Set External Validation Dataset

**File:** `eva_set_external_validation.csv`

## Purpose
This dataset contains manual fluorescence measurements from two independent reviewers (Adam and Eva) for the Eva image set. It's designed for external statistical validation and is formatted for easy interpretation.

## Dataset Summary
- **Total records:** 256 manual crypt measurements
- **Reviewers:** Adam (135 measurements), Eva (121 measurements)
- **Unique mice:** 15
- **Source:** Manual ImageJ quantification by two independent raters

## Column Descriptions

### Identifiers
- **manual_reviewer**: Who performed the manual rating (Adam or Eva)
- **mouse_id**: Mouse identifier (e.g., G2ER, G3FB, G5EL)
- **slice_name**: Name of the tissue slice/image analyzed
- **subject_key**: Unique key for matching with automated analysis results

### Crypt Measurements
- **crypt_count**: Number of crypts manually identified in the region

### Pixel-Based Fluorescence Measurements
- **total_area_px**: Total area of all crypts in pixels
- **avg_crypt_size_px**: Average size of crypts in pixels
- **mean_fluorescence_px**: Mean fluorescence intensity per pixel
- **integrated_fluorescence_px**: Total integrated fluorescence in pixels (sum of all intensities)

### Micrometer-Based Measurements
- **um_per_px**: Conversion factor from pixels to micrometers
- **integrated_fluorescence_um2**: Total integrated fluorescence in square micrometers (µm²)

### Other Measurements
- **percent_tissue_area**: Percentage of tissue area covered by crypts

## Units
- **_px**: Measurements in pixels
- **_um2**: Measurements in square micrometers (µm²)
- **Fluorescence values**: Arbitrary units from microscopy imaging

## Notes
- Each row represents manual measurements from a single region of interest (ROI/crypt)
- The same slice may have multiple measurements (different regions)
- Some slices have multiple subject_keys due to different automated processing runs (separate_channels vs combined_channels)
- Missing subject_keys indicate slices that were manually rated but not included in automated analysis
- Both reviewers independently assessed the same image set for inter-rater reliability validation

## Usage
This dataset is intended for:
1. Validating automated fluorescence quantification methods
2. Assessing inter-rater reliability between Adam and Eva
3. Correlating manual and automated measurements
4. External statistical analysis and peer review

## Data Provenance
Generated from:
- Source files: `manual_stats_results/image_j_quantification_eva_04222025(Adam).csv` and `image_j_quantification_eva_04222025(Eva).csv`
- Processing: `outputs/final/manual_uncollapsed_with_metrics.csv`
- Mapping: `outputs/final/manual_to_auto_mapping.csv`
- Script: `create_eva_external_validation_dataset.py`
