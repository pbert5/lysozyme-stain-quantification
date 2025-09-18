# Terms and Conditions: Data Output Definitions

This document provides comprehensive definitions for all metrics and terminology used in the lysozyme stain quantification pipeline output data.

## Core Fluorescence Measurements

### `red_intensity`
**Definition**: Average red channel pixel intensity within each detected crypt region.

**Calculation**: `red_intensity = prop.mean_intensity` where `prop` is from scikit-image's `regionprops`

**Technical Details**:
- Extracted from the raw red channel (RFP) fluorescence microscopy image
- Calculated as the mean pixel intensity value across all pixels within the segmented crypt region
- Units: Pixel intensity values (typically 0-255 for 8-bit images, 0-65535 for 16-bit)
- Represents the average brightness/intensity of lysozyme staining within each crypt

### `red_sum`
**Definition**: Total accumulated red channel intensity for each crypt region.

**Calculation**: `red_sum = red_intensity × area_pixels`

**Technical Details**:
- Sum of all pixel intensity values within the crypt region
- Units: Total intensity units (intensity × pixels)
- Used as an intermediate calculation for other metrics

### `fluorescence`
**Definition**: Area-normalized fluorescence measurement representing the total lysozyme expression in each crypt.

**Calculation**: `fluorescence = red_intensity × area_um2`

**Alternative formulation**: `fluorescence = red_sum_pixels / area_pixels × area_um2`

**Technical Details**:
- Combines intensity information with physical area measurement
- Units: Intensity × square micrometers
- Primary metric for quantifying lysozyme expression per crypt
- Accounts for both staining intensity and crypt size

## Normalization Metrics

### `background_tissue_intensity`
**Definition**: Average red-to-blue channel intensity ratio in background tissue regions.

**Calculation**: 
```
background_mask = (ws_labels == 1)  # Label 1 = background tissue
red_bg = red_img[background_mask]
blue_bg = blue_img[background_mask]
background_tissue_intensity = mean(red_bg / blue_bg)  # where blue_bg > threshold
```

**Technical Details**:
- Calculated from watershed segmentation label 1 (background tissue)
- Represents baseline tissue autofluorescence and imaging conditions
- Used for cross-image normalization
- Units: Ratio (dimensionless)

### `average_crypt_intensity`
**Definition**: Average red-to-blue channel intensity ratio across all crypt regions.

**Calculation**:
```
crypt_mask = (ws_labels > 1)  # Labels > 1 = crypt regions
red_crypt = red_img[crypt_mask]
blue_crypt = blue_img[crypt_mask]
average_crypt_intensity = mean(red_crypt / blue_crypt)  # where blue_crypt > threshold
```

**Technical Details**:
- Calculated from all watershed segmentation labels except 0 (background) and 1 (tissue background)
- Represents overall crypt staining intensity relative to DAPI signal
- Used for cross-image normalization
- Units: Ratio (dimensionless)

## Spatial Measurements

### `pos_x_um`, `pos_y_um`
**Definition**: Physical coordinates of crypt centroid in micrometers.

**Calculation**: 
```
pos_y, pos_x = prop.centroid  # Note: centroid returns (row, col)
pos_x_um = pos_x × pixel_dimension
pos_y_um = pos_y × pixel_dimension
```

**Technical Details**:
- Converted from pixel coordinates to physical units using calibrated pixel dimensions
- Origin typically at top-left corner of image
- Units: Micrometers (μm)

### `area_um2`
**Definition**: Physical area of each crypt region in square micrometers.

**Calculation**: `area_um2 = area_pixels × (pixel_dimension)²`

**Technical Details**:
- Converted from pixel count to physical area using calibrated pixel dimensions
- Units: Square micrometers (μm²)

## Normalization Solution

### Contrast Ratio
**Definition**: Per-image contrast metric comparing crypt staining to background tissue.

**Calculation**: `contrast_ratio = average_crypt_intensity / background_tissue_intensity`

**Purpose**: Quantifies the relative staining contrast within each image to enable cross-image comparison.

### Normalized Fluorescence
**Definition**: Fluorescence measurement normalized for cross-image comparison.

**Calculation**: 
```
fluorescence_norm = fluorescence × background_tissue_intensity / average_crypt_intensity
                  = fluorescence / contrast_ratio
```

**Technical Details**:
- Compensates for differences in imaging conditions, staining efficiency, and tissue autofluorescence
- Enables meaningful comparison of lysozyme expression levels between different images
- Higher values indicate stronger lysozyme expression relative to the image's baseline

## Image Classification Metadata

### `image_name`
**Definition**: Cleaned and standardized image identifier derived from the source filename.

### `is_retake`
**Definition**: Boolean flag indicating whether the image is a retake/repeat acquisition.

### `subdir`
**Definition**: Source subdirectory classification for experimental grouping.

### `red_file`, `blue_file`
**Definition**: Original filenames for the red (RFP) and blue (DAPI) channel images.

### `red_path`
**Definition**: Full filesystem path to the source red channel image file.

## Quality Control Metrics

Images and regions are filtered based on:
- Minimum and maximum area thresholds
- Intensity thresholds for positive staining
- Circularity and morphological criteria
- Outlier detection based on statistical distributions

## Units Summary

| Metric | Units | Range |
|--------|--------|--------|
| `red_intensity` | Pixel intensity | 0-255 (8-bit) or 0-65535 (16-bit) |
| `red_sum` | Total intensity | Varies with region size |
| `fluorescence` | Intensity × μm² | Varies with intensity and area |
| `background_tissue_intensity` | Ratio | Typically 0.1-2.0 |
| `average_crypt_intensity` | Ratio | Typically 0.5-5.0 |
| `pos_x_um`, `pos_y_um` | μm | Image-dependent |
| `area_um2` | μm² | Typically 100-10000 |
