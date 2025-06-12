# Lysozyme Stain Quantification Pipeline

This document provides an overview of the project with a focus on the blob detection pipeline used to identify RFP lysozyme stained crypts in intestinal sections. It is aimed at researchers with a biology background who may be less familiar with image processing concepts.

## Project Goal

The original goal of this codebase is to quantify intestinal crypts that are positive for an RFP lysozyme stain. The images come from fluorescence microscopy and typically contain a red channel where positive crypts appear bright. The pipeline cleans the raw images, segments putative crypts, merges fragmented regions, and exports the results for further analysis.

## High Level Workflow

1. **Load the microscopy image** – usually a TIFF file.
2. **Preprocess** – remove a scale bar if present and enhance contrast.
3. **Create a positive mask** – use red chromaticity and thresholding to isolate regions with strong RFP signal.
4. **Initial segmentation** – watershed segmentation on a contrast-enhanced grayscale channel to produce labeled regions.
5. **Merge step** – refine the labels with a two‑stage algorithm so that single crypts are represented by single labels.
6. **Export** – save labels as NumPy arrays and export to GeoJSON for tools such as QuPath.

The following sections describe the main components in more detail.

## Image Handling Utilities

`src/img/img_handeler.py` contains helper classes for operations such as:

- **Scale bar removal** – automatically detects and fills in a scale bar region using interpolation.
- **Threshold utilities** – for example `chromaticity` computes per-channel chromaticity to emphasize the RFP channel over others.
- **Contrast enhancement** – functions like CLAHE (adaptive histogram equalization) help highlight features before segmentation.
- **Mask creation** – Otsu thresholding followed by morphological cleanup produces a binary mask of candidate regions.
- **Watershed segmentation** – converts the grayscale image into labeled regions (blobs) based on intensity minima and a user-provided low and high threshold.

## Blob Detection (`src/blob_det.py`)

`BlobDetector` wraps the above utilities into a convenient pipeline. Typical usage is:

```python
image = tifffile.imread('sample.tif')
detector = BlobDetector(channel=0, debug=True)
labels = detector.detect(image, segmentation_low_thresh=10, segmentation_high_thresh=150)
```

The `detect` method performs the following steps:

1. **Remove scale bar** using `ImgHandler.InconvenientObjectRemover`.
2. **Create a positive mask** from red chromaticity – only strongly red pixels are kept.
3. **Watershed segmentation** on a CLAHE‑enhanced grayscale channel. This produces `expanded_labels`.
4. **Two-stage merge** via `LabelHandeler.MergePipeline` to combine fragmented pieces of the same crypt. The result is stored as `swallowed_labels`.
5. **Return final labels** for downstream processing.

Debug mode optionally saves intermediate arrays for inspection.

## Merge Algorithm (`src/np_labels/label_handeler.py`)

Microscopy data often contains fragmented or touching regions after simple watershed segmentation. The `MergePipeline` class addresses this by evaluating possible merges in two passes.

### Stage 1 – Candidate Groups

1. **Compute statistics** for each label: area, perimeter, centroid, etc.
2. **Shared perimeter graph** – labels are considered neighbors if they share a border. Shared lengths are recorded.
3. **Triangles detection** – groups of three mutually adjacent labels are noted because they often form a natural unit.
4. **Candidate generation** – for each label, possible groups include:
   - The label alone (singleton).
   - The label paired with any neighbor.
   - Triangles that include the label.
   - Pairs of triangles that share two labels (larger groups).
5. **Scoring** – each candidate group is scored by how much border is shared relative to its total perimeter. A `singleton_penalty` discourages merging when very little boundary is shared.
6. **Best group** – for each label, the candidate with the highest score is kept for stage 2.

### Stage 2 – Choosing Leaders

1. **Normalize groups** so equivalent combinations are mapped together.
2. **Group leaders** – within each candidate group, the label with the largest area is selected as the representative.
3. **Evaluate each member** – compute a compactness-like score using total area, total perimeter, and distance from each member to the group centroid.
4. **Relabel** – every label is reassigned to its group leader if that improves the score. True singletons remain unchanged.

The final result is a label image where fragmented pieces that likely belong to the same crypt are merged into a single label.

## Bulk Processing (`src/primary.py`)

`BulkBlobProcessor` demonstrates processing many images at once. It loads images, runs `BlobDetector`, saves NumPy arrays and GeoJSON files, and can produce quick overlay images for inspection.

## Why Each Step Matters

1. **Scale bar removal** – prevents artifacts from being interpreted as crypts.
2. **Positive mask** – focuses analysis on the RFP stained regions, reducing noise from the background tissue.
3. **Watershed segmentation** – provides an initial guess of crypt boundaries but often over‑segments.
4. **Merge algorithm** – crucial for combining fragmented pieces so each crypt corresponds to a single ROI. This ensures accurate quantification of stained crypts.
5. **Export** – GeoJSON output allows visualization and further measurement in tools like QuPath or downstream scripts.

## Conclusion

This pipeline automates detection of RFP lysozyme positive crypts in intestinal samples. By combining classical image preprocessing, watershed segmentation, and a custom two‑stage merge strategy, the code outputs clean label maps ready for quantification or further biological analysis.
