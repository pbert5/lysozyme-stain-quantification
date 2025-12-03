input: seperate RFP and DAPI flourecent image pairs ( reffered to as red_img and blue_img)

pipeline:
    (red_image, blue_img) -> 
    grayscale ->  
    extractor_run: ->
        red_img & blue_img -> %%it looks like i dont normalize them%% 
            # in general, Dapi has overal high signals for anywhere that has tissue
            ## and RFP has lower signals overall for tissue, with slightly higher signals for the crypts


        diff_r = bool where red stronger than min envelope: ->
            diff_r = red > blue # aka mask of crypts are

            clean up diff_r: ->
                binary erosion with a 3x3 square kernel ->
                remove small objects less then 100 px area ->
            # 
        
        Secondary mask using absolute difference = mask of non crypt tissue -> 
            mask_gt_red = blue > 2*red = non-crypt-tissue
            mask_gt_red_eroded = mask_gt_red is realy speckeled by noise, so this fixes it

        label handeling: ->
            Combined Labels = (0 bg, 1 diff_r, 2 mask_gt_red_eroded) aka ( 0 bg, 1 cypts, 2 non crypt tissue)
            then the diff_r & mask_gt_red_eroded get expanded by 100px to meet/ fill up bg between them->
            use ndi_label to seperatly label disconnected regions of diff_r-> ( the non expanded cypt mask)

        set up reworked labels ( which are used as the watershed markers) -> 
            1 =expanded_labels[2] =  mask_gt_red_eroded = non-crypt-tissue
            mask_copy = select pixels that are NOT in expanded class 2 AND that belong to a connected component from diff_r.
                i.e., crypt pixels that lie outside the class-2 region (or in class-1/background area).
            reworked[mask_copy] = copy the connected-component ids into the markers image, shifting by +1 so they don't collide with the marker 1 used for class 2.
                result: each diff_r component gets its own marker label (2,3,4,...).
            %%this basicaly allows us to get much tigher and precise marks for each of the crypts so that we start with much more confdent positions, and prevent the overmergeing seen in raw expanded labels, while still having the very intact non crypt tissue regions of expanded labels %%

        then we get elevation: -> 
            this is a distance transform on combined lables, where non crypt tissue mountains, and crypt tissue form valies
         
        then using the reworked lables as markers and combined label derived elevation we run watershed
            getting our likely crypt candidates

    scoring:
        uses a set of self descripteve weights:
            self.weights = weights if weights is not None else {
                'circularity': 0.35,    # Most important - want circular regions
                'area': 0.25,           # Second - want consistent sizes
                'line_fit': 0.15,       # Moderate - want aligned regions
                'red_intensity': 0.15,  # Moderate - want bright regions
                'com_consistency': 0.10 # Least - center consistency
            }
        line fit reffers to proximity to a line through the center of mass of all the detected crypt region, this is an approximation of "crypts are genraly arrayed in a straightish line along the gut wall
        by default choses the 5 best
        
        get: (background_tissue_intensity, average_crypt_intensity)

ok, so theres the pipeline, but how does it work:

    starting with the RFP(red) and DAPI(Blue) images, we first find the 




    note: cant take combo straigt from imager as it preforms some sort of preproc that significantly changes the images, likely they are normalized on combined scales instead of seperate scales

## Cross-Image Normalization Solution

Since the contrast between crypt and non-crypt staining can vary significantly between images due to differences in staining efficiency, imaging conditions, and tissue autofluorescence, we implement a sophisticated normalization approach to enable meaningful cross-image comparison.

### Normalization Methodology

**Background Tissue Intensity Calculation:**
```
background_tissue_intensity = mean(red_channel / blue_channel) for pixels where ws_labels == 1
```
- Uses watershed label 1 (background tissue) as reference
- Represents baseline red-to-blue intensity ratio for each image
- Accounts for imaging conditions and tissue autofluorescence

**Average Crypt Intensity Calculation:**
```
average_crypt_intensity = mean(red_channel / blue_channel) for pixels where ws_labels > 1
```
- Uses all crypt regions (watershed labels > 1) 
- Represents overall crypt staining intensity relative to DAPI
- Captures the average lysozyme expression level across all crypts in the image

**Contrast Ratio:**
```
contrast_ratio = average_crypt_intensity / background_tissue_intensity
```
- Quantifies per-image contrast between crypts and background tissue
- Higher values indicate stronger crypt-specific staining relative to background

**Normalized Fluorescence:**
```
fluorescence_normalized = fluorescence × background_tissue_intensity / average_crypt_intensity
                        = fluorescence / contrast_ratio
```

### Technical Implementation

The normalization is calculated during the watershed segmentation phase, before region 1 (background tissue) is removed from the label map. This ensures we capture the true background tissue signal for each image.

### Biological Rationale

This approach effectively sets a standardized gain factor that:
1. **Compensates for staining variability** - Different batches of antibodies or staining protocols
2. **Normalizes imaging conditions** - Variations in exposure time, lamp intensity, or camera sensitivity  
3. **Accounts for tissue differences** - Natural variation in tissue autofluorescence
4. **Preserves biological signal** - Maintains relative differences in lysozyme expression while enabling cross-image comparison

The result is a normalized fluorescence metric that represents lysozyme expression strength relative to each image's baseline characteristics, enabling robust statistical analysis across experimental conditions.



well written:
Paired RFP (red channel) and DAPI (blue channel) fluorescence images were used as inputs for crypt segmentation. The DAPI signal generally marked all tissue regions, whereas the RFP signal was comparatively weaker overall but exhibited locally stronger intensity within crypt regions. To ensure consistent interpretation across samples, analyses were performed on the unnormalized individual channel images, as normalization by the imaging software appeared to artificially alter relative intensities. 

# Preprocessing and Binary Mask Generation:
Two complementary binary masks were generated to distinguish crypt from non-crypt tissue:
1. crypt regions were identified by selecting areas where RFP signal exceded DAPI (red > blue). The resulting mask was cleaned by applying binary erosion with a 3×3 square kernel and removing objects smaller than 100 pixels.
2. Non-crypt tissue was defined as areas where DAPI signla exceeded twize the RFP signal ( blue > 2 * red). This mask underwent morphological erosion to improve spatial continuity by removing speckle noise

# Label construciton and expansion:
The two masks were combined into a priliminary lable map: background(0), crypts (1), and non-crypt tissue (2). Then to smooth out the non-crypt tissue without overtakingcypts, both crypt and non-crypt labels were dilated by 100 pixels
Disconnected crypt components from the non expanded mask were then independently labeled using connected-component analysis. These specificaly come from the non-expanded map, as their purpose is not to mark the entierety of each crypt, but to represent a precice marking( phisicaly seperated) for each individual crypt to prevent merging. Forming a labled crypt map.

To improve watershed marker quality, a reworked label map was created. Non-crypt tissue (class 2 now 1) was preserved as a single intact marker, while crypt components outside this region were derived from the labeld crypt map and assigned unique labels (2, 3, 4, …). This strategy prevented over-merging of adjacent crypts and provided precise initial markers for segmentation while including the more limiting expanded tissue layer.

# Watershed Segmentation

A distance transform of the combined label map was computed, producing an elevation image where non-crypt regions formed ridges and crypt regions formed valleys. The reworked labels were then used as watershed markers on this elevation image, yielding candidate crypt segments. This allowed for the crypt labels to be expanded to more accurately encompase the entirety of of each crypt while remaining seperate labels


# Scoring and Candidate Selection

Segmented regions were ranked according to a weighted scoring system prioritizing morphological and intensity features. Features and weights were: circularity (0.35), area consistency (0.25), linear alignment along the gut wall (0.15), average RFP intensity (0.15), and center-of-mass consistency (0.10). The top five scoring regions were selected as putative crypts. Additionally, average background tissue intensity and crypt-specific RFP intensity were recorded for downstream normalization.

# Intensity Normalization Across Images

To account for slide-to-slide variability in staining and imaging conditions, RFP signal was standardized relative to DAPI. Specifically, the ratio of RFP to DAPI intensities was computed separately for crypt and non-crypt tissue regions, and the ratio between these two values was used to apply a gain-like scaling factor. This ensured that crypt RFP intensities were comparable across images, independent of global signal variation.




math notes: 
flourescense = red_sum_pixels/area_pixels*area_um2

## Output Values and Units

- Image-level outputs live in `results/simple_dask/simple_dask_image_summary.csv` (manual-style columns) and `simple_dask_image_summary_detailed.csv` (all metrics). `Count` is the number of crypt labels; `Total Area`/`Average Size` are reported in µm² (converted from pixel area with microns-per-pixel); `% Area` is relative to the full image; `Mean` is the normalized per-crypt integrated intensity scaled by microns-per-pixel.
- The detailed file also stores the raw, non-normalized sums with a `raw_` prefix (e.g., `raw_rfp_sum_mean`, `raw_rfp_intensity_mean`) that are direct sums/means of the original RFP pixel values.
- Manual ImageJ tables in `src/statistical_validation/manual_stats_results` (`image_j_quantification_eva_04222025(Adam).csv` and `...Eva).csv`) use 8-bit pixel intensities: `Total Area` is pixel count, and `Mean` is the average raw pixel intensity. Integrated intensity in the R scripts is `Total Area × Mean`, optionally scaled by (µm/px)² to match area units.

## Why Automated Numbers Look Different From ImageJ

- Automated normalization uses the ratio `red/blue`, subtracts the average background-tissue ratio, and divides by the average crypt ratio:
  ```
  normalized = (red/blue - background_tissue_intensity) / average_crypt_intensity
  ```
  Background pixels are driven toward 0 and the average crypt signal is ~1, so the dynamic range is compressed and can include negatives where red < blue. In contrast, ImageJ’s `Mean` column is an 8-bit raw intensity (0–255) with no background centering.
- `rfp_sum_*` in the detailed summary are sums of these normalized values, so they are orders of magnitude smaller than the raw ImageJ sums. To compare on the same scale, use the `raw_` fields (or the new `auto_rfp_sum_mean_raw` in the consolidated validation CSV) and, if needed, multiply by (µm/px)² for area-normalized units.

## Statistical Validation Workflow

- `src/statistical_validation/final_build_consolidated.R` ingests the manual ImageJ tables, resolves microns-per-pixel from slice names (0.4476 default, 0.2253 for “40x”), computes per-crypt integrated fluorescence, and joins to automated outputs. The consolidated file now includes both `auto_rfp_sum_mean` (normalized) and `auto_rfp_sum_mean_raw` (direct raw sums).
- `src/statistical_validation/run_final_correlation.R` now runs correlations for both metrics:
  - Normalized: `src/statistical_validation/outputs/final/correlation_by_set_originals_retakes.csv` (Retake r=0.15, originals r=0.12, Eva subset r=0.82).
  - Raw (non-normalized): `src/statistical_validation/outputs/final/correlation_by_set_originals_retakes_raw.csv` (Retake r=0.95, originals r=0.47, Eva subset r=0.85).
  - Scatter plots for each are written to `src/statistical_validation/outputs/final/plots/` with `scatter_correlation_*.png` (normalized) and `scatter_correlation_raw_*.png`.
- The raw comparisons align with the ImageJ tables because they use the same underlying pixel values instead of the background-centered red/blue ratios. Use the raw columns when you want magnitude comparability with the manual measurements and the normalized columns when you want cross-slide contrast invariance.
