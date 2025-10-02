# Lysozyme Stain Quantification Pipeline

A complete Python pipeline for detecting and quantifying lysozyme stains in microscopy images using watershed segmentation and intelligent region merging.

## Project Structure

```
src/
├── run.py                          # Main entry point
├── pipeline/
│   ├── __init__.py
│   └── bulk_processor.py          # Handles multiple image pairs
├── processing/
│   ├── __init__.py
│   ├── extractor_pipeline.py      # Blob detection using watershed
│   ├── merge_pipeline.py          # Intelligent region merging
│   └── individual_processor.py    # Single image pair processing
└── utils/
    ├── __init__.py
    ├── file_utils.py              # File I/O and image loading
    └── image_utils.py             # Image processing utilities
```

## Installation

1. Make sure you have the virtual environment activated:
   ```bash
   source "C:/Users/admin/Documents/Pierre lab/projects/Colustrum-ABX/lysozyme stain quantification/.venv/Scripts/activate"
   ```

2. All required packages should already be installed. If not:
   ```bash
   pip install numpy scikit-image tifffile matplotlib scipy opencv-python pandas
   ```

## Usage

### Basic Usage

```bash
python src/run.py <img_dir> <results_dir>
```

### With Options

```bash
python src/run.py "lysozyme images/Jej LYZ/G2" "results" \
    --red_channel RFP \
    --blue_channel DAPI \
    --debug
```

### Parameters

- `img_dir`: Directory containing TIFF images (searches subdirectories)
- `results_dir`: Output directory for results
- `--red_channel`: String to identify red channel files (default: "RFP")
- `--blue_channel`: String to identify blue channel files (default: "DAPI")
- `--pixel_dims`: JSON string of pixel dimensions or use defaults
- `--debug`: Enable debug output with visualizations

### Default Pixel Dimensions

The pipeline uses these default pixel dimensions (in micrometers):
- `default`: 0.4476 μm/pixel
- `40x-4`: 0.2253 μm/pixel

Custom pixel dimensions can be provided as JSON:
```bash
python src/run.py img_dir results --pixel_dims '{"default": 0.4476, "high-res": 0.1}'
```

## Pipeline Workflow

1. **File Discovery**: Finds paired red/blue channel TIFF files
2. **Individual Processing**: For each pair:
   - Loads and preprocesses images
   - Runs watershed-based blob detection
   - Applies intelligent region merging
   - Generates summary statistics
3. **Bulk Processing**: 
   - Consolidates results from all pairs
   - Saves summary CSV files
   - Generates debug visualizations (if enabled)

## Output Files

### Standard Output

- `summaries/consolidated_summary.csv`: Per-region data for all images
- `summaries/by_image_summary.csv`: Aggregated statistics by image

### Debug Output (when --debug is used)

- `debug/individual/`: Individual debug images for each processing step
- `debug/merged/`: Final merged region overlays
- `debug/quick_check/`: Consolidated overview visualization

### Summary Data Columns

Each detected region includes:
- `id`: Region identifier
- `pos_x_um`, `pos_y_um`: Position in micrometers
- `area_um2`: Area in square micrometers
- `red_sum`: Total red channel intensity
- `red_intensity`: Average red channel intensity
- `image_name`: Source image identifier
- `red_file`, `blue_file`: Source filenames

## Testing

Run the test suite to verify installation:

```bash
cd code
python test_pipeline.py
```

## Algorithm Details

### Blob Detection (ExtractorPipeline)

1. **Morphological Operations**: Identifies regions where red channel is stronger than blue
2. **Label Expansion**: Expands detected regions to capture full extent
3. **Watershed Segmentation**: Uses distance transforms to separate touching objects
4. **Boundary Refinement**: Finds precise object boundaries

### Region Merging (MergePipeline)

1. **Adjacency Analysis**: Calculates shared perimeters between regions
2. **Triangle Detection**: Identifies three-way adjacency relationships
3. **Stage 1 Grouping**: Groups regions based on shared perimeter ratios
4. **Stage 2 Optimization**: Refines groups using area/perimeter and distance metrics
5. **Final Relabeling**: Creates merged label array

## Example Results

After processing, you'll get:
- Quantitative measurements for each detected lysozyme stain
- Spatial coordinates and morphological parameters
- Summary statistics aggregated by image
- Visual overlays showing detected regions (in debug mode)

## Troubleshooting

1. **Import Errors**: Run from the `src` directory or ensure Python path is set correctly
2. **No Image Pairs Found**: Check that filenames follow the pattern `*_RFP.*` and `*_DAPI.*`
3. **Memory Issues**: Process smaller batches or disable debug mode for large datasets
4. **Processing Failures**: Check image file integrity and matching dimensions

## Performance Notes

- Processing time scales with image size and number of detected objects
- Debug mode generates additional files and uses more memory
- The pipeline is designed for robustness over speed
- For large datasets, consider processing in batches
