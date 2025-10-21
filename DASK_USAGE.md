# Dask Lysozyme Pipeline Usage Guide

## Overview

`karen_detect_crypts_dask.py` is a pure Dask implementation of the lysozyme crypt detection pipeline. It builds a complete lazy computation graph before executing, allowing for optimal parallelization and memory usage.

## Key Features

- **Pure Lazy Evaluation**: Builds entire computation graph before execution
- **No Eager Computation**: Unlike the xarray-based approach, nothing is computed until explicitly requested
- **Optional Cluster Support**: Can run on local threads or distributed Dask cluster
- **Identical Output**: Produces same CSV and PNG outputs as original pipeline
- **Clean Architecture**: No framework overhead, just Dask arrays and delayed functions

## Basic Usage

### Run with default settings (threaded scheduler, 10 subjects, with images)
```bash
python karen_detect_crypts_dask.py
```

### Run with debug output
```bash
python karen_detect_crypts_dask.py --debug
```

### Process limited number of subjects
```bash
python karen_detect_crypts_dask.py --max-subjects 5
```

### Skip image generation (faster)
```bash
python karen_detect_crypts_dask.py --no-images
```

## Cluster Support

### Use existing or spawn local cluster
```bash
python karen_detect_crypts_dask.py --use-cluster --n-workers 8
```

### Connect to external cluster (if one is running)
The script will automatically detect and use an existing Dask cluster via `Client.current()`.

### Using the cluster module directly
```bash
# Start a standalone cluster first
python src/dask/cluster.py --n-workers 8 --dashboard-port 8787

# Then run the pipeline (will connect automatically)
python karen_detect_crypts_dask.py --use-cluster
```

## Output Files

All files are saved to `results/karen_dask/`:

1. **karen_detect_crypts_dask.csv** - Image-level summary (one row per subject)
   - Fields: subject_name, crypt_count, crypt_area_um2_sum, rfp_sum_total, etc.
   
2. **karen_detect_crypts_dask_per_crypt.csv** - Per-crypt details (one row per detected crypt)
   - Fields: subject_name, crypt_label, crypt_index, pixel_area, um_area, etc.

3. **karen_detect_crypts_dask.png** - Grid visualization of all detected crypts

4. **renderings/** - Individual overlay images for each subject
   - 3-panel view: RFP channel, DAPI channel, detected crypts

## Command-Line Arguments

```
--use-cluster         Use Dask distributed cluster (default: False)
--n-workers N         Number of workers for local cluster (default: 4)
--no-images          Skip generating overlay images and visualizations
--debug              Enable detailed debug output
--max-subjects N     Maximum number of subjects to process (default: 10)
```

## Performance Comparison

### Threaded Scheduler (default)
- Uses Python threading for parallelism
- Good for I/O-bound tasks
- Lower memory overhead
- Simpler setup (no cluster needed)

### Distributed Scheduler (--use-cluster)
- Uses separate processes for parallelism
- Better for CPU-bound tasks
- Can scale to multiple machines
- Provides dashboard for monitoring (http://localhost:8787)
- Requires `dask[distributed]` package

## How It Works

1. **Graph Building Phase** (lazy)
   - Loads image paths (not actual images)
   - Creates delayed wrappers for each processing step
   - Connects all operations in a DAG
   - No computation occurs yet!

2. **Compute Phase** (single call)
   - `dask.compute(*all_delayed_objects)` 
   - Dask optimizes the graph
   - Executes in parallel using chosen scheduler
   - Returns all results at once

3. **Output Phase**
   - Converts numpy arrays to pandas DataFrames
   - Saves CSVs with proper field ordering
   - Generates visualizations (if enabled)

## Architecture Differences from Original

### Original (karen_detect_crypts.py)
- Uses xarray.apply_ufunc framework
- Forces eager computation when inspecting dask arrays
- Framework overhead and complexity
- Fine-grained delays throughout codebase

### Dask Version (karen_detect_crypts_dask.py)
- Pure Dask implementation
- True lazy evaluation until `compute()`
- No framework dependencies (except Dask itself)
- High-level delayed wrappers only
- Simpler and faster

## Troubleshooting

### "Cluster support not available"
Install distributed scheduler support:
```bash
pip install "dask[distributed]"
```

### Out of memory errors
- Use `--no-images` to skip visualization
- Reduce `--n-workers` count
- Process fewer subjects with `--max-subjects`

### Different results from original
This should not happen! The computation is identical, only the execution strategy differs. If you see differences, please report as a bug.

## Future Enhancements

Possible improvements:
- [ ] Add support for custom output directories
- [ ] Add support for different blob sizes per subject
- [ ] Add resume capability (skip already processed subjects)
- [ ] Add cloud cluster support (AWS, GCP, Azure)
- [ ] Add progress bars with tqdm
- [ ] Add profiling output for performance analysis
