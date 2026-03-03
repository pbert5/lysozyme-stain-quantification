# Dask Pipeline Improvements

## Summary of Changes (October 21, 2025)

### 1. Fixed Import Paths ✅
**Problem**: Imports were failing due to incorrect path structure.

**Solution**: Updated all imports to use `src.*` prefix:
```python
from src.scientific_image_finder.finder import find_subject_image_sets
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from src.dask.cluster import start_local_cluster
```

### 2. Proper Cluster Connection ✅
**Problem**: Script wasn't connecting to existing Dask clusters.

**Solution**: Implemented proper cluster detection using `Client()` with timeout:
```python
try:
    client = Client(timeout='2s')  # Try to connect to default scheduler
    print(f"\n✓ Connected to existing Dask cluster!")
    print(f"  Scheduler: {client.scheduler.address}")
    print(f"  Dashboard: {client.dashboard_link}")
except (OSError, TimeoutError):
    # Start new cluster if none exists
    cluster = LocalCluster(...)
    client = cluster.get_client()
```

**Now shows**:
- ✓ Connected to existing Dask cluster!
- Scheduler: tcp://127.0.0.1:40125
- Dashboard: http://127.0.0.1:8787/status
- Workers: 5

### 3. Parallelized Overlay Rendering ✅
**Problem**: Overlay images were rendered sequentially in a loop.

**Solution**: Wrapped rendering in `@delayed` decorator:
```python
@delayed
def _render_and_save_one(subject_name, rfp, dapi, labels):
    overlay_xr = render_label_overlay(channels=[rfp, dapi, labels])
    # ... save overlay ...
    return output_path

# Build delayed tasks (parallelizable!)
delayed_saves = [
    _render_and_save_one(name, rfp, dapi, labels)
    for name, rfp, dapi, labels in zip(...)
]

# Compute all in parallel
output_paths = dask.compute(*delayed_saves)
```

### 4. Proper Overlay Rendering ✅
**Problem**: Simple matplotlib rendering didn't match original pipeline quality.

**Solution**: Integrated `render_label_overlay` from image-ops-framework:
```python
from image_ops_framework.helpers.overlays import render_label_overlay

overlay_xr = render_label_overlay(
    channels=[rfp, dapi, labels],
    fill_alpha=0.35,
    outline_alpha=1.0,
    outline_width=2,
    normalize_scalar=True,
)
```

**Features**:
- Proper RGB blending of RFP (red) and DAPI (blue) channels
- Semi-transparent colored fills for labeled regions
- Bold outlines around crypt boundaries
- Deterministic pseudo-random colors per label
- Automatic normalization of scalar channels

### 5. Clear Dashboard Reporting ✅
**Problem**: Dashboard URL wasn't clearly reported.

**Solution**: Prominent dashboard URL display:
```
✓ Connected to existing Dask cluster!
  Scheduler: tcp://127.0.0.1:40125
  Dashboard: http://127.0.0.1:8787/status
  Workers: 5
```

When starting a new cluster:
```
  ⚠️  OPEN DASHBOARD: http://127.0.0.1:8787/status
```

## Usage Examples

### Connect to Existing Cluster
```bash
# Terminal 1: Start cluster
python src/dask/cluster.py --n-workers 8

# Terminal 2: Run pipeline (will auto-connect)
source .venv/bin/activate
python karen_detect_crypts_dask.py --use-cluster --debug
```

### Standalone Execution
```bash
# Without cluster (threaded scheduler)
python karen_detect_crypts_dask.py --max-subjects 10

# With new cluster (if none exists)
python karen_detect_crypts_dask.py --use-cluster --n-workers 4
```

## Performance Benefits

1. **Parallelized Overlay Rendering**: All 10 overlay images render simultaneously instead of sequentially
2. **Distributed Execution**: When cluster is available, all subjects process in parallel across workers
3. **Lazy Graph Building**: Still maintains single-compute-call architecture for optimal scheduling

## Output Quality

Overlay images now match original pipeline quality:
- ✅ Proper RGB channel blending
- ✅ Colored label overlays with transparency
- ✅ Bold crypt boundary outlines
- ✅ Normalized intensity scaling

## Dashboard Monitoring

Open dashboard URL to monitor:
- Task progress in real-time
- Worker utilization
- Memory usage per worker
- Task stream visualization
- Graph structure

Example: http://127.0.0.1:8787/status
