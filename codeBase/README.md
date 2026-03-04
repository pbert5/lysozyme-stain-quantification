# CodeBase Pipeline Guide

This README documents the current end-to-end run flow implemented in `codeBase/`.

## Pipeline Modules

1. Image discovery and pairing
- Module: `codeBase/image_utils/discover_lysozyme_images.py`
- Purpose: discover lysozyme/tissue image pairs and write CSV input rows.
- Output CSV columns: `subject_id`, `lysozyme_path`, `tissue_path`, `microns_per_pixel`, `source_dataset`, `source_label`, `notes`.

2. Spark local/cluster setup
- Module: `codeBase/pipeline_implementations/spark_implementation/cluster.py`
- Purpose: create/configure `SparkSession` (local master by default, configurable via YAML).

3. Single-subject analysis endpoint
- Module: `codeBase/crypt_detection_code/lysozyme_stain_quantification/single_subject.py`
- Function: `analyze_single_subject(...)`
- Input: lysozyme path, tissue path, metadata, output directory, analysis config.
- Output: image-level rows, detailed rows, per-crypt rows, optional overlays, effective-count metadata.

4. Spark batch implementation
- Module: `codeBase/pipeline_implementations/spark_implementation/pipeline.py`
- Function: `run_spark_pipeline(...)`
- Purpose: ingest discovery CSV, distribute per-row analysis with Spark, save aggregate CSV outputs.

5. Central orchestrator
- Module: `codeBase/run.py`
- Wrapper script: `run.sh`
- Purpose:
  - bootstrap config YAML when missing
  - run discovery (with flexible rewrite policy)
  - run analysis backend (`dask` or `spark`)
  - run stats post-processing

## Step-by-Step Run Instructions

### 1) Prepare environment

```bash
cd /home/ash/documents/code/lysozyme
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install pyspark  # required for spark backend
```

### 2) Bootstrap config in your work directory

```bash
./run.sh --work-dir scratch_space/karens_data --yes --skip-analysis --skip-stats
```

This creates:

- `scratch_space/karens_data/lysozyme_pipeline_config.yaml`
- `scratch_space/karens_data/lysozyme_input_data.csv` (empty/template if no matches)

### 3) Edit YAML config

Edit at least:

- `dataset_config.discovery.datasets[*].root_dir`
- dataset `mode` and matching rules (`channel_file_names` for `structured_dir`, `channel_tokens` for `token_match`)
- `dataset_config.exp_name`
- `pipeline_config.results_root`
- `pipeline_config.backend` (`dask` or `spark`)

Optional spark tuning:

- `pipeline_config.spark.master`
- `pipeline_config.spark.partitions`
- `pipeline_config.spark.config`

### 4) Run discovery only (optional but recommended first)

```bash
./run.sh --work-dir scratch_space/karens_data --discovery-only --rewrite-csv always
```

### 5) Run analysis only

Spark:

```bash
./run.sh --work-dir scratch_space/karens_data --analysis-only --backend spark
```

Dask:

```bash
./run.sh --work-dir scratch_space/karens_data --analysis-only --backend dask
```

### 6) Run stats only

```bash
./run.sh --work-dir scratch_space/karens_data --stats-only
```

### 7) Run full pipeline in one command

```bash
./run.sh --work-dir scratch_space/karens_data --backend spark
```

## Useful CLI Flags

From `codeBase/run.py`:

- `--rewrite-csv never|always|ask`
- `--skip-discovery`
- `--skip-analysis`
- `--skip-stats`
- `--max-subjects N`
- `--spark-partitions N`
- `--debug`

And convenience aliases in `run.sh`:

- `--discovery-only`
- `--analysis-only`
- `--stats-only`

## Output Layout

All outputs are under:

- `<results_root>/results/<exp_name>/`

Common:

- `renderings/` (overlays when enabled)
- `stats/` (basic stats post-processing)

Dask:

- `simple_dask_image_summary.csv`
- `simple_dask_image_summary_detailed.csv`
- `simple_dask_image_summary_simpson.csv`
- `simple_dask_image_summary_detailed_simpson.csv`
- `simple_dask_per_crypt.csv`

Spark:

- `simple_spark_image_summary.csv`
- `simple_spark_image_summary_detailed.csv`
- `simple_spark_image_summary_simpson.csv`
- `simple_spark_image_summary_detailed_simpson.csv`
- `simple_spark_per_crypt.csv`
- `simple_spark_errors.csv` (present if any row fails)
