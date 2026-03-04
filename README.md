# Lysozyme Analysis Pipeline

This repository contains a modular image-analysis pipeline for lysozyme-positive intestinal crypt quantification with two execution backends:

- `dask`
- `spark`

The main entrypoint is now:

- `codeBase/run.py` (Python CLI)
- `run.sh` (shell wrapper around `codeBase/run.py`)

## Current Run Flow

The pipeline runs in three stages:

1. Discovery: scan configured datasets and build/update `lysozyme_input_data.csv`.
2. Analysis: process each CSV row with `dask` or `spark`.
3. Stats: generate summary stats outputs from the image-level CSV.

Each stage can be run independently.

## Installation

```bash
git clone https://github.com/phillip-silbert/lysozyme.git
cd lysozyme

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For Spark backend support, also install:

```bash
pip install pyspark
```

## Quick Start (Recommended)

1. Choose a work directory (where config + CSV live).
2. Run the orchestrator once to bootstrap config:

```bash
./run.sh --work-dir scratch_space/karens_data --yes --skip-analysis --skip-stats
```

3. Edit `scratch_space/karens_data/lysozyme_pipeline_config.yaml`.
4. Run the full pipeline:

```bash
./run.sh --work-dir scratch_space/karens_data --backend spark
```

## Stage-Only Runs

Discovery only:

```bash
./run.sh --work-dir scratch_space/karens_data --discovery-only --rewrite-csv always
```

Analysis only:

```bash
./run.sh --work-dir scratch_space/karens_data --analysis-only --backend dask
```

Stats only:

```bash
./run.sh --work-dir scratch_space/karens_data --stats-only
```

You can pass through any `codeBase/run.py` args via `run.sh`.

## Main CLI Options

```bash
python3 codeBase/run.py --help
```

Common options:

- `--work-dir <dir>` or `--config <yaml>`
- `--backend auto|dask|spark`
- `--rewrite-csv never|always|ask`
- `--skip-discovery`
- `--skip-analysis`
- `--skip-stats`
- `--max-subjects <N>`
- `--spark-partitions <N>`

## Outputs

Results are written under:

- `<results_root>/results/<exp_name>/`

Backend outputs:

- Dask:
  - `simple_dask_image_summary.csv`
  - `simple_dask_image_summary_detailed.csv`
  - `simple_dask_per_crypt.csv`
- Spark:
  - `simple_spark_image_summary.csv`
  - `simple_spark_image_summary_detailed.csv`
  - `simple_spark_per_crypt.csv`
  - `simple_spark_errors.csv` (if row failures occurred)

Stats outputs:

- `<results_dir>/stats/basic_stats_overall.json`
- `<results_dir>/stats/basic_stats_by_source_dataset.csv` (when available)
- `<results_dir>/stats/basic_stats_by_image_source_type.csv` (when available)

## Detailed CodeBase Guide

For a step-by-step guide tied to concrete module paths in `codeBase/`, see:

- `codeBase/README.md`

## Legacy Scripts

Legacy `src/` wrappers have been removed. Use `codeBase/run.py` (or `./run.sh`) as the supported entrypoint.
