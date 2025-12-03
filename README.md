# Lysozyme Stain Quantification & Dask Pipeline

Tools for detecting and quantifying lysozyme‑positive intestinal crypts from paired fluorescence images (RFP/lysozyme and DAPI). The core algorithms live in `src/lysozyme_stain_quantification`, and the Dask-based batch pipeline is wired up through small entry-point scripts in this repository.

- Overview video (concept + walkthrough): https://youtu.be/Qp7FabiPl2A
- Detailed theory and background: `docs/PipelineTheory.md`, `docs/crypt_fluorescence_summary.md`, `docs/WIKI.md`

## Installation

These instructions assume a Unix-like environment and Python ≥3.9.

```bash
git clone https://github.com/phillip-silbert/lysozyme.git
cd lysozyme

# (optional) create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# install dependencies and this package in editable mode
pip install -r requirements.txt
pip install -e .
```

After this, you should be able to import the library (e.g. `import lysozyme_stain_quantification`) and run the Dask pipelines below.

## Input Data Expectations

- Images are `.tif`/`.TIF` files organized under a root directory.
- Each subject is typically a pair of images:
  - A lysozyme/RFP channel (red)
  - A DAPI (nuclear) channel (blue)
- Pairing is done by filename patterns:
  - For the Karen dataset, files end with `"_RFP"` and `"_DAPI"` (see `src/dask_lysozyme_pipeline.py`).
  - For the Yen lab dataset, files end with `"c2"` (lysozyme) and `"c1"` (DAPI) (see `yen_detect_crypts.py`).
- The root directory for a given dataset is configured via `DatasetConfig.image_base_dir`.

If your images live somewhere else or use different naming, adapt the `DatasetConfig` in the relevant entry script.

## Running the Dask Pipeline (Karen Dataset)

The shared Dask pipeline is configured for the “Karen” dataset in `src/dask_lysozyme_pipeline.py`.

1. Edit the input path if needed  
   In `src/dask_lysozyme_pipeline.py`, update:
   ```python
   KAREN_DATASET = DatasetConfig(
       image_base_dir=Path("/home/phillip/documents/experimental_data/inputs/karen/lysozyme"),
       ...
       channel_keys=("_RFP", "_DAPI"),
   )
   ```
   so that `image_base_dir` points to your Karen image root.

2. Run the pipeline
   ```bash
   # from the repo root
   python src/dask_lysozyme_pipeline.py --max-subjects 10
   ```

   Useful flags (from `src/lysozyme_pipelines/cli.py`):
   - `--max-subjects N` – limit how many subjects/images are processed.
   - `--capture-debug-images` / `--no-capture-debug-images` – toggle saving intermediate debug images.
   - `--debug-stage STAGE` – add one or more stages to the debug whitelist (repeatable).
   - `--debug-subject-count N` – only capture debug intermediates for the first `N` subjects.
   - `--debug-subject NAME` – restrict debug capture to specific subject names (repeatable).

3. Inspect results  
   For the Karen dataset, results are written under a run-specific directory, e.g.:
   - `results/<exp_name>/simple_dask_image_summary.csv`
   - `results/<exp_name>/simple_dask_image_summary_detailed.csv`
   - `results/<exp_name>/simple_dask_per_crypt.csv`
   - `results/<exp_name>/renderings/` – overlay images (RFP/DAPI/crypt labels)
   - `results/<exp_name>/debug_intermediates/` – optional stepwise debug images

   The `exp_name` is set in the `DatasetConfig` (e.g. `"rendering_test_run"`).

## Running the Dask Pipeline (Yen Lab Dataset)

The Yen lab variant is wired up in `yen_detect_crypts.py` and uses the same shared pipeline under the hood.

1. Edit the input path if needed  
   In `yen_detect_crypts.py`, update:
   ```python
   YEN_IMAGE_BASE = Path("/home/phillip/documents/experimental_data/inputs/yen_lab/rfp/Lyz Fabp1")
   ```
   so it points to your Yen dataset root. The default expects files ending in `"c2"` (lysozyme) and `"c1"` (DAPI).

2. Run the pipeline
   ```bash
   # from the repo root
   python yen_detect_crypts.py --max-subjects 10
   ```

   The same debug flags described above are available (`--capture-debug-images`, `--debug-stage`, `--debug-subject`, etc.). Results and CSVs are written under `results/<exp_name>/` where `exp_name` is `"yen_lab_run"` by default.

## Dask Cluster Usage

The shared pipeline (`src/lysozyme_pipelines/pipeline.py`) supports both threaded and distributed execution:

- By default, the entry scripts (`src/dask_lysozyme_pipeline.py`, `yen_detect_crypts.py`) set `use_cluster=True`.
- If `dask.distributed` is available, the pipeline will:
  - Try to connect to an existing local cluster (if `connect_to_existing_cluster=True`).
  - Otherwise, start a `LocalCluster` with a reasonable number of workers/threads and report:
    - Scheduler URL
    - Dashboard URL (typically `http://127.0.0.1:8787/status`)
- If Dask distributed is not installed or cluster start fails, it falls back to a threaded scheduler.

You can monitor progress, memory use, and task graphs by opening the printed Dask dashboard URL in a browser.

## What the Pipeline Produces

At a high level, the Dask pipeline:

- Discovers and pairs RFP (lysozyme) and DAPI images from a dataset directory.
- Applies a watershed-based segmentation pipeline to identify candidate crypts.
- Scores regions using morphology and intensity features (circularity, area, alignment, lysozyme intensity, etc.).
- Selects the best crypt candidates per image.
- Normalizes lysozyme signal relative to DAPI to enable cross-image comparisons.
- Writes:
  - Per-image summaries (`simple_dask_image_summary*.csv`)
  - Detailed per-image metrics (`simple_dask_image_summary_detailed*.csv`)
  - Per-crypt tables (`simple_dask_per_crypt.csv`)
  - Optional overlay and debug images.

For in-depth definitions of these outputs and the biological/statistical rationale, see `docs/crypt_fluorescence_summary.md` and the R scripts in `src/statistical_validation`.

