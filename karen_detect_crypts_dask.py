"""
Pure Dask implementation of lysozyme crypt detection pipeline.
Builds the entire computation graph lazily, then computes all at once.
Supports optional distributed cluster execution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dask
import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.scientific_image_finder.finder import find_subject_image_sets
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from src.lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
from src.lysozyme_stain_quantification.quantify.crypt_fluorescence_summary import (
    summarize_crypt_fluorescence,
    summarize_crypt_fluorescence_per_crypt,
    SUMMARY_FIELD_ORDER,
    PER_CRYPT_FIELD_ORDER,
)
from src.lysozyme_stain_quantification.utils.setup_tools import setup_results_dir, plot_all_crypts

# Import image-ops-framework for proper overlay rendering
sys.path.insert(0, str(Path.home() / "documents" / "image-ops-framework" / "src"))
from image_ops_framework.helpers.overlays import render_label_overlay

# Try to import cluster support
try:
    from dask.distributed import Client, LocalCluster
    CLUSTER_AVAILABLE = True
except ImportError:
    CLUSTER_AVAILABLE = False


# =============================================================================
# CONFIGURATION - Edit these defaults for quick runs without command-line args
# =============================================================================

USE_CLUSTER = True              # Use distributed Dask cluster for parallel processing
N_WORKERS = None                # Number of workers (None = auto-detect: CPU_COUNT/2)
THREADS_PER_WORKER = None       # Threads per worker (None = auto: CPU_COUNT/N_WORKERS)
SAVE_IMAGES = True              # Generate overlay visualizations and plots
DEBUG = True                    # Show detailed progress information
MAX_SUBJECTS = None           # Limit number of subjects (None = process all)


# Advanced settings
BLOB_SIZE_UM = 50.0            # Expected crypt size in microns
MEMORY_PER_WORKER = "4GB"      # Memory limit per worker

# =============================================================================


def array_to_dask_lazy(arr: np.ndarray) -> da.Array:
    """Convert a numpy array to a dask array without copying (lazy reference)."""
    return da.from_array(arr, chunks=arr.shape)


def convert_summary_to_dataframe(
    subject_names: List[str],
    summary_arrays: List[np.ndarray],
    scale_values: List[float],
    image_source_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert image-level summary arrays to a DataFrame matching the original format."""
    records = []
    
    for idx, (subject_name, summary, scale) in enumerate(zip(subject_names, summary_arrays, scale_values)):
        # summary is a 1D array with values in SUMMARY_FIELD_ORDER
        record = {"subject_name": subject_name}
        
        for i, field in enumerate(SUMMARY_FIELD_ORDER):
            if i < len(summary):
                record[field] = float(summary[i])
            else:
                record[field] = np.nan
        
        record["microns_per_px"] = scale
        
        # Add image source type annotation
        if image_source_types:
            record["image_source"] = image_source_types[idx]
        
        records.append(record)
    
    # Create DataFrame with proper column order
    columns = ["subject_name"] + list(SUMMARY_FIELD_ORDER) + ["microns_per_px"]
    if image_source_types:
        columns.append("image_source")
    return pd.DataFrame(records, columns=columns)


def convert_per_crypt_to_dataframe(
    subject_names: List[str],
    per_crypt_arrays: List[np.ndarray],
    image_source_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert per-crypt summary arrays to a DataFrame matching the original format."""
    per_crypt_records = []
    
    # Fields from PER_CRYPT_FIELD_ORDER (excluding subject_name which we add)
    numeric_fields = list(PER_CRYPT_FIELD_ORDER[1:])  # Skip subject_name
    int_fields = {"crypt_label", "crypt_index", "pixel_area"}
    
    for subject_idx, (subject_name, data) in enumerate(zip(subject_names, per_crypt_arrays)):
        if data is None or len(data) == 0:
            continue
        
        # data is a 2D array: (num_crypts, num_fields)
        # Each row is one crypt with values in order: crypt_label, crypt_index, pixel_area, ...
        for row_idx in range(data.shape[0]):
            row = {"subject_name": subject_name}
            
            for field_idx, field in enumerate(numeric_fields):
                if field_idx < data.shape[1]:
                    value = data[row_idx, field_idx]
                    
                    if np.isnan(value):
                        row[field] = np.nan
                    elif field in int_fields:
                        row[field] = int(round(value))
                    else:
                        row[field] = float(value)
                else:
                    row[field] = np.nan
            
            # Add image source type annotation
            if image_source_types:
                row["image_source"] = image_source_types[subject_idx]
            
            per_crypt_records.append(row)
    
    columns = list(PER_CRYPT_FIELD_ORDER)
    if image_source_types:
        columns.append("image_source")
    return pd.DataFrame(per_crypt_records, columns=columns)


def save_overlay_images(
    subject_names: List[str],
    rfp_images: List[np.ndarray],
    dapi_images: List[np.ndarray],
    crypt_labels: List[np.ndarray],
    output_dir: Path,
    image_source_types: Optional[List[str]] = None,
) -> None:
    """
    Render and save crypt overlay images using proper render_label_overlay.
    
    Uses the image-ops-framework overlay renderer for proper RGB blending
    of RFP and DAPI channels with label overlays.
    """
    overlay_dir = output_dir / "renderings"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[OVERLAYS] Rendering overlay images...")
    
    @delayed
    def _render_and_save_one(subject_name: str, rfp: np.ndarray, dapi: np.ndarray, labels: np.ndarray, source_type: str = "unknown") -> Path:
        """Render and save a single overlay (delayed for parallelization)."""
        # Use the proper render_label_overlay from image-ops-framework
        # It expects channels=[rfp, dapi, labels]
        overlay_xr = render_label_overlay(
            channels=[rfp, dapi, labels],
            fill_alpha=0.35,
            outline_alpha=1.0,
            outline_width=2,
            normalize_scalar=True,
        )
        
        # overlay_xr is (channel, y, x) with channel=['r', 'g', 'b']
        # Convert to (y, x, channel) for matplotlib
        overlay_rgb = np.moveaxis(overlay_xr.values, 0, -1)
        
        # Save with a sanitized filename that includes source type
        safe_name = subject_name.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
        output_path = overlay_dir / f"{safe_name}_{source_type}_overlay.png"
        
        # Save as PNG (values are already in [0, 1] range)
        import matplotlib.pyplot as plt
        plt.imsave(output_path, overlay_rgb)
        
        return output_path
    
    # Build delayed tasks for all overlays (parallelizable!)
    if image_source_types:
        delayed_saves = [
            _render_and_save_one(name, rfp, dapi, labels, source_type)
            for name, rfp, dapi, labels, source_type in zip(subject_names, rfp_images, dapi_images, crypt_labels, image_source_types)
        ]
    else:
        delayed_saves = [
            _render_and_save_one(name, rfp, dapi, labels, "unknown")
            for name, rfp, dapi, labels in zip(subject_names, rfp_images, dapi_images, crypt_labels)
        ]
    
    # Compute all overlays in parallel
    output_paths = dask.compute(*delayed_saves)
    
    print(f"  Saved {len(output_paths)} overlay images to {overlay_dir}")


def main(
    use_cluster: bool = USE_CLUSTER,
    n_workers: Optional[int] = N_WORKERS,
    threads_per_worker: Optional[int] = THREADS_PER_WORKER,
    save_images: bool = SAVE_IMAGES,
    debug: bool = DEBUG,
    max_subjects: Optional[int] = MAX_SUBJECTS,
) -> None:
    # Suppress the "large graph" warning - we handle this with client.scatter()
    import warnings
    warnings.filterwarnings('ignore', message='.*large graph.*')
    
    print("=" * 80)
    print("DASK-BASED LYSOZYME CRYPT DETECTION PIPELINE")
    print("=" * 80)
    
    # Check for cluster support and connect FIRST (before any other setup)
    cluster_context = None
    client = None
    
    if use_cluster:
        if not CLUSTER_AVAILABLE:
            print("\nWARNING: Cluster support not available. Falling back to threaded scheduler.")
            print("  Install dask.distributed or check cluster.py import.")
            use_cluster = False
        else:
            try:
                from dask.distributed import Client, LocalCluster
                
                # Try to connect to existing cluster first
                try:
                    client = Client(timeout='2s')  # Try to connect to default scheduler
                    print(f"\n✓ Connected to existing Dask cluster!")
                    print(f"  Scheduler: {client.scheduler.address}")
                    print(f"  Dashboard: {client.dashboard_link}")
                    print(f"  Workers: {len(client.scheduler_info()['workers'])}")
                    print(f"\n  📊 MONITOR: {client.dashboard_link}")
                    print()
                except (OSError, TimeoutError):
                    # No existing cluster, start our own
                    print(f"\nNo existing cluster found. Starting local Dask cluster...")
                    
                    # Auto-detect optimal worker configuration
                    import multiprocessing
                    n_cpus = multiprocessing.cpu_count()
                    
                    if n_workers is None:
                        # Default: balanced strategy (half cores as workers, 2 threads each)
                        n_workers = max(1, n_cpus // 2)
                    
                    if threads_per_worker is None:
                        # Calculate threads to use all CPUs
                        threads_per_worker = max(1, n_cpus // n_workers)
                    
                    # Memory per worker: total RAM / workers (with safety margin)
                    # Assume 4GB per worker as safe default, can be increased
                    memory_per_worker = "4GB"
                    
                    if debug:
                        print(f"  Detected {n_cpus} CPUs")
                        print(f"  Configuring: {n_workers} workers × {threads_per_worker} threads = {n_workers * threads_per_worker} total threads")
                    
                    cluster = LocalCluster(
                        n_workers=n_workers,
                        threads_per_worker=threads_per_worker,
                        memory_limit=memory_per_worker,
                        dashboard_address=":8787",
                    )
                    client = cluster.get_client()
                    cluster_context = cluster  # Store for cleanup
                    print(f"  Scheduler: {cluster.scheduler_address}")
                    print(f"  Dashboard: {cluster.dashboard_link}")
                    print(f"  Workers: {n_workers} × {threads_per_worker} threads = {n_workers * threads_per_worker} total")
                    print(f"\n  📊 MONITOR: {cluster.dashboard_link}")
                    print()
            except Exception as e:
                print(f"\nWARNING: Failed to start cluster: {e}")
                print("  Falling back to threaded scheduler.")
                use_cluster = False
    
    # Setup results directory (after cluster connection)
    results_dir = setup_results_dir(SCRIPT_DIR, exp_name="karen_dask")
    
    # Configuration
    IMAGE_BASE_DIR = Path("lysozyme images")
    BLOB_SIZE_UM = 50.0
    
    # Find and organize images
    print("\n[1/6] Finding images...")
    
    # Try combined-channel images first, collecting all valid ones
    combined_subject_names = []
    combined_rfp_images = []
    combined_dapi_images = []
    failed_subjects = []
    
    # Helper functions for combined image processing
    def _has_multi_channel(arr: np.ndarray) -> bool:
        array = np.asarray(arr)
        return array.ndim >= 3 and (array.shape[-1] >= 3 or array.shape[0] >= 3)
    
    def _split_combined_image(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(arr)
        array = np.squeeze(array)
        if array.ndim == 3 and array.shape[-1] >= 3:
            channels_last = array
        elif array.ndim == 3 and array.shape[0] >= 3:
            channels_last = np.moveaxis(array, 0, -1)
        else:
            raise ValueError(f"Combined image must provide a 3-channel RGB payload; got shape {array.shape}")
        if channels_last.shape[-1] < 3:
            raise ValueError(f"Combined image requires at least 3 channels; got {channels_last.shape[-1]}")
        red = channels_last[..., 0]
        blue = channels_last[..., 2]
        return red, blue
    
    # Search for combined images (finder now filters out shape mismatches automatically)
    all_subject_names, combined_sources, _ = find_subject_image_sets(
        img_dir=IMAGE_BASE_DIR,
        sources=[("combined", "")],
        max_subjects=max_subjects,
    )
    
    combined_images = combined_sources[0] if combined_sources else []
    
    # Process each subject individually - failures don't compromise others
    for subject_name, img in zip(all_subject_names, combined_images):
        try:
            if _has_multi_channel(img):
                red, blue = _split_combined_image(img)
                combined_subject_names.append(subject_name)
                combined_rfp_images.append(red)
                combined_dapi_images.append(blue)
            else:
                # Not multi-channel, mark for separate channel search
                failed_subjects.append(subject_name)
        except (ValueError, Exception) as e:
            # Individual failure - mark this subject for separate channel search
            if debug:
                print(f"  Note: Could not use combined image for '{subject_name}': {str(e)[:80]}")
            failed_subjects.append(subject_name)
    
    if combined_subject_names:
        print(f"✓ Loaded {len(combined_subject_names)} combined-channel images (red+blue extraction).")
    
    # For subjects that failed combined search, try per-channel search
    separate_subject_names = []
    separate_rfp_images = []
    separate_dapi_images = []
    
    if failed_subjects or not combined_subject_names:
        if failed_subjects:
            print(f"→ Searching for separate RFP/DAPI channels for {len(failed_subjects)} subjects that failed combined loading...")
        else:
            print("→ No combined-channel images found; searching for separate RFP/DAPI channels...")
        
        # Search for separate RFP and DAPI channels
        sep_names, images_by_source, source_names = find_subject_image_sets(
            img_dir=IMAGE_BASE_DIR,
            sources=[("rfp", "rfp", "r"), ("dapi", "dapi", "b")],
            max_subjects=max_subjects,
        )
        
        # If we had failed subjects, only take those that were in the failed list
        if failed_subjects:
            for idx, name in enumerate(sep_names):
                if name in failed_subjects:
                    separate_subject_names.append(name)
                    separate_rfp_images.append(images_by_source[0][idx])
                    separate_dapi_images.append(images_by_source[1][idx])
        else:
            # No combined images worked, use all separate channel images
            separate_subject_names = sep_names
            separate_rfp_images = images_by_source[0]
            separate_dapi_images = images_by_source[1]
        
        if separate_subject_names:
            print(f"✓ Loaded {len(separate_subject_names)} subjects with separate channels.")
    
    # Merge both lists
    subject_names = combined_subject_names + separate_subject_names
    rfp_images = combined_rfp_images + separate_rfp_images
    dapi_images = combined_dapi_images + separate_dapi_images
    
    # Track which subjects used which method (for annotation)
    image_source_type = ["combined"] * len(combined_subject_names) + ["separate"] * len(separate_subject_names)
    
    if not subject_names:
        raise ValueError("No valid images found!")
    
    print(f"\n✓ Image loading complete: {len(subject_names)} total subjects")
    print(f"  → {len(combined_subject_names)} from combined RGB images")
    print(f"  → {len(separate_subject_names)} from separate channel images")
    
    source_names = ["rfp", "dapi"]
    
    if debug:
        print(f"Sources: {source_names}")
        print(f"Found {len(subject_names)} subjects with images in both channels.")
        for subject, rfp, dapi in zip(subject_names, rfp_images, dapi_images):
            print(f"Subject: {subject}, Red shape: {rfp.shape}, Blue shape: {dapi.shape}")
    
    # Setup scale lookup
    scale_keys = ["40x"]
    scale_values = [0.2253]
    default_scale_value = 0.4476
    microns_per_px_values = []
    
    for name in subject_names:
        matched_value = default_scale_value
        for key, value in zip(scale_keys, scale_values):
            if key.lower() in name.lower():
                matched_value = value
                break
        microns_per_px_values.append(matched_value)
    
    if debug:
        print(f"\nUsing blob size (crypt size) of {BLOB_SIZE_UM} microns.")
    
    # Pre-scatter data to workers if using cluster (avoids large graph warning)
    if use_cluster and client:
        print("\n[2/6] Scattering data to cluster workers...")
        if debug:
            print("  (This avoids sending large graphs over the network)")
        
        # Scatter all images to workers at once
        scattered_rfp = client.scatter(rfp_images, broadcast=False)
        scattered_dapi = client.scatter(dapi_images, broadcast=False)
        
        if debug:
            print(f"  Scattered {len(scattered_rfp)} image pairs to workers")
    else:
        scattered_rfp = rfp_images
        scattered_dapi = dapi_images
    
    # Build the lazy computation graph for all subjects
    print("\n[3/6] Building dask computation graph...")
    
    # Dictionary to store delayed computations for each subject
    subject_computations: Dict[str, Dict] = {}
    
    for idx, (subject_name, scale) in enumerate(zip(subject_names, microns_per_px_values)):
        if debug:
            print(f"  Adding subject: {subject_name}")
        
        # Get the images (either scattered futures or numpy arrays)
        if use_cluster and client:
            rfp_np = scattered_rfp[idx]
            dapi_np = scattered_dapi[idx]
            # Convert futures to dask arrays
            rfp_img = da.from_delayed(rfp_np, shape=rfp_images[idx].shape, dtype=rfp_images[idx].dtype)
            dapi_img = da.from_delayed(dapi_np, shape=dapi_images[idx].shape, dtype=dapi_images[idx].dtype)
        else:
            rfp_np = rfp_images[idx]
            dapi_np = dapi_images[idx]
            # Convert numpy arrays to dask arrays (lazy reference, no copy)
            rfp_img = array_to_dask_lazy(rfp_np)
            dapi_img = array_to_dask_lazy(dapi_np)
        
        # Create the computation graph (all lazy!)
        # Step 1: Segment crypts
        crypt_labels = segment_crypts(
            channels=[rfp_img, dapi_img, scale],
            blob_size_um=BLOB_SIZE_UM,
            debug=False,
            max_regions=5,
        )
        
        # Step 2: Normalize RFP
        normalized_rfp = compute_normalized_rfp(
            channels=[rfp_img, dapi_img, crypt_labels]
        )
        
        # Step 3: Quantify fluorescence (per-image summary)
        @delayed
        def _summarize_image(norm_rfp, labels, scale_val):
            """Summarize fluorescence for entire image."""
            # Compute inputs if they're dask arrays
            norm_rfp = norm_rfp.compute() if isinstance(norm_rfp, da.Array) else np.asarray(norm_rfp)
            labels = labels.compute() if isinstance(labels, da.Array) else np.asarray(labels)
            
            # Function expects [normalized_rfp, crypt_labels, microns_per_px]
            result = summarize_crypt_fluorescence(
                channels=[norm_rfp, labels, scale_val],
                intensity_upper_bound=1.0,
            )
            return result
        
        # Step 4: Quantify per-crypt fluorescence
        @delayed
        def _summarize_per_crypt(norm_rfp, labels, scale_val):
            """Summarize fluorescence per individual crypt."""
            # Compute inputs if they're dask arrays
            norm_rfp = norm_rfp.compute() if isinstance(norm_rfp, da.Array) else np.asarray(norm_rfp)
            labels = labels.compute() if isinstance(labels, da.Array) else np.asarray(labels)
            
            # Function expects [normalized_rfp, crypt_labels, microns_per_px]
            result = summarize_crypt_fluorescence_per_crypt(
                channels=[norm_rfp, labels, scale_val]
            )
            return result
        
        image_summary = _summarize_image(normalized_rfp, crypt_labels, scale)
        per_crypt_summary = _summarize_per_crypt(normalized_rfp, crypt_labels, scale)
        
        # Store the delayed computations
        subject_computations[subject_name] = {
            "crypt_labels": crypt_labels,
            "normalized_rfp": normalized_rfp,
            "image_summary": image_summary,
            "per_crypt_summary": per_crypt_summary,
            "rfp_original": rfp_images[idx],  # Keep original for overlay
            "dapi_original": dapi_images[idx],
        }
    
    if debug:
        print(f"\n✓ Graph built for {len(subject_computations)} subjects")
        print("  (No computation has occurred yet - everything is lazy!)")
    
    # Now compute everything at once!
    step_num = "4/7" if use_cluster else "3/6"
    print(f"\n[{step_num}] Computing all results (this may take a while)...")
    if use_cluster and client:
        print(f"  Using Dask distributed cluster with {len(client.scheduler_info()['workers'])} workers")
    else:
        print("  Using threaded scheduler with 4 workers")
    
    # Gather all the delayed objects we want to compute
    image_summaries = [comp["image_summary"] for comp in subject_computations.values()]
    per_crypt_summaries = [comp["per_crypt_summary"] for comp in subject_computations.values()]
    crypt_labels_list = [comp["crypt_labels"] for comp in subject_computations.values()]
    
    # Compute all at once!
    if use_cluster:
        # Use distributed scheduler (already set as default)
        if debug:
            print("  Computing with distributed scheduler...")
        image_results = dask.compute(*image_summaries)
        per_crypt_results = dask.compute(*per_crypt_summaries)
        crypt_labels_computed = dask.compute(*crypt_labels_list)
    else:
        # Use threaded scheduler
        with dask.config.set(scheduler='threads', num_workers=4):
            if debug:
                print("  Computing image summaries...")
            image_results = dask.compute(*image_summaries)
            
            if debug:
                print("  Computing per-crypt summaries...")
            per_crypt_results = dask.compute(*per_crypt_summaries)
            
            if debug and save_images:
                print("  Computing crypt labels for visualization...")
            crypt_labels_computed = dask.compute(*crypt_labels_list)
    
    if debug:
        print("\n✓ All computations complete!")
    
    # Convert results to DataFrames
    step_num = "5/7" if use_cluster else "4/6"
    print(f"\n[{step_num}] Converting results to DataFrames...")
    
    # Image-level summary
    image_summary_df = convert_summary_to_dataframe(
        subject_names, image_results, microns_per_px_values, image_source_type
    )
    if debug:
        print(f"  Image summary: {len(image_summary_df)} rows")
    
    # Per-crypt summary  
    per_crypt_df = convert_per_crypt_to_dataframe(
        subject_names, per_crypt_results, image_source_type
    )
    if debug:
        print(f"  Per-crypt summary: {len(per_crypt_df)} rows")
    
    # Save CSV results
    step_num = "6/7" if use_cluster else "5/6"
    print(f"\n[{step_num}] Saving results...")
    
    csv_output = results_dir / "karen_detect_crypts_dask.csv"
    image_summary_df.to_csv(csv_output, index=False)
    if debug:
        print(f"  Saved image summary: {csv_output}")
    
    per_crypt_output = results_dir / "karen_detect_crypts_dask_per_crypt.csv"
    per_crypt_df.to_csv(per_crypt_output, index=False)
    if debug:
        print(f"  Saved per-crypt summary: {per_crypt_output}")
    
    # Save overlay images if requested
    if save_images:
        step_num = "7/7" if use_cluster else "6/6"
        print(f"\n[{step_num}] Generating visualizations...")
        try:
            save_overlay_images(
                subject_names,
                rfp_images,
                dapi_images,
                crypt_labels_computed,
                results_dir,
                image_source_type,
            )
            
            # Also create the grid plot
            if debug:
                print(f"  Creating grid visualization...")
            
            # Create a simple dataset-like structure for plot_all_crypts
            # We'll create a matplotlib figure with all crypts
            fig, axes = plt.subplots(2, 5, figsize=(18, 8))
            axes = axes.flatten()
            
            for idx, (name, labels) in enumerate(zip(subject_names, crypt_labels_computed)):
                if idx < len(axes):
                    axes[idx].imshow(labels, cmap='tab20', interpolation='nearest')
                    axes[idx].set_title(name.split('[')[0].strip(), fontsize=8)
                    axes[idx].axis('off')
            
            # Hide unused subplots
            for idx in range(len(subject_names), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            grid_output = results_dir / "karen_detect_crypts_dask.png"
            fig.savefig(grid_output, dpi=200, bbox_inches="tight")
            plt.close(fig)
            
            if debug:
                print(f"  Saved grid visualization: {grid_output}")
                
        except Exception as e:
            print(f"  WARNING: Failed to generate visualizations: {e}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nSuccessfully processed {len(subject_names)} subjects using Dask lazy evaluation!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    # Clean up cluster if we started one
    if cluster_context:
        print("\n✓ Shutting down local cluster...")
        cluster_context.close()
        if client:
            client.close()
        if debug:
            print("  Cluster shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dask-based lysozyme crypt detection pipeline"
    )
    parser.add_argument(
        "--use-cluster",
        action="store_true",
        default=USE_CLUSTER,
        help=f"Use Dask distributed cluster (local or existing) [default: {USE_CLUSTER}]",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=N_WORKERS,
        help=f"Number of workers for local cluster [default: {N_WORKERS} (auto-detect: CPU_COUNT/2)]",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=THREADS_PER_WORKER,
        help=f"Threads per worker [default: {THREADS_PER_WORKER} (auto-calculated to use all CPUs)]",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help=f"Skip generating overlay images and visualizations [default: {not SAVE_IMAGES}]",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=DEBUG,
        help=f"Enable debug output [default: {DEBUG}]",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=MAX_SUBJECTS,
        help=f"Maximum number of subjects to process [default: {MAX_SUBJECTS} (all)]",
    )
    
    args = parser.parse_args()
    
    main(
        use_cluster=args.use_cluster,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        save_images=not args.no_images,
        debug=args.debug,
        max_subjects=args.max_subjects,
    )
   