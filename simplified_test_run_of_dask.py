from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dask
import dask.array as da
from dask.delayed import delayed
import dask.bag as db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask_image.imread import imread

from src.scientific_image_finder.finder import find_subject_image_sets, find_tif_images_by_keys
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from src.lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
from src.lysozyme_stain_quantification.quantify.crypt_fluorescence_summary import (
    summarize_crypt_fluorescence,
    summarize_crypt_fluorescence_per_crypt,
    SUMMARY_FIELD_ORDER,
    PER_CRYPT_FIELD_ORDER,
)
from src.lysozyme_stain_quantification.utils.setup_tools import setup_results_dir, plot_all_crypts
# Add src to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
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
FORCE_RESPAWN_CLUSTER = False   # Force respawn cluster if params don't match (WARNING: closes existing cluster)
N_WORKERS = None                # Number of workers (None = auto-detect: CPU_COUNT/2)
THREADS_PER_WORKER = None       # Threads per worker (None = auto: CPU_COUNT/N_WORKERS)
SAVE_IMAGES = True              # Generate overlay visualizations and plots
DEBUG = True                    # Show detailed progress information
MAX_SUBJECTS = 5           # Limit number of subjects (None = process all)


# Advanced settings
BLOB_SIZE_UM = 50.0            # Expected crypt size in microns
MEMORY_PER_WORKER = "4GB"      # Memory limit per worker
# pathing
IMAGE_BASE_DIR = Path("lysozyme images")
BLOB_SIZE_UM = 50.0
# =============================================================================


def main(
    use_cluster: bool = USE_CLUSTER,
    force_respawn_cluster: bool = FORCE_RESPAWN_CLUSTER,
    n_workers: Optional[int] = N_WORKERS,
    threads_per_worker: Optional[int] = THREADS_PER_WORKER,
    save_images: bool = SAVE_IMAGES,
    debug: bool = DEBUG,
    max_subjects: Optional[int] = MAX_SUBJECTS,
) -> None:
    print("=" * 80)
    print("DASK-BASED LYSOZYME CRYPT DETECTION PIPELINE")
    print("=" * 80)

    if use_cluster:
        if not CLUSTER_AVAILABLE:
            print("\nWARNING: Cluster support not available. Falling back to threaded scheduler.")
            print("  Install dask.distributed or check cluster.py import.")
            use_cluster = False
        else:
            try:
                from dask.distributed import Client, LocalCluster
                
                # Auto-detect optimal worker configuration first (needed for comparison)
                import multiprocessing
                n_cpus = multiprocessing.cpu_count()
                
                desired_n_workers = n_workers
                desired_threads_per_worker = threads_per_worker
                
                if desired_n_workers is None:
                    # Default: balanced strategy (half cores as workers, 2 threads each)
                    desired_n_workers = max(1, n_cpus // 2 - 2)
                
                if desired_threads_per_worker is None:
                    # Calculate threads to use all CPUs
                    desired_threads_per_worker = max(1, n_cpus // desired_n_workers)
                
                # Try to connect to existing cluster first
                existing_client = None
                needs_respawn = False
                
                try:
                    existing_client = Client(timeout='2s')  # Try to connect to default scheduler
                    scheduler_info = existing_client.scheduler_info()
                    workers_info = scheduler_info['workers']
                    actual_n_workers = len(workers_info)
                    
                    # Check threads per worker (sample from first worker)
                    actual_threads_per_worker = None
                    if workers_info:
                        first_worker = list(workers_info.values())[0]
                        actual_threads_per_worker = first_worker.get('nthreads', None)
                    
                    print(f"\n✓ Found existing Dask cluster!")
                    print(f"  Scheduler: {existing_client.scheduler.address}") #type: ignore[attr-defined]
                    print(f"  Dashboard: {existing_client.dashboard_link}")
                    print(f"  Workers: {actual_n_workers} × {actual_threads_per_worker} threads")
                    
                    # Check if parameters match
                    params_match = (
                        actual_n_workers == desired_n_workers and
                        actual_threads_per_worker == desired_threads_per_worker
                    )
                    
                    if not params_match:
                        print(f"\n⚠️  Cluster parameters don't match desired configuration:")
                        print(f"    Existing: {actual_n_workers} workers × {actual_threads_per_worker} threads")
                        print(f"    Desired:  {desired_n_workers} workers × {desired_threads_per_worker} threads")
                        
                        if force_respawn_cluster:
                            print(f"    FORCE_RESPAWN_CLUSTER=True: Closing existing cluster and respawning...")
                            needs_respawn = True
                            existing_client.close()
                            existing_client = None
                        else:
                            print(f"    Using existing cluster anyway (set FORCE_RESPAWN_CLUSTER=True to respawn)")
                    else:
                        print(f"  ✓ Parameters match desired configuration")
                    
                    if existing_client is not None:
                        client = existing_client
                        print(f"\n  📊 MONITOR: {client.dashboard_link}")
                        print()
                        
                except (OSError, TimeoutError):
                    # No existing cluster, need to start our own
                    needs_respawn = True
                    if debug:
                        print(f"\nNo existing cluster found.")
                
                # Start new cluster if needed
                if needs_respawn or existing_client is None:
                    print(f"\nStarting new Dask cluster...")
                    
                    # Memory per worker: total RAM / workers (with safety margin)
                    # Assume 4GB per worker as safe default, can be increased
                    memory_per_worker = "4GB"
                    
                    if debug:
                        print(f"  Detected {n_cpus} CPUs")
                        print(f"  Configuring: {desired_n_workers} workers × {desired_threads_per_worker} threads = {desired_n_workers * desired_threads_per_worker} total threads")
                    
                    cluster = LocalCluster(
                        n_workers=desired_n_workers,
                        threads_per_worker=desired_threads_per_worker,
                        memory_limit=memory_per_worker,
                        dashboard_address=":8787",
                    )
                    client = cluster.get_client()
                    cluster_context = cluster  # Store for cleanup
                    print(f"  Scheduler: {cluster.scheduler_address}")
                    print(f"  Dashboard: {cluster.dashboard_link}")
                    print(f"  Workers: {desired_n_workers} × {desired_threads_per_worker} threads = {desired_n_workers * desired_threads_per_worker} total")
                    print(f"\n  📊 MONITOR: {cluster.dashboard_link}")
                    print()
                    
            except Exception as e:
                print(f"\nWARNING: Failed to start cluster: {e}")
                print("  Falling back to threaded scheduler.")
                use_cluster = False
    # Setup results directory (after cluster connection)
    results_dir: Path = setup_results_dir(SCRIPT_DIR, exp_name="simple_dask")
    print(f"Results will be saved to: {results_dir.resolve()}\n")
    # Find subject image sets
    unmatched, pairs = find_tif_images_by_keys(
        IMAGE_BASE_DIR,
        keys=["_RFP", "_DAPI"],

        max_subjects=max_subjects,
    )
    image_bags = {}
    # for key, paths in :
    #     print(f"Found {len(paths)} images for key '{key}'")
    #     image_bags[key] = db.from_sequence(paths).map(imread).map(da.squeeze)
    seperate_channels_bag = db.from_sequence(pairs).map(
        lambda x:dict(
            paths=x,
            rfp=imread(x[0]).squeeze()[...,0],
            dapi=imread(x[1]).squeeze()[...,2],
        ))
    combined_channels_bag = db.from_sequence(unmatched).map(
        lambda x:dict(
            paths=[x],
            image=imread(x).squeeze(),
        )).map( #TODO: prob need to add in remove rectangles
            lambda x:dict(
                paths=x["paths"],
                rfp=x["image"][...,0],
                dapi=x["image"][...,2],
            )
        )
    full_bag = db.concat([seperate_channels_bag, combined_channels_bag])









   




if __name__ == "__main__":
    main()