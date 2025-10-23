# region imports
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
# endregion 

# region configuration
# =============================================================================
# CONFIGURATION - Edit these defaults for quick runs without command-line args
# =============================================================================

USE_CLUSTER = True              # Use distributed Dask cluster for parallel processing
FORCE_RESPAWN_CLUSTER = False   # Force respawn cluster if params don't match (WARNING: closes existing cluster)
N_WORKERS = None                # Number of workers (None = auto-detect: CPU_COUNT/2)
THREADS_PER_WORKER = None       # Threads per worker (None = auto: CPU_COUNT/N_WORKERS)
SAVE_IMAGES = True              # Generate overlay visualizations and plots
DEBUG = True                    # Show detailed progress information
MAX_SUBJECTS = 10           # Limit number of subjects (None = process all)


# Advanced settings
BLOB_SIZE_UM = 50.0            # Expected crypt size in microns
MEMORY_PER_WORKER = "4GB"      # Memory limit per worker
# pathing
IMAGE_BASE_DIR = Path("lysozyme images")
BLOB_SIZE_UM = 50.0 * 0.4476
# =============================================================================
# endregion configuration

# region helper funcs
def find_tif_images_by_keys(
    root_dir: Path,
    keys: List[str],
    max_subjects: Optional[int] = None,
) -> Tuple[List[Path], List[Tuple[Path, ...]]]:
    """
    Recursively search for .tif images, pair them by matching keys and base name.
    
    Images are paired by extracting their base name (removing the key suffix) and 
    grouping images with the same base name. Each image is assigned to the first 
    matching key only (no duplicates).
    
    Parameters
    ----------
    root_dir : Path
        Directory to search recursively
    keys : List[str]
        List of string keys to match and pair (e.g., ["_DAPI", "_RFP"])
        Order matters - first match wins
    max_subjects : Optional[int]
        Maximum number of paired subjects to return
    
    Returns
    -------
    Tuple[List[Path], List[Tuple[Path, ...]]]
        (unmatched_paths, paired_paths) where:
        - unmatched_paths: List of .tif images that don't match any key
        - paired_paths: List of tuples, each containing paths in key order
          that share the same base name
        
    Example
    -------
    >>> unmatched, pairs = find_tif_images_by_keys(
    ...     Path("./images"),
    ...     keys=["_RFP", "_DAPI"]
    ... )
    >>> # pairs might look like:
    >>> # [
    >>> #   (Path("subject1_RFP.tif"), Path("subject1_DAPI.tif")),
    >>> #   (Path("subject2_RFP.tif"), Path("subject2_DAPI.tif")),
    >>> # ]
    >>> # unmatched contains any .tif that doesn't have _RFP or _DAPI
    """
    def _extract_base_name(file_name: str, search_key: str) -> str | None:
        key = (search_key or "").lower().strip()
        lower_name = file_name.lower()

        if not key:
            stem = Path(file_name).stem
            stem_lower = stem.lower()
            if stem_lower.endswith(("_rfp", "_dapi")):
                return None
            return stem.rstrip(" _-")

        separators = ("_", "-", " ", "")
        for sep in separators:
            token = f"{sep}{key}."
            idx = lower_name.find(token)
            if idx != -1:
                return file_name[:idx].rstrip(" _-")

        token = f"{key}."
        idx = lower_name.find(token)
        if idx != -1:
            return file_name[:idx].rstrip(" _-")

        return None
    # Find all .tif images recursively
    tif_paths = list(root_dir.rglob("*.tif")) + list(root_dir.rglob("*.TIF"))
    
    if not tif_paths:
        return [], []
    
    # Organize images by key and base name
    matched_by_key: Dict[str, Dict[str, Path]] = {key: {} for key in keys}
    unmatched: List[Path] = []
    subjects = 0
    for path in tif_paths:
        matched = False
        
        # Try to match with each key in order
        for key in keys:
            if key.lower() in path.name.lower():
                # Extract base name by removing the key
                base_name = _extract_base_name(path.name, key)
                if base_name:
                    matched_by_key[key][base_name] = path
                    matched = True
                    break
        
        if not matched:
            unmatched.append(path)
            subjects += 1
            if max_subjects is not None and subjects >= max_subjects:
                break
            
    
    # Find common base names across all keys
    if not keys:
        return unmatched, []
    
    # Get base names that exist for all keys
    common_bases = set(matched_by_key[keys[0]].keys())
    for key in keys[1:]:
        common_bases &= set(matched_by_key[key].keys())
    
    # Build paired tuples
    paired: List[Tuple[Path, ...]] = []
    for base_name in sorted(common_bases):
        pair = tuple(matched_by_key[key][base_name] for key in keys)
        paired.append(pair)
        subjects += 1
        if max_subjects is not None and subjects >= max_subjects:
            break
        
        
    
    # # Add unpaired images (matched a key but no complete set) to unmatched
    # paired_bases = set(base_name for base_name in common_bases)
    # for key in keys:
    #     for base_name, path in matched_by_key[key].items():
    #         if base_name not in paired_bases:
    #             unmatched.append(path)
    
    return unmatched, paired


def get_scale_um_per_px(image_path: Path, default_scale_value: float, scale_keys: list[str], scale_values: list[float]) -> float:
    matched_value = default_scale_value
    for key, value in zip(scale_keys, scale_values):
            if key.lower() in image_path.name.lower():
                matched_value = value
                break
    return matched_value
# endregion helper funcs

# region main function
def main(
    use_cluster: bool = USE_CLUSTER,
    n_workers: Optional[int] = N_WORKERS,
    threads_per_worker: Optional[int] = THREADS_PER_WORKER,
    save_images: bool = SAVE_IMAGES,
    force_respawn_cluster: bool = FORCE_RESPAWN_CLUSTER,
    debug: bool = DEBUG,
    max_subjects: Optional[int] = MAX_SUBJECTS,
    blob_size_um: float = BLOB_SIZE_UM,
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
    # region cluster setup
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
                    desired_n_workers = max(1, n_cpus // 2)
                
                if desired_threads_per_worker is None:
                    # Calculate threads to use all CPUs
                    desired_threads_per_worker = max(1, n_cpus // desired_n_workers)
                
                # Try to connect to existing cluster first
                existing_client = None
                needs_respawn = False
                
                try:
                    existing_client = Client(address="tcp://127.0.0.1:45693", timeout='2s')  # Try to connect to default scheduler
                    scheduler_info = existing_client.scheduler_info()
                    workers_info = scheduler_info['workers']
                    actual_n_workers = scheduler_info["n_workers"]
                    
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
                        print(f"\n[x]  Cluster parameters don't match desired configuration:")
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
    # endregion cluster setup

    # region build graph
    # Setup results directory (after cluster connection)

    results_dir: Path = setup_results_dir(SCRIPT_DIR, exp_name="simple_dask")
    print(f"[x] Results will be saved to: {results_dir.resolve()}\n")
    # Find subject image sets

    unmatched, pairs = find_tif_images_by_keys(
        IMAGE_BASE_DIR,
        keys=["_RFP", "_DAPI"],

        max_subjects=max_subjects,
    )


    image_bags = {}
  
    seperate_channels_bag = db.from_sequence(pairs).map(
        lambda x:dict(
            paths=x,
            rfp=(imread(x[0])[...,0]).squeeze(),
            dapi=(imread(x[1])[...,2]).squeeze(),
            source_type="separate_channels",
        ))

    combined_channels_bag = db.from_sequence(unmatched).map(
        lambda x:dict(
            paths=[x],
            image=imread(x),
        )).map( #TODO: prob need to add in remove rectangles
            lambda x:dict(
                paths=x["paths"],
                rfp=x["image"][...,0].squeeze(),
                dapi=x["image"][...,2].squeeze(),
                source_type="combined_channels",
            )
        )
    if debug:
        print(f"[x] Created Dask bag with:\n\t {combined_channels_bag.count().compute()} subjects from combined channels\n\t and {seperate_channels_bag.count().compute()} subjects from seperate channels.\n")

  
    full_bag = db.concat([seperate_channels_bag, combined_channels_bag])
    full_bag = full_bag.map( #TODO should add a propagate old keys func
        lambda x: x | dict(
            # Add more processing steps here as needed
            scale_um_per_px=get_scale_um_per_px(
                image_path=x["paths"][0],
                default_scale_value=0.4476,
                scale_keys=["40x"],
                scale_values=[0.2253],
            )

        )).map(lambda x: x |dict(
            crypt_labels= segment_crypts(
                channels=(x["rfp"], x["dapi"] ),
                microns_per_px=x["scale_um_per_px"],
                blob_size_um=blob_size_um,
                debug=False,
                max_regions=5)
        )).map(
            lambda x: x |dict(
                normalized_rfp=compute_normalized_rfp(
                    rfp_image=x["rfp"],
                    dapi_image=x["dapi"],
                    crypt_labels=x["crypt_labels"],
                )
            )
        ).map(
            lambda x: x | dict(
                summary_image=summarize_crypt_fluorescence(
                    normalized_rfp=x["normalized_rfp"],
                    crypt_labels=x["crypt_labels"],
                    microns_per_px=x["scale_um_per_px"],
                ),
                per_crypt_df=summarize_crypt_fluorescence_per_crypt(
                    normalized_rfp=x["normalized_rfp"],
                    crypt_labels=x["crypt_labels"],
                    microns_per_px=x["scale_um_per_px"],
                    subject_name=x["paths"][0].stem,
                )
            )
        )

    # end region build graph

    # region execute graph
    if debug:
        print(f"[x] Executing Dask bag with {full_bag.count().compute()} subjects...\n")

    results = full_bag.compute()
    if debug:
        print(f"\n[x] Completed processing all subjects.\n")




# endregion main function
   




if __name__ == "__main__":
    main()