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


# =============================================================================
# CONFIGURATION - Edit these defaults for quick runs without command-line args
# =============================================================================

USE_CLUSTER = True              # Use distributed Dask cluster for parallel processing
FORCE_RESPAWN_CLUSTER = True   # Force respawn cluster if params don't match (WARNING: closes existing cluster)
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
BLOB_SIZE_UM = 50.0 * 0.4476
# =============================================================================



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
        
        # Apply max_subjects limit
        if max_subjects is not None and len(paired) >= max_subjects:
            break
    
    # Add unpaired images (matched a key but no complete set) to unmatched
    paired_bases = set(base_name for base_name in common_bases)
    for key in keys:
        for base_name, path in matched_by_key[key].items():
            if base_name not in paired_bases:
                unmatched.append(path)
    
    return unmatched, paired


def get_scale_um_per_px(image_path: Path, default_scale_value: float, scale_keys: list[str], scale_values: list[float]) -> float:
    matched_value = default_scale_value
    for key, value in zip(scale_keys, scale_values):
            if key.lower() in image_path.name.lower():
                matched_value = value
                break
    return matched_value


def main(
    use_cluster: bool = USE_CLUSTER,
    n_workers: Optional[int] = N_WORKERS,
    threads_per_worker: Optional[int] = THREADS_PER_WORKER,
    save_images: bool = SAVE_IMAGES,
    debug: bool = DEBUG,
    max_subjects: Optional[int] = MAX_SUBJECTS,
    blob_size_um: float = BLOB_SIZE_UM,
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

            cluster_context = None  # Store cluster reference to prevent garbage collection
            
            
            print(f"\nStarting new Dask cluster...")
            
            # Memory per worker: total RAM / workers (with safety margin)
            # Assume 4GB per worker as safe default, can be increased
            memory_per_worker = "4GB"
            
            if debug:
                print(f"  Detected {n_cpus} CPUs")
                print(f"  Configuring: {desired_n_workers} workers × {desired_threads_per_worker} threads = {desired_n_workers * desired_threads_per_worker} total threads")
            
            cluster = LocalCluster(
                n_workers=10,
                threads_per_worker=desired_threads_per_worker,
                memory_limit=memory_per_worker,
                dashboard_address=":8787",
            )
            client = Client(cluster)
            #cluster_context = cluster  # Store for cleanup
            
            # Verify the cluster was created with correct parameters
            
            
            print(f"  Scheduler: {cluster.scheduler_address}")
            print(f"  Dashboard: {cluster.dashboard_link}")
            
            
            
            
            print(f"\n  📊 MONITOR: {cluster.dashboard_link}")
            print()
                
    
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
            source_type="separate_channels",
        ))
    combined_channels_bag = db.from_sequence(unmatched).map(
        lambda x:dict(
            paths=[x],
            image=imread(x).squeeze(),
        )).map( #TODO: prob need to add in remove rectangles
            lambda x:dict(
                paths=x["paths"],
                rfp=x["image"],
                dapi=x["image"],
                source_type="combined_channels",
            )
        )

  
    full_bag = db.concat([seperate_channels_bag, combined_channels_bag])
    full_bag = full_bag.map( #TODO should add a propagate old keys func
        lambda x: dict(
            paths=x["paths"],
            rfp=x["rfp"],
            dapi=x["dapi"],
            source_type=x["source_type"],
            # Add more processing steps here as needed
            scale_um_per_px=get_scale_um_per_px(
                image_path=x["paths"][0],
                default_scale_value=0.4476,
                scale_keys=["40x"],
                scale_values=[0.2253],
            )

        )).map(lambda x: dict(
            paths=x["paths"],
            rfp=x["rfp"],
            dapi=x["dapi"],
            source_type=x["source_type"],
            scale_um_per_px=x["scale_um_per_px"],
            crypt_labels= segment_crypts(
                channels=(x["rfp"], x["dapi"] ),
                microns_per_px=x["scale_um_per_px"],
                blob_size_um=blob_size_um,
                debug=False,
                max_regions=5)
        )).compute()










   




if __name__ == "__main__":
    main()