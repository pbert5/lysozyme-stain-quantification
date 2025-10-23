# region imports
from __future__ import annotations
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import dask
import dask.array as da
import dask.bag as db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask_image.imread import imread


try:
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
    from src.lysozyme_stain_quantification.utils.remove_artifacts import (
        remove_rectangular_artifacts,
    )
except ImportError:
    try:
        from scientific_image_finder.finder import find_subject_image_sets
        from lysozyme_stain_quantification.segment_crypts import segment_crypts
        from lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
        from lysozyme_stain_quantification.quantify.crypt_fluorescence_summary import (
            summarize_crypt_fluorescence,
            summarize_crypt_fluorescence_per_crypt,
            SUMMARY_FIELD_ORDER,
            PER_CRYPT_FIELD_ORDER,
        )
        from lysozyme_stain_quantification.utils.setup_tools import setup_results_dir, plot_all_crypts
        from lysozyme_stain_quantification.utils.remove_artifacts import (
            remove_rectangular_artifacts,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("  Ensure that 'scientific_image_finder' and 'lysozyme_stain_quantification' are installed and accessible.")
        raise e
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
import logging
try:
    logging.getLogger("src.lysozyme_stain_quantification.normalize_rfp").setLevel(logging.ERROR)
except Exception:
    logging.getLogger("lysozyme_stain_quantification.normalize_rfp").setLevel(logging.ERROR)
    

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
MAX_SUBJECTS = None           # Limit number of subjects (None = process all)


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
) -> Tuple[List[Path], List[Tuple[Path, ...]], List[str], List[str]]:
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
    Tuple[List[Path], List[Tuple[Path, ...]], List[str], List[str]]
        (unmatched_paths, paired_paths, paired_subject_names, unmatched_subject_names) where:
        - unmatched_paths: List of .tif images that don't match any key
        - paired_paths: List of tuples, each containing paths in key order
          that share the same base name
        - paired_subject_names: Subject labels for paired_paths (same order)
        - unmatched_subject_names: Subject labels for unmatched_paths (same order)
        
    Example
    -------
    >>> unmatched, pairs, pair_names, unmatched_names = find_tif_images_by_keys(
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
    from datetime import datetime, timezone
    _TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

    def _make_subject_label(base_name: str, subdir_label: str, paths: List[Path], existing: set[str]) -> str:
        label = base_name.strip()
        if subdir_label:
            label = f"{label} [{subdir_label}]"
        if label in existing:
            ts = min(p.stat().st_mtime for p in paths)
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(_TIMESTAMP_FMT)
            base_with_ts = f"{label} [{ts_str}]"
            candidate = base_with_ts
            suffix = 2
            while candidate in existing:
                candidate = f"{base_with_ts} ({suffix})"
                suffix += 1
            label = candidate
        return label

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
    
    # Build paired tuples and subject names
    paired: List[Tuple[Path, ...]] = []
    paired_subject_names: List[str] = []
    used_labels: set[str] = set()
    for base_name in sorted(common_bases):
        pair = tuple(matched_by_key[key][base_name] for key in keys)
        paired.append(pair)
        # Prefer subdir relative to root if both in same folder; else choose first's
        rel_dirs = []
        for p in pair:
            try:
                rel_dirs.append(str(p.parent.relative_to(root_dir)))
            except Exception:
                rel_dirs.append("")
        # pick most common non-empty or empty
        subdir_label = ""
        non_empty = [d for d in rel_dirs if d]
        if non_empty:
            # if both equal, take it; else pick lexicographically for determinism
            subdir_label = sorted(non_empty)[0] if len(set(non_empty)) > 1 else non_empty[0]
        paired_subject_names.append(
            _make_subject_label(base_name, subdir_label, list(pair), used_labels)
        )
        used_labels.add(paired_subject_names[-1])
        subjects += 1
        if max_subjects is not None and subjects >= max_subjects:
            break
        
        
    
    # # Add unpaired images (matched a key but no complete set) to unmatched
    # paired_bases = set(base_name for base_name in common_bases)
    # for key in keys:
    #     for base_name, path in matched_by_key[key].items():
    #         if base_name not in paired_bases:
    #             unmatched.append(path)
    
    # Build subject names for unmatched (e.g., combined-channel images)
    unmatched_subject_names: List[str] = []
    for path in unmatched:
        # derive base by treating as generic (no key)
        base = _extract_base_name(path.name, "") or Path(path.name).stem
        try:
            subdir = str(path.parent.relative_to(root_dir))
        except Exception:
            subdir = ""
        label = _make_subject_label(base, subdir, [path], used_labels)
        unmatched_subject_names.append(label)
        used_labels.add(label)

    return unmatched, paired, paired_subject_names, unmatched_subject_names



def save_overlay_image(
    subject_name: Union[Path, str, Sequence[Union[Path, str]]],
    rfp_image: Union[np.ndarray, da.Array, Sequence[Union[np.ndarray, da.Array]]],
    dapi_image: Union[np.ndarray, da.Array, Sequence[Union[np.ndarray, da.Array]]],
    crypt_labels: Union[np.ndarray, da.Array, Sequence[Union[np.ndarray, da.Array]]],
    output_dir: Path,
    image_source_type: Union[str, Sequence[str], None] = None,
) -> Path:
    """Render and store a single subject overlay using render_label_overlay."""

    def _as_single(value: Union[Sequence, object], label: str):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{label} must describe exactly one subject (got {len(value)}).")
            return value[0]
        return value

  

    overlay_dir = output_dir / "renderings"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Convert potential dask arrays to numpy inside the task and min-max normalize to [0, 1]
    def _to_numpy_local(a: Union[np.ndarray, da.Array, Sequence[np.ndarray | da.Array]]) -> np.ndarray:
        if isinstance(a, da.Array):
            return a.compute()
        elif isinstance(a, (list, tuple)):
            return np.asarray(a)
        return np.asarray(a)

    def _minmax01(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        out = (arr - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    rfp_np = _minmax01(_to_numpy_local(rfp_image))
    dapi_np = _minmax01(_to_numpy_local(dapi_image))
    labels_np = _to_numpy_local(crypt_labels)

    overlay_xr = render_label_overlay(
        channels=[rfp_np, dapi_np, labels_np],
        fill_alpha=0.35,
        outline_alpha=1.0,
        outline_width=2,
        normalize_scalar=True,
    )
    overlay_rgb = np.moveaxis(overlay_xr.values, 0, -1)
    if isinstance(subject_name, (list, tuple)):
        safe_name = "_".join(str(s) for s in subject_name)
    elif isinstance(subject_name, (str, Path)):
        safe_name: str = (
            str(subject_name).replace("/", "_")
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            )
    output_path = overlay_dir / f"{safe_name}_{image_source_type}_overlay.png"
    plt.imsave(output_path, overlay_rgb)

    return output_path


def get_scale_um_per_px(image_path: Path, default_scale_value: float, scale_keys: list[str], scale_values: list[float]) -> float:
    matched_value = default_scale_value
    for key, value in zip(scale_keys, scale_values):
            if key.lower() in image_path.name.lower():
                matched_value = value
                break
    return matched_value



# endregion helper funcs

# region channel extraction helpers (no early compute)
def _identify_channel_axis_lite(arr_shape: Tuple[int, ...]) -> int | None:
    """Heuristic to locate channel axis for HxWxC or CxHxW images."""
    if len(arr_shape) < 3:
        return None
    # Prefer channels-last when small channel count
    if arr_shape[-1] <= 4 and (len(arr_shape) == 3 or arr_shape[-1] != arr_shape[-2]):
        return -1
    # Fallback to channels-first when first dim looks like channels
    if arr_shape[0] <= 4 and arr_shape[0] != arr_shape[1]:
        return 0
    return None


def _moveaxis_like(arr: Union[np.ndarray, da.Array], source: int, dest: int):
    return da.moveaxis(arr, source, dest) if isinstance(arr, da.Array) else np.moveaxis(arr, source, dest)


def _squeeze_like(arr: Union[np.ndarray, da.Array]):
    return arr.squeeze()


def _to_2d_channel(arr: Union[np.ndarray, da.Array], preferred_index: int = 0) -> Union[np.ndarray, da.Array]:
    """Return a 2D view of arr, selecting a channel if needed (no compute)."""
    # Already 2D
    if hasattr(arr, "ndim") and arr.ndim == 2:
        return arr
    arr_sq = _squeeze_like(arr)
    if hasattr(arr_sq, "ndim") and arr_sq.ndim == 2:
        return arr_sq
    shape = getattr(arr_sq, "shape", None)
    if shape is None:
        return arr_sq
    if len(shape) == 3:
        axis = _identify_channel_axis_lite(shape)
        if axis is None:
            axis = -1  # assume channels-last
        arr_ch_last = _moveaxis_like(arr_sq, axis, -1)
        channels = arr_ch_last.shape[-1]
        # Choose a safe index
        idx = preferred_index if isinstance(channels, int) and channels > preferred_index else 0
        return arr_ch_last[..., idx]
    return arr_sq


def _is_2d(arr: Union[np.ndarray, da.Array]) -> bool:
    return getattr(arr, "ndim", None) == 2
# endregion channel extraction helpers

# region artifact removal helper (executed lazily on workers)
def _remove_rectangles_for_combined(image: Union[np.ndarray, da.Array]) -> np.ndarray:
    # Ensure numpy array for OpenCV operations inside remove_rectangular_artifacts
    np_img = image.compute() if isinstance(image, da.Array) else np.asarray(image)
    cleaned = remove_rectangular_artifacts(channels=[np_img])
    return cleaned
# endregion artifact removal helper

# endregion channel extraction helpers

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
    connect_to_existing_cluster: bool = False,
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
                    desired_n_workers = max(1, n_cpus // 2 - 2)
                
                if desired_threads_per_worker is None:
                    # Calculate threads to use all CPUs
                    desired_threads_per_worker = max(1, n_cpus // desired_n_workers)
                
                # Try to connect to existing cluster first
                existing_client = None
                needs_respawn = False

                if connect_to_existing_cluster:
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
                        silence_logs=logging.FATAL
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

    # region build bag graph
    # Setup results directory (after cluster connection)

    results_dir: Path = setup_results_dir(SCRIPT_DIR, exp_name="simple_dask")
    print(f"[x] Results will be saved to: {results_dir.resolve()}\n")
    # Find subject image sets

    unmatched, pairs, paired_subject_names, unmatched_subject_names = find_tif_images_by_keys(
        IMAGE_BASE_DIR,
        keys=["_RFP", "_DAPI"],

        max_subjects=max_subjects,
    )


    image_bags = {}
    # region build bags
    seperate_channels_bag = db.from_sequence(list(zip(pairs, paired_subject_names))).map(
        lambda p: dict(
            paths=p[0],
            rfp=_to_2d_channel(imread(p[0][0]), preferred_index=0),
            dapi=_to_2d_channel(imread(p[0][1]), preferred_index=2),
            source_type="separate_channels",
            subject_name=p[1],
        )
    )

    combined_channels_bag = (
        db.from_sequence(list(zip(unmatched, unmatched_subject_names)))
        .map(lambda p: dict(paths=[p[0]], image=imread(p[0]), subject_name=p[1]))
        .map(lambda x: x | {"image": _remove_rectangles_for_combined(x["image"])})
        .map(
            lambda x: dict(
                paths=x["paths"],
                rfp=_to_2d_channel(x["image"], preferred_index=0),
                dapi=_to_2d_channel(x["image"], preferred_index=2),
                source_type="combined_channels",
                subject_name=x["subject_name"],
            )
        )
    )
    if debug:
        print(f"[x] Created Dask bag with:\n\t {len(unmatched)} subjects from combined channels\n\t and {len(paired_subject_names)} subjects from seperate channels.\n")

    
    full_bag = db.concat([seperate_channels_bag, combined_channels_bag])
    # Filter out any subjects that did not yield 2D RFP/DAPI (no early compute)
    full_bag = full_bag.filter(lambda x: _is_2d(x["rfp"]) and _is_2d(x["dapi"]))

    # endregion build bags
    # region map ops


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
                ))
        ).map(
        lambda x: x | dict(
            per_crypt_df=summarize_crypt_fluorescence_per_crypt(
                normalized_rfp=x["normalized_rfp"],
                crypt_labels=x["crypt_labels"],
                microns_per_px=x["scale_um_per_px"],
                subject_name=x.get("subject_name", Path(x["paths"][0]).stem),
            )
        )
    )
    
    # region map overlay saves

    if save_images:
        full_bag = full_bag.map(
            lambda x: x
            | dict(
                overlay_paths=[
                    save_overlay_image(
                        subject_name=x["subject_name"],
                        rfp_image=x["rfp"],
                        dapi_image=x["dapi"],
                        crypt_labels=x["crypt_labels"],
                        output_dir=results_dir,
                        image_source_type=x["source_type"],
                    )
                ]
            )
        )
    else:
        full_bag = full_bag.map(lambda x: x | dict(overlay_paths=[]))
    # endregion map overlay saves
    # region prune heavy
    # Prune heavy arrays before returning to the driver; keep only lightweight results
    def _prune_heavy(x: Dict[str, object]) -> Dict[str, object]:
        return {
            "subject_name": x["subject_name"],
            "source_type": x["source_type"],
            "microns_per_px": x["scale_um_per_px"],
            "summary_image": x["summary_image"],
            "per_crypt": x["per_crypt_df"],
            "overlay_paths": x.get("overlay_paths", []),
        }

    full_bag = full_bag.map(_prune_heavy)
    # endregion prune heavy

    # end region build bag graph

    # region execute graph
    if debug:
        print(f"[x] Executing Dask bag with {len(unmatched) + len(paired_subject_names)} subjects...\n")

    start_time = time.perf_counter()
    results = full_bag.compute()
    elapsed = time.perf_counter() - start_time
    print(f"[x] Compute took {elapsed:.2f} seconds")
    if debug:
        print(f"\n[x] Completed processing all subjects.\n")

    # endregion execute graph
    # Cleanup cluster if needed
    cluster_context.close() if cluster_context is not None else None
    # region save stats

    # Collate lightweight results into two CSVs
    image_summary_records: List[Dict[str, object]] = []
    per_crypt_records: List[Dict[str, object]] = []

    for item in results:
        name = str(item.get("subject_name", ""))
        source_type = str(item.get("source_type", "unknown"))
        scale = item.get("microns_per_px", None)
        summary = item.get("summary_image", {})
        per_crypt = item.get("per_crypt", {})

        # Image-level record
        row: Dict[str, object] = {
            "subject_name": name,
            "image_source_type": source_type,
            "microns_per_px": scale,
        }
        if isinstance(summary, dict):
            for field in SUMMARY_FIELD_ORDER:
                row[field] = summary.get(field, float("nan"))
        image_summary_records.append(row)

        # Per-crypt records
        if isinstance(per_crypt, dict):
            records = per_crypt.get("records", [])
            if isinstance(records, list):
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    rec_out = dict(rec)
                    rec_out["subject_name"] = rec_out.get("subject_name", name)
                    rec_out["image_source_type"] = source_type
                    per_crypt_records.append(rec_out)

    # Build DataFrames and save
    image_summary_df = pd.DataFrame(image_summary_records)
    if not image_summary_df.empty:
        cols = ["subject_name", "image_source_type", "microns_per_px"] + list(SUMMARY_FIELD_ORDER)
        image_summary_df = image_summary_df.reindex(columns=[c for c in cols if c in image_summary_df.columns])

    per_crypt_df = pd.DataFrame(per_crypt_records)
    if not per_crypt_df.empty:
        cols_pc = list(PER_CRYPT_FIELD_ORDER)
        if "image_source_type" not in cols_pc:
            cols_pc.insert(1, "image_source_type")
        per_crypt_df = per_crypt_df.reindex(columns=[c for c in cols_pc if c in per_crypt_df.columns])

    print("\n[Saving] Writing aggregated CSVs...")
    img_csv = results_dir / "simple_dask_image_summary.csv"
    pc_csv = results_dir / "simple_dask_per_crypt.csv"
    if not image_summary_df.empty:
        image_summary_df.to_csv(img_csv, index=False)
        if debug:
            print(f"  Saved: {img_csv}")
    else:
        if debug:
            print("  No image summary rows to save.")

    if not per_crypt_df.empty:
        per_crypt_df.to_csv(pc_csv, index=False)
        if debug:
            print(f"  Saved: {pc_csv}")
    else:
        if debug:
            print("  No per-crypt rows to save.")
    # endregion save stats




# endregion main function
   




if __name__ == "__main__":
    main()
