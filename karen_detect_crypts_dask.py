"""
Pure Dask implementation of lysozyme crypt detection pipeline.
Builds the entire computation graph lazily, then computes all at once.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import dask
import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import tifffile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scientific_image_finder.finder import find_subject_image_sets
from lysozyme_stain_quantification.segment_crypts import segment_crypts
from lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
from lysozyme_stain_quantification.quantify.crypt_fluorescence_summary import (
    summarize_crypt_fluorescence,
    summarize_crypt_fluorescence_per_crypt,
)


def array_to_dask_lazy(arr: np.ndarray) -> da.Array:
    """Convert a numpy array to a dask array without copying (lazy reference)."""
    return da.from_array(arr, chunks=arr.shape)


def main():
    print("=" * 80)
    print("DASK-BASED LYSOZYME CRYPT DETECTION PIPELINE")
    print("=" * 80)
    
    # Configuration
    IMAGE_BASE_DIR = Path("lysozyme images")
    BLOB_SIZE_UM = 50.0
    MAX_SUBJECTS = 10
    
    # Find and organize images
    print("\n[1/5] Finding images...")
    
    # Try combined-channel images first
    subject_names, combined_sources, _ = find_subject_image_sets(
        img_dir=IMAGE_BASE_DIR,
        sources=[("combined", "")],
        max_subjects=MAX_SUBJECTS,
    )
    
    combined_images = combined_sources[0] if combined_sources else []
    
    def _has_multi_channel(arr: np.ndarray) -> bool:
        array = np.asarray(arr)
        return array.ndim >= 3 and (array.shape[-1] >= 3 or array.shape[0] >= 3)
    
    use_combined = bool(combined_images) and all(_has_multi_channel(img) for img in combined_images)
    
    if use_combined:
        print("Detected combined-channel images; extracting red and blue channels.")
        
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
        
        rfp_images: list[np.ndarray] = []
        dapi_images: list[np.ndarray] = []
        for img in combined_images:
            red, blue = _split_combined_image(img)
            rfp_images.append(red)
            dapi_images.append(blue)
        
        source_names = ["rfp", "dapi"]
    else:
        print("Combined-channel images not found; falling back to per-channel search.")
        subject_names, images_by_source, source_names = find_subject_image_sets(
            img_dir=IMAGE_BASE_DIR,
            sources=[("rfp", "rfp", "r"), ("dapi", "dapi", "b")],
            max_subjects=MAX_SUBJECTS,
        )
        rfp_images = images_by_source[0]
        dapi_images = images_by_source[1]
    
    print(f"Sources: {source_names}")
    print(f"Found {len(subject_names)} subjects with images in both channels.")
    
    # Print subject info
    for i, (name, rfp, dapi) in enumerate(zip(subject_names, rfp_images, dapi_images)):
        print(f"Subject: {name}, Red shape: {rfp.shape}, Blue shape: {dapi.shape}")
    
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
    
    print(f"\nUsing blob size (crypt size) of {BLOB_SIZE_UM} microns.")
    
    # Build the lazy computation graph for all subjects
    print("\n[2/5] Building dask computation graph...")
    
    # Dictionary to store delayed computations for each subject
    subject_computations: Dict[str, Dict] = {}
    
    for subject_name, rfp_np, dapi_np, scale in zip(subject_names, rfp_images, dapi_images, microns_per_px_values):
        print(f"  Adding subject: {subject_name}")
        
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
                channels=[norm_rfp, labels, scale_val]
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
        }
    
    print(f"\n✓ Graph built for {len(subject_computations)} subjects")
    print("  (No computation has occurred yet - everything is lazy!)")
    
    # Now compute everything at once!
    print("\n[3/5] Computing all results (this may take a while)...")
    print("  Dask will optimize the computation graph and execute in parallel.")
    
    # Gather all the delayed objects we want to compute
    image_summaries = [comp["image_summary"] for comp in subject_computations.values()]
    per_crypt_summaries = [comp["per_crypt_summary"] for comp in subject_computations.values()]
    
    # Compute all at once!
    with dask.config.set(scheduler='threads', num_workers=4):
        print("  Computing image summaries...")
        image_results = dask.compute(*image_summaries)
        
        print("  Computing per-crypt summaries...")
        per_crypt_results = dask.compute(*per_crypt_summaries)
    
    print("\n✓ All computations complete!")
    
    # Combine results
    print("\n[4/5] Combining results...")
    
    # Combine image summaries
    image_df_list = []
    for result in image_results:
        if result is not None and not result.empty:
            image_df_list.append(result)
    
    if image_df_list:
        combined_image_summary = pd.concat(image_df_list, ignore_index=True)
        print(f"  Image summary: {len(combined_image_summary)} rows")
    else:
        combined_image_summary = pd.DataFrame()
        print("  Image summary: empty")
    
    # Combine per-crypt summaries
    per_crypt_df_list = []
    for result in per_crypt_results:
        if result is not None and not result.empty:
            per_crypt_df_list.append(result)
    
    if per_crypt_df_list:
        combined_per_crypt_summary = pd.concat(per_crypt_df_list, ignore_index=True)
        print(f"  Per-crypt summary: {len(combined_per_crypt_summary)} rows")
    else:
        combined_per_crypt_summary = pd.DataFrame()
        print("  Per-crypt summary: empty")
    
    # Save results
    print("\n[5/5] Saving results...")
    output_dir = Path("results/dask")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not combined_image_summary.empty:
        image_output = output_dir / "image_summary.csv"
        combined_image_summary.to_csv(image_output, index=False)
        print(f"  Saved: {image_output}")
    
    if not combined_per_crypt_summary.empty:
        per_crypt_output = output_dir / "per_crypt_summary.csv"
        combined_per_crypt_summary.to_csv(per_crypt_output, index=False)
        print(f"  Saved: {per_crypt_output}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    # Print summary statistics
    if not combined_image_summary.empty:
        print("\nImage Summary Statistics:")
        print(combined_image_summary.describe())
    
    if not combined_per_crypt_summary.empty:
        print("\nPer-Crypt Summary Statistics:")
        print(combined_per_crypt_summary.describe())


if __name__ == "__main__":
    main()
