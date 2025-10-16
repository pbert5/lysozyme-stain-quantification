"""
Primary script to run crypt segmentation on lysozyme stain images and save plots.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from image_ops_framework.analysis_stack_xr import AnalysisStackXR
from image_ops_framework.helpers import render_label_overlay
from src.scientific_image_finder.finder import find_subject_image_sets
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from src.lysozyme_stain_quantification.utils.subject_scale_lookup import (
    subject_scale_from_name,
)
from src.lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
from src.lysozyme_stain_quantification.quantify.crypt_fluorescence_summary import (
    SUMMARY_FIELD_ORDER,
    PER_CRYPT_FIELD_ORDER,
    summarize_crypt_fluorescence,
    summarize_crypt_fluorescence_per_crypt,
)
from src.lysozyme_stain_quantification.utils.setup_tools import setup_results_dir, plot_all_crypts

DEBUG = True
MAX_SUBJECTS = 10
SAVE_IMAGES = False  # whether to save overlay images




def main() -> None:
    results_dir = setup_results_dir(SCRIPT_DIR)
    render_dir = results_dir / "renderings"
    render_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path("/home/phillip/documents/lysozyme/lysozyme images")
    lysozyme_channel = "rfp"
    dapi_channel = "dapi"

    subject_names, images_by_source, source_names = find_subject_image_sets(
        img_dir=img_dir,
        sources=[("rfp", lysozyme_channel, "r"), ("dapi", dapi_channel, "b")],
        max_subjects=MAX_SUBJECTS,
    )

    scale_keys = ["40x"]
    scale_values = [0.2253]
    default_scale_value = 0.4476

    if DEBUG:
        print(f"Sources: {source_names}")
        print(f"Found {len(subject_names)} subjects with images in both channels.")
        for subject, red_img, blue_img in zip(subject_names, images_by_source[0], images_by_source[1]):
            print(f"Subject: {subject}, Red shape: {red_img.shape}, Blue shape: {blue_img.shape}")

    blob_size_um = 18.0  # approximate crypt size in microns
    if DEBUG:
        print(f"Using blob size (crypt size) of {blob_size_um} microns.")
        print("Starting analysis stack...")
    stk = AnalysisStackXR().add_sources(subject=subject_names, sources=images_by_source, sourcenames=source_names)
    if DEBUG:
        # let knwo added sources
        print(f"Added sources")

    stk = stk.run(
        subject_scale_from_name,
        channels=["subject_name"],
        output_name="microns_per_px",
        keys=scale_keys,
        values=scale_values,
        default=default_scale_value,
    )
    if DEBUG:
        print(f"[END] Computed microns_per_px for subjects")
        print(f"[BEGIN] segmentation")
    stk = stk.run(
        segment_crypts,
        channels=["rfp", "dapi", "microns_per_px"], #TODO: its prob super slow bc microns_per_px is not chuncked, need to improve handeling  of run to ensuer everyting passed is chunked
        output_name="crypts",
        # use_dask=True,
        blob_size_um=blob_size_um,
    )
    if DEBUG:
        print(f"[END] segmentation")
        print(f"[BEGIN] Computing normalized rfp")
    stk = stk.run(
        compute_normalized_rfp,
        channels=["rfp", "dapi", "crypts"],
        output_name="normalized_rfp",
        
    )
    if DEBUG:
        print(f"[END] Computed normalized rfp")
        print(f"[BEGIN] Summarizing crypt fluorescence")
    stk = stk.run(
        summarize_crypt_fluorescence,
        channels=["normalized_rfp", "crypts", "microns_per_px"],
        output_name="crypt_fluorescence_summary",
        intensity_upper_bound=1,
    )
    if DEBUG:
        print(f"[END] Summarized crypt fluorescence")
        print(f"[BEGIN] Summarizing per-crypt fluorescence details")
    stk = stk.run(
        summarize_crypt_fluorescence_per_crypt,
        channels=["normalized_rfp", "crypts", "microns_per_px"],
        output_name="crypt_fluorescence_per_crypt",
        use_apply_ufunc=True,
        vectorize=False, #TODO: what is this doing? 
    )
    if DEBUG:
        print(f"[END] Summarized per-crypt fluorescence details")
    if SAVE_IMAGES:
        if DEBUG:
            print(f"[BEGIN] Rendering overlay images")
        stk.run(
            render_label_overlay,
            channels=["rfp", "dapi", "crypts"],
            output_name="crypt_overlay",
            outline_width=2,
            fill_alpha=0.35,
            outline_alpha=0.9,
            normalize_scalar=True,
        )
        if DEBUG:
            print(f"[END] Rendered overlay images")

    print(stk)
    if DEBUG:
        print(f"[BEGIN] Saving results to {results_dir}")
    ds = stk.output(format="dataset")

    if SAVE_IMAGES:
        overlay_exports = stk.save_images(
            channels=["crypt_overlay"],
            directory=render_dir,
            normalize={"crypt_overlay": False},
            image_format="png",
        )
    if DEBUG and SAVE_IMAGES:
        print(f"Saved {len(overlay_exports.get('crypt_overlay', []))} overlay images to {render_dir}")
    
    if SAVE_IMAGES:
        fig = plot_all_crypts(ds, lab_name="crypts", ncols=6, figsize_per=(3, 3))
    if SAVE_IMAGES:
        output_path = results_dir / "karen_detect_crypts.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if DEBUG and SAVE_IMAGES:
        print(f"[END] Saved visualization to {output_path}")


    # Save crypt fluorescence summary as CSV
    if DEBUG:
        print("[BEGIN] Saving crypt fluorescence summary...")
    df = stk.to_dataframe(channels=["subject_name","crypt_fluorescence_summary","microns_per_px"], columns=["subject_name", list(SUMMARY_FIELD_ORDER), "microns_per_px"])
    df.to_csv(results_dir / "karen_detect_crypts.csv", index=False)
    if DEBUG:
        print(f"[END] Saved crypt fluorescence summary to {results_dir / 'karen_detect_crypts.csv'}")

    # Save per-crypt detail table as CSV
    if DEBUG:
        print("[BEGIN] Saving per-crypt detail summary to csv...")
    per_crypt_da = ds["crypt_fluorescence_per_crypt"]
    per_crypt_records = []
    numeric_fields = list(per_crypt_da.coords["field"].values)
    record_counts = per_crypt_da.coords["record_count"].values if "record_count" in per_crypt_da.coords else np.zeros(per_crypt_da.sizes["subject"], dtype=int)
    subject_display = per_crypt_da.coords.get("subject_display_name", per_crypt_da.coords["subject"]).values
    int_fields = {"crypt_label", "crypt_index", "pixel_area"}
    for subj_idx, subject in enumerate(per_crypt_da.coords["subject"].values):
        count = int(record_counts[subj_idx]) if record_counts.size else 0
        if count <= 0:
            continue
        data = per_crypt_da.isel(subject=subj_idx).values  # shape (max_records, len(fields))
        for rec_idx in range(count):
            row = {"subject_name": subject_display[subj_idx]}
            for field_idx, field in enumerate(numeric_fields):
                value = data[rec_idx, field_idx]
                if np.isnan(value):
                    row[field] = np.nan
                elif field in int_fields:
                    row[field] = int(round(value))
                else:
                    row[field] = float(value)
            per_crypt_records.append(row)
    per_crypt_df = pd.DataFrame(per_crypt_records, columns=list(PER_CRYPT_FIELD_ORDER))
    per_crypt_df.to_csv(results_dir / "karen_detect_crypts_per_crypt.csv", index=False)
    if DEBUG:
        print(f"[END] Saved per-crypt detail summary to {results_dir / 'karen_detect_crypts_per_crypt.csv'}")

if __name__ == "__main__":
    main()
