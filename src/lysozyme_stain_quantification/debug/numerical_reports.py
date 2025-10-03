"""Numerical reporting utilities for crypt segmentation."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from skimage.measure import regionprops

_DebugPrinter = Optional[Callable[[str], None]]


def generate_label_summary(
    labels: np.ndarray,
    red_img: np.ndarray,
    pixel_dim: float,
    background_tissue_intensity: float = 0.0,
    average_crypt_intensity: float = 0.0,
    debug: bool = False,
    debug_printer: _DebugPrinter = None,
) -> np.ndarray:
    """
    Vectorised summary of region-level measurements for selected crypt labels.
    """
    printer = debug_printer if debug_printer is not None else print
    props = regionprops(labels, intensity_image=red_img)

    if debug:
        unique_labels = np.unique(labels)
        printer(
            f"[SUMMARY DEBUG] Input labels: {len(unique_labels)} unique ({unique_labels})"
        )
        printer(f"[SUMMARY DEBUG] Region props found: {len(props)} regions")
        printer(
            f"[SUMMARY DEBUG] Red image for intensity: range [{red_img.min():.2f}, {red_img.max():.2f}]"
        )

    if not props:
        if debug:
            printer("[SUMMARY DEBUG] No regions to summarize, returning empty array")
        return np.empty((0, 9))

    summary_data = []
    for i, prop in enumerate(props):
        if prop.label == 0:
            if debug:
                printer("[SUMMARY DEBUG] Skipping background region (label 0)")
            continue

        pos_y, pos_x = prop.centroid
        area_pixels = prop.area
        area_um2 = area_pixels * (pixel_dim ** 2)
        red_intensity = prop.mean_intensity
        red_sum = red_intensity * area_pixels
        fluorescence = red_intensity * area_um2

        if debug and i < 5:
            printer(
                "[SUMMARY DEBUG] Region {}: area={}px ({:.2f}μm²), centroid=({:.1f},{:.1f}), red_sum={:.0f}, "
                "intensity={:.2f}, fluorescence={:.2f}".format(
                    prop.label,
                    area_pixels,
                    area_um2,
                    pos_x,
                    pos_y,
                    red_sum,
                    red_intensity,
                    fluorescence,
                )
            )

        summary_data.append(
            [
                prop.label,
                pos_x * pixel_dim,
                pos_y * pixel_dim,
                area_um2,
                red_sum,
                red_intensity,
                fluorescence,
                background_tissue_intensity,
                average_crypt_intensity,
            ]
        )

    if summary_data:
        result = np.array(summary_data)
        if debug:
            printer(f"[SUMMARY DEBUG] Final summary shape: {result.shape}")
            printer(
                f"[SUMMARY DEBUG] Background tissue intensity: {background_tissue_intensity:.3f}"
            )
            printer(
                f"[SUMMARY DEBUG] Average crypt intensity: {average_crypt_intensity:.3f}"
            )
        return result

    if debug:
        printer("[SUMMARY DEBUG] No regions to summarize, returning empty array")
    return np.empty((0, 9))
