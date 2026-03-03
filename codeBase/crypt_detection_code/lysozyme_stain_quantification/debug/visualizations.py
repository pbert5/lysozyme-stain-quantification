"""Utilities for generating visual debugging artifacts for crypt segmentation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.segmentation import find_boundaries


def generate_standard_visualization(
    rgb_img: np.ndarray,
    labels: np.ndarray,
    base_name: str,
) -> Dict[str, object]:
    """Return the default overlay plus metadata for a labeled image."""
    boundaries = find_boundaries(labels, mode="inner")
    overlay = rgb_img.copy()
    overlay[boundaries] = [255, 0, 0]
    return {
        "standard_overlay": overlay,
        "num_regions": int(len(np.unique(labels)) - 1),
        "base_name": base_name,
    }


def generate_debug_visuals(
    rgb_img: np.ndarray,
    initial_labels: np.ndarray,
    selected_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return diagnostic overlays comparing pre- and post-selection labels."""
    initial_boundaries = find_boundaries(initial_labels, mode="inner")
    initial_overlay = rgb_img.copy()
    initial_overlay[initial_boundaries] = [255, 255, 0]

    selected_boundaries = find_boundaries(selected_labels, mode="inner")
    selected_overlay = rgb_img.copy()
    selected_overlay[selected_boundaries] = [255, 0, 0]

    combined_overlay = rgb_img.copy()
    combined_overlay[initial_boundaries] = [255, 255, 0]
    combined_overlay[selected_boundaries] = [255, 0, 0]

    return {
        "initial_overlay": initial_overlay,
        "selected_overlay": selected_overlay,
        "combined_overlay": combined_overlay,
    }
