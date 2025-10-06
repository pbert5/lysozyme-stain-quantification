from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None

import src.lysozyme_stain_quantification.segment_crypts as segment_module
from src.lysozyme_stain_quantification.crypts import identify_potential_crypts_mod as identify_module
from src.lysozyme_stain_quantification.crypts import remove_edge_touching_regions_mod as remove_module
from src.lysozyme_stain_quantification.crypts.scoring_selector_mod import scoring_selector
from tests.test_lysoszyme_stain_quantification.tools.demo_crypt_image_generator import generate_test_crypts

DEBUG_ROOT = os.environ.get("LYSOZYME_SEGMENT_DEBUG_DIR")
DEBUG_PATH = Path(DEBUG_ROOT) if DEBUG_ROOT else None

NUM_CRYPTS = 5
IMAGE_SIZE = (200, 200)

RED_CHANNEL, BLUE_CHANNEL, _ = generate_test_crypts(num_crypts=NUM_CRYPTS, image_size=IMAGE_SIZE)


def _patch_segment_dependencies() -> None:
    segment_module.identify_potential_crypts = identify_module.identify_potential_crypts
    segment_module.remove_edge_touching_regions_mod = remove_module.remove_edge_touching_regions_sk

    def _run_selector(
        labels: np.ndarray,
        raw_img: np.ndarray,
        *,
        debug: bool = False,
        max_regions=0,
        weights=None,
        return_details: bool = False,
    ):
        max_regions_int = max_regions if isinstance(max_regions, int) and max_regions > 0 else NUM_CRYPTS
        return scoring_selector(
            labels,
            raw_img,
            debug=debug,
            max_regions=max_regions_int,
            weights=weights,
            return_details=return_details,
        )

    segment_module.scoring_selector = _run_selector


def _maybe_save_debug_collage(
    name: str,
    *,
    red: np.ndarray,
    blue: np.ndarray,
    expected: np.ndarray,
    result: np.ndarray,
) -> None:
    if DEBUG_PATH is None or plt is None:
        return

    DEBUG_PATH.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(red, cmap="inferno")
    axes[0].set_title("Red Input")

    axes[1].imshow(blue, cmap="Blues")
    axes[1].set_title("Blue Input")

    axes[2].imshow(expected, cmap="gray")
    axes[2].set_title("Expected Mask")

    axes[3].imshow(result, cmap="gray")
    axes[3].set_title("Result Mask")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(DEBUG_PATH / f"{name}_collage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_segment_crypts_runs_at_all() -> None:
    _patch_segment_dependencies()

    initial_labels = identify_module.identify_potential_crypts(
        RED_CHANNEL, BLUE_CHANNEL, blob_size_px=15, debug=False
    )
    expected_mask = remove_module.remove_edge_touching_regions_sk(initial_labels) > 0

    segmented = segment_module.segment_crypts((RED_CHANNEL.copy(), BLUE_CHANNEL.copy()))
    assert isinstance(segmented, np.ndarray)
    assert segmented.shape == IMAGE_SIZE

    result_mask = segmented > 0
    assert np.array_equal(result_mask, expected_mask)

    _maybe_save_debug_collage(
        "segment_crypts_runs_at_all",
        red=RED_CHANNEL,
        blue=BLUE_CHANNEL,
        expected=expected_mask,
        result=result_mask,
    )
