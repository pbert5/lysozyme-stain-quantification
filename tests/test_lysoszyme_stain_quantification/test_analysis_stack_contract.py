from __future__ import annotations

import numpy as np
import pytest

from src.lysozyme_stain_quantification.normalize_rfp import compute_normalized_rfp
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from src.lysozyme_stain_quantification.utils.subject_scale_lookup import subject_scale_from_name
from tests.test_lysoszyme_stain_quantification.tools.demo_crypt_image_generator import (
    generate_test_crypts,
)


def test_segment_crypts_contract_compliance() -> None:
    red, blue, _ = generate_test_crypts(
        num_crypts=3,
        image_size=(128, 128),
        blob_size_px=24,
        intensities={"crypt_blue_sub": 0.2},
    )
    microns_per_px = np.array(0.5, dtype=float)

    result = segment_crypts(
        channels=[red, blue, microns_per_px],
        masks=[],
        blob_size_um=12.0,
        max_regions=6,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == red.shape
    assert np.issubdtype(result.dtype, np.integer)


def test_subject_scale_from_name_contract_compliance() -> None:
    value = subject_scale_from_name(
        channels=[np.array("Mouse_40x")],
        masks=[],
        keys=["20x", "40x"],
        values=[0.9, 0.45],
        default=0.5,
    )

    assert isinstance(value, float)
    assert value == pytest.approx(0.45)


def test_compute_normalized_rfp_contract_compliance() -> None:
    red = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 6, 6, 6, 6, 1],
            [1, 6, 8, 8, 6, 1],
            [1, 6, 8, 8, 6, 1],
            [1, 6, 6, 6, 6, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    blue = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    crypts = np.zeros_like(red, dtype=int)
    crypts[2:4, 2:4] = 1

    result = compute_normalized_rfp(
        channels=[red, blue, crypts],
        masks=[],
        extra_config="ignored",
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == red.shape
    assert np.issubdtype(result.dtype, np.floating)
