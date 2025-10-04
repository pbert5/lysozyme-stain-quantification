from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas", reason="scoring_selector depends on pandas")

from src.lysozyme_stain_quantification.crypts.scoring_selector import scoring_selector


WEIGHTS_ALL_ONES = {
    "circularity": 1.0,
    "area": 1.0,
    "line_fit": 1.0,
    "red_intensity": 1.0,
}

IMAGE_SHAPE = (120, 240)


def _even_centers(count: int) -> list[tuple[int, int]]:
    height, width = IMAGE_SHAPE
    if count == 1:
        return [(height // 2, width // 2)]
    x_positions = np.linspace(30, width - 30, count)
    y_coord = height // 2
    return [(y_coord, int(round(x))) for x in x_positions]


def _draw_disc(dest: np.ndarray, center: tuple[int, int], radius: int, label: int) -> np.ndarray:
    cy, cx = center
    yy, xx = np.ogrid[:IMAGE_SHAPE[0], :IMAGE_SHAPE[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    dest[mask] = label
    return mask


def _draw_rectangle(dest: np.ndarray, center: tuple[int, int], *, width: int, height: int, label: int) -> np.ndarray:
    cy, cx = center
    half_h = height // 2
    half_w = width // 2
    y0 = max(cy - half_h, 0)
    y1 = min(cy + half_h + (height % 2), IMAGE_SHAPE[0])
    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w + (width % 2), IMAGE_SHAPE[1])
    mask = np.zeros_like(dest, dtype=bool)
    mask[y0:y1, x0:x1] = True
    dest[mask] = label
    return mask


def _build_standard_scene(
    count: int,
    *,
    base_radius: int = 12,
    radius_overrides: dict[int, int] | None = None,
    rectangles: dict[int, tuple[int, int]] | None = None,
    offsets: dict[int, int] | None = None,
    intensities: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    radius_overrides = radius_overrides or {}
    rectangles = rectangles or {}
    offsets = offsets or {}
    intensities = intensities or {}

    label_img = np.zeros(IMAGE_SHAPE, dtype=np.int32)
    raw_img = np.ones(IMAGE_SHAPE, dtype=np.float32)

    for idx, (base_y, base_x) in enumerate(_even_centers(count), start=1):
        cy = base_y + offsets.get(idx, 0)
        cx = base_x

        if idx in rectangles:
            height, width = rectangles[idx]
            mask = _draw_rectangle(label_img, (cy, cx), width=width, height=height, label=idx)
        else:
            radius = radius_overrides.get(idx, base_radius)
            mask = _draw_disc(label_img, (cy, cx), radius, idx)

        raw_img[mask] = intensities.get(idx, 1.0)

    return label_img, raw_img


def _mask_without_labels(label_img: np.ndarray, labels: int | tuple[int, ...] | list[int]) -> np.ndarray:
    if isinstance(labels, int):
        labels = (labels,)
    filtered = label_img.copy()
    for label in labels:
        filtered[filtered == label] = 0
    return filtered > 0


def _select_binary(label_img: np.ndarray, raw_img: np.ndarray | None, *, max_regions: int) -> np.ndarray:
    selector = scoring_selector(
        label_img,
        raw_img=raw_img,
        max_regions=max_regions,
        weights=WEIGHTS_ALL_ONES,
    )
    result = selector.select()
    return result > 0


BASIC_LABELS, BASIC_RAW = _build_standard_scene(3)
LIMIT_LABELS, LIMIT_RAW = _build_standard_scene(6)
CIRC_LABELS, CIRC_RAW = _build_standard_scene(5, rectangles={5: (8, 28)})
AREA_SMALL_LABELS, AREA_SMALL_RAW = _build_standard_scene(5, radius_overrides={5: 6})
AREA_LARGE_LABELS, AREA_LARGE_RAW = _build_standard_scene(5, radius_overrides={5: 26})
LINE_FIT_LABELS, LINE_FIT_RAW = _build_standard_scene(5, offsets={5: 80})
RED_INT_LABELS, RED_INT_RAW = _build_standard_scene(5, intensities={5: 0.5})

EXPECTED_BASIC = BASIC_LABELS > 0
EXPECTED_LIMIT = _mask_without_labels(LIMIT_LABELS, (5, 6))
EXPECTED_CIRCULARITY = _mask_without_labels(CIRC_LABELS, 5)
EXPECTED_AREA_SMALL = _mask_without_labels(AREA_SMALL_LABELS, 5)
EXPECTED_AREA_LARGE = _mask_without_labels(AREA_LARGE_LABELS, 4)
EXPECTED_LINE_FIT = _mask_without_labels(LINE_FIT_LABELS, 5)
EXPECTED_RED_INT = _mask_without_labels(RED_INT_LABELS, 5)


def test_scoring_selector_basic_passthrough() -> None:
    mask = _select_binary(BASIC_LABELS.copy(), BASIC_RAW.copy(), max_regions=5)
    assert np.array_equal(mask, EXPECTED_BASIC)


def test_scoring_selector_limits_to_max_regions() -> None:
    mask = _select_binary(LIMIT_LABELS.copy(), LIMIT_RAW.copy(), max_regions=4)
    assert np.array_equal(mask, EXPECTED_LIMIT)


def test_scoring_selector_penalizes_circularity() -> None:
    mask = _select_binary(CIRC_LABELS.copy(), CIRC_RAW.copy(), max_regions=4)
    assert np.array_equal(mask, EXPECTED_CIRCULARITY)


def test_scoring_selector_penalizes_small_area() -> None:
    mask = _select_binary(AREA_SMALL_LABELS.copy(), AREA_SMALL_RAW.copy(), max_regions=4)
    assert np.array_equal(mask, EXPECTED_AREA_SMALL)


def test_scoring_selector_penalizes_large_area() -> None:
    mask = _select_binary(AREA_LARGE_LABELS.copy(), AREA_LARGE_RAW.copy(), max_regions=4)
    assert np.array_equal(mask, EXPECTED_AREA_LARGE)


def test_scoring_selector_penalizes_line_fit_offset() -> None:
    mask = _select_binary(LINE_FIT_LABELS.copy(), None, max_regions=4)
    assert np.array_equal(mask, EXPECTED_LINE_FIT)


def test_scoring_selector_penalizes_low_red_intensity() -> None:
    mask = _select_binary(RED_INT_LABELS.copy(), RED_INT_RAW.copy(), max_regions=4)
    assert np.array_equal(mask, EXPECTED_RED_INT)
