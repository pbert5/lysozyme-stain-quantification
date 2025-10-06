import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scientific_image_finder import find_subject_image_sets


def _make_image(path: Path, value: int | np.ndarray, *, mtime: float | None = None) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, np.ndarray):
        data = value.astype(np.uint16, copy=False)
    else:
        data = np.full((4, 4), value, dtype=np.uint16)
    tifffile.imwrite(path, data)
    if mtime is not None:
        os.utime(path, times=(mtime, mtime))
    return data


def test_find_subject_image_sets_basic(tmp_path: Path) -> None:
    red1 = _make_image(tmp_path / "mouse1_RFP.tif", value=1)
    blue1 = _make_image(tmp_path / "mouse1_DAPI.tif", value=10)
    red2 = _make_image(tmp_path / "mouse2_RFP.tif", value=2)
    blue2 = _make_image(tmp_path / "mouse2_DAPI.tif", value=20)

    subjects, images_by_source, source_names = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects == ["mouse1", "mouse2"]
    assert source_names == ["red", "blue"]
    assert np.array_equal(images_by_source[0][0], red1)
    assert np.array_equal(images_by_source[0][1], red2)
    assert np.array_equal(images_by_source[1][0], blue1)
    assert np.array_equal(images_by_source[1][1], blue2)


def test_find_subject_image_sets_with_subdirs(tmp_path: Path) -> None:
    day1_red = _make_image(tmp_path / "day1" / "subject_RFP.tif", value=3)
    day1_blue = _make_image(tmp_path / "day1" / "subject_DAPI.tif", value=30)
    day2_red = _make_image(tmp_path / "day2" / "subject_RFP.tif", value=4)
    day2_blue = _make_image(tmp_path / "day2" / "subject_DAPI.tif", value=40)

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects == ["subject [day1]", "subject [day2]"]
    assert np.array_equal(images_by_source[0][0], day1_red)
    assert np.array_equal(images_by_source[0][1], day2_red)
    assert np.array_equal(images_by_source[1][0], day1_blue)
    assert np.array_equal(images_by_source[1][1], day2_blue)


def test_find_subject_image_sets_uses_timestamp_for_duplicates(tmp_path: Path) -> None:
    first_red = _make_image(tmp_path / "subject01_RFP.tif", value=5, mtime=0)
    first_blue = _make_image(tmp_path / "subject01_DAPI.tif", value=50, mtime=0)
    second_red = _make_image(tmp_path / "subject01_RFP.2.tif", value=6, mtime=1)
    second_blue = _make_image(tmp_path / "subject01_DAPI.2.tif", value=60, mtime=1)

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects[0] == "subject01"
    assert subjects[1].startswith("subject01 [1970-01-01 00:00:01]")
    assert subjects[1] != subjects[0]
    assert np.array_equal(images_by_source[0][0], first_red)
    assert np.array_equal(images_by_source[1][0], first_blue)
    assert np.array_equal(images_by_source[0][1], second_red)
    assert np.array_equal(images_by_source[1][1], second_blue)


def test_drop_zero_channels_and_convert_to_grayscale(tmp_path: Path) -> None:
    red_data = np.zeros((4, 4, 3), dtype=np.uint16)
    red_data[..., 1] = 7
    blue_data = np.zeros((4, 4, 2), dtype=np.uint16)
    blue_data[..., 0] = 3

    red = _make_image(tmp_path / "sample_RFP.tif", value=red_data)
    blue = _make_image(tmp_path / "sample_DAPI.tif", value=blue_data)

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects == ["sample"]
    assert images_by_source[0][0].shape == red.shape[:2]
    assert np.array_equal(images_by_source[0][0], np.full((4, 4), 7, dtype=np.uint16))
    assert images_by_source[1][0].shape == blue.shape[:2]
    assert np.array_equal(images_by_source[1][0], np.full((4, 4), 3, dtype=np.uint16))


def test_channel_selector(tmp_path: Path) -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint16)
    rgb[..., 0] = 1
    rgb[..., 1] = 2
    rgb[..., 2] = 3

    _make_image(tmp_path / "sample_RFP.tif", value=rgb)
    _make_image(tmp_path / "sample_DAPI.tif", value=rgb)

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP", "g"), ("blue", "DAPI", "b")],
    )

    assert subjects == ["sample"]
    assert np.array_equal(images_by_source[0][0], np.full((4, 4), 2, dtype=np.uint16))
    assert np.array_equal(images_by_source[1][0], np.full((4, 4), 3, dtype=np.uint16))


def test_inconsistent_shapes_raise(tmp_path: Path) -> None:
    _make_image(tmp_path / "subj1_RFP.tif", value=1)
    _make_image(tmp_path / "subj1_DAPI.tif", value=10)
    _make_image(tmp_path / "subj2_RFP.tif", value=np.full((5, 5), 2, dtype=np.uint16))
    _make_image(tmp_path / "subj2_DAPI.tif", value=20)

    with pytest.raises(ValueError, match="Inconsistent image shape"):
        find_subject_image_sets(
            tmp_path,
            [("red", "RFP"), ("blue", "DAPI")],
        )


def test_squeeze_simple_nesting(tmp_path: Path) -> None:
    nested = np.full((1, 4, 4), 5, dtype=np.uint16)
    _make_image(tmp_path / "nested_RFP.tif", value=nested)
    _make_image(tmp_path / "nested_DAPI.tif", value=2)

    _make_image(tmp_path / "nested_RFP.2.tif", value=5)
    _make_image(tmp_path / "nested_DAPI.2.tif", value=2)

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert len(subjects) == 2
    red_shapes = {img.shape for img in images_by_source[0]}
    blue_shapes = {img.shape for img in images_by_source[1]}
    assert red_shapes == {(4, 4)}
    assert blue_shapes == {(4, 4)}
