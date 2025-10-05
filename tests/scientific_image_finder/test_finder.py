import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scientific_image_finder import find_subject_image_sets


def _make_image(path: Path, *, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    if mtime is not None:
        os.utime(path, times=(mtime, mtime))


def test_find_subject_image_sets_basic(tmp_path: Path) -> None:
    _make_image(tmp_path / "mouse1_RFP.tif")
    _make_image(tmp_path / "mouse1_DAPI.tif")
    _make_image(tmp_path / "mouse2_RFP.tif")
    _make_image(tmp_path / "mouse2_DAPI.tif")

    subjects, images_by_source, source_names = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects == ["mouse1", "mouse2"]
    assert source_names == ["red", "blue"]
    assert [path.name for path in images_by_source[0]] == ["mouse1_RFP.tif", "mouse2_RFP.tif"]
    assert [path.name for path in images_by_source[1]] == ["mouse1_DAPI.tif", "mouse2_DAPI.tif"]


def test_find_subject_image_sets_with_subdirs(tmp_path: Path) -> None:
    _make_image(tmp_path / "day1" / "subject_RFP.tif")
    _make_image(tmp_path / "day1" / "subject_DAPI.tif")
    _make_image(tmp_path / "day2" / "subject_RFP.tif")
    _make_image(tmp_path / "day2" / "subject_DAPI.tif")

    subjects, images_by_source, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects == ["subject [day1]", "subject [day2]"]
    assert [path.parent.name for path in images_by_source[0]] == ["day1", "day2"]
    assert [path.parent.name for path in images_by_source[1]] == ["day1", "day2"]


def test_find_subject_image_sets_uses_timestamp_for_duplicates(tmp_path: Path) -> None:
    _make_image(tmp_path / "subject01_RFP.tif", mtime=0)
    _make_image(tmp_path / "subject01_DAPI.tif", mtime=0)
    _make_image(tmp_path / "subject01_RFP.2.tif", mtime=1)
    _make_image(tmp_path / "subject01_DAPI.2.tif", mtime=1)

    subjects, _, _ = find_subject_image_sets(
        tmp_path,
        [("red", "RFP"), ("blue", "DAPI")],
    )

    assert subjects[0] == "subject01"
    assert subjects[1].startswith("subject01 [1970-01-01 00:00:01]")
    assert subjects[1] != subjects[0]
