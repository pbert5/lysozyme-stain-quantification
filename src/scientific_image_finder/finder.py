"""Utilities for locating multi-source scientific image sets."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


_ALLOWED_SUFFIXES = (".tif", ".tiff")
_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SourceSpec:
    """Specification for an image source to locate."""

    index: int
    name: str
    search_key: str
    channel_selector: Optional[str] = None


@dataclass
class SourceImage:
    """Metadata for a discovered image file."""

    spec: SourceSpec
    path: Path
    base_name: str
    subdir_key: str
    subdir_label: str
    timestamp: float


def validate_directories(img_dir: Path, results_dir: Path | None = None) -> bool:
    """Validate that the image directory exists and optionally prepare results directory."""
    if not img_dir.exists():
        print(f"Error: Image directory does not exist: {img_dir}")
        return False

    if not img_dir.is_dir():
        print(f"Error: Image path is not a directory: {img_dir}")
        return False

    if results_dir is not None:
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - os-specific failure mode
            print(f"Error: Cannot create results directory {results_dir}: {exc}")
            return False

    return True


def find_subject_image_sets(
    img_dir: Path,
    sources: Sequence[Tuple[str, str] | Tuple[str, str, str]],
    *,
    allowed_suffixes: Sequence[str] | None = None,
    max_subjects: Optional[int] = None,
) -> Tuple[List[str], List[List[np.ndarray]], List[str]]: #out: (subject_names, images_by_source, source_names)
    """
    Discover aligned image sets for each subject across multiple sources.

    Args:
        img_dir: Root directory to search.
        sources: Sequence of (source_name, search_key) pairs or
            (source_name, search_key, channel_selector) where channel_selector is one of
            "r", "g", "b" indicating which channel to extract.
        allowed_suffixes: Optional tuple of file suffixes to accept; defaults to TIFF variants.
        max_subjects: Optional upper bound on the number of subject image sets to return.

    Returns:
        Tuple of (subject_names, images_by_source, source_names) where images_by_source[i]
        contains a numpy array for each subject with data loaded from the
        corresponding source.
    """
    if not sources:
        raise ValueError("sources must not be empty")

    suffixes = tuple(sfx.lower() for sfx in (allowed_suffixes or _ALLOWED_SUFFIXES))
    specs: List[SourceSpec] = []
    for idx, spec in enumerate(sources):
        if len(spec) == 2:
            name, key = spec
            channel = None
        elif len(spec) == 3:
            name, key, channel = spec
            channel = channel.lower() if channel else None
            if channel not in {"r", "g", "b", None}:
                raise ValueError(f"Unsupported channel selector '{channel}' for source '{name}'")
        else:
            raise ValueError("sources entries must be (name, search_key) or (name, search_key, channel)")
        specs.append(SourceSpec(idx, name, key, channel))
    grouped = _group_images_by_base(img_dir, specs, suffixes)

    if max_subjects is not None:
        if max_subjects < 0:
            raise ValueError("max_subjects must be non-negative")

    subject_names: List[str] = []
    images_by_source: List[List[np.ndarray]] = [[] for _ in specs]
    source_names = [spec.name for spec in specs]
    existing_labels: set[str] = set()
    image_cache: Dict[Path, np.ndarray] = {}
    subjects_found = 0

    if max_subjects == 0:
        return subject_names, images_by_source, source_names

    for base_name in sorted(grouped):
        if max_subjects is not None and subjects_found >= max_subjects:
            break
        base_matches = _build_matches_for_base(base_name, grouped[base_name], specs, existing_labels)
        for label, records in base_matches:
            subject_names.append(label)
            existing_labels.add(label)
            for record in records:
                img = _load_image(record.path, image_cache)
                prepared = _prepare_image(img, record.spec.channel_selector)
                images_by_source[record.spec.index].append(prepared)
            subjects_found += 1
            if max_subjects is not None and subjects_found >= max_subjects:
                break
        if max_subjects is not None and subjects_found >= max_subjects:
            break

    # Filter out subjects with inconsistent shapes instead of failing
    subject_names, images_by_source = _filter_consistent_shapes(
        images_by_source, source_names, subject_names
    )
    return subject_names, images_by_source, source_names


def _group_images_by_base(
    img_dir: Path,
    specs: Sequence[SourceSpec],
    suffixes: Sequence[str],
) -> Dict[str, Dict[int, List[SourceImage]]]:
    grouped: Dict[str, Dict[int, List[SourceImage]]] = defaultdict(lambda: defaultdict(list))
    seen: set[Tuple[int, Path]] = set()

    for path in _iter_candidate_files(img_dir, suffixes):
        for spec in specs:
            base_name = _extract_base_name(path.name, spec.search_key)
            if base_name is None:
                continue
            key = (spec.index, path)
            if key in seen:
                continue
            seen.add(key)
            rel_parts = path.relative_to(img_dir).parts[:-1]
            subdir_key = "/".join(rel_parts)
            subdir_label = "/".join(rel_parts) if rel_parts else ""
            timestamp = path.stat().st_mtime
            record = SourceImage(spec, path, base_name, subdir_key, subdir_label, timestamp)
            grouped[base_name][spec.index].append(record)

    return grouped


def _iter_candidate_files(img_dir: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    for path in img_dir.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if any(name_lower.endswith(sfx) for sfx in suffixes):
            yield path


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


def _build_matches_for_base(
    base_name: str,
    records_by_source: Dict[int, List[SourceImage]],
    specs: Sequence[SourceSpec],
    existing_labels: set[str],
) -> List[Tuple[str, List[SourceImage]]]:
    per_source: Dict[int, Dict[str, Deque[SourceImage]]] = {}
    used_labels = set(existing_labels)
    for spec in specs:
        source_records = records_by_source.get(spec.index, [])
        if not source_records:
            return []
        bucket: Dict[str, Deque[SourceImage]] = defaultdict(deque)
        for record in sorted(
            source_records,
            key=lambda rec: (rec.subdir_key, rec.timestamp, rec.path.name.lower()),
        ):
            bucket[record.subdir_key].append(record)
        per_source[spec.index] = bucket

    matches: List[Tuple[str, List[SourceImage]]] = []

    # First pass: align using shared subdirectories
    common_subdirs = set.intersection(*(set(buckets.keys()) for buckets in per_source.values()))
    for subdir_key in sorted(common_subdirs):
        aligned: List[Deque[SourceImage]] = [per_source[spec.index][subdir_key] for spec in specs]
        count = min(len(queue) for queue in aligned)
        for _ in range(count):
            records = [queue.popleft() for queue in aligned]
            subdir_label = records[0].subdir_label
            label = _make_subject_label(base_name, subdir_label, records, used_labels)
            matches.append((label, records))
            used_labels.add(label)

    # Collect leftovers for timestamp-based alignment
    remaining: List[Deque[SourceImage]] = []
    for spec in specs:
        leftovers: List[SourceImage] = []
        for queue in per_source[spec.index].values():
            while queue:
                leftovers.append(queue.popleft())
        leftovers.sort(key=lambda rec: (rec.timestamp, rec.path.name.lower()))
        remaining.append(deque(leftovers))

    # Second pass: fall back to timestamp ordering
    while remaining and all(remaining[idx] for idx in range(len(remaining))):
        records = [remaining[idx].popleft() for idx in range(len(remaining))]
        subdir_label = _pick_best_subdir_label(records)
        label = _make_subject_label(base_name, subdir_label, records, used_labels)
        matches.append((label, records))
        used_labels.add(label)

    return matches


def _make_subject_label(
    base_name: str,
    subdir_label: str,
    records: Sequence[SourceImage],
    existing_labels: set[str],
) -> str:
    label = base_name.strip()
    if subdir_label:
        label = f"{label} [{subdir_label}]"

    if label in existing_labels:
        timestamp = min(record.timestamp for record in records)
        timestamp_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(_TIMESTAMP_FMT)
        label_with_ts = f"{label} [{timestamp_str}]"
        candidate = label_with_ts
        suffix = 2
        while candidate in existing_labels:
            candidate = f"{label_with_ts} ({suffix})"
            suffix += 1
        label = candidate

    return label


def _load_image(path: Path, cache: Dict[Path, np.ndarray]) -> np.ndarray:
    if path not in cache:
        cache[path] = tifffile.imread(path)
    return cache[path]


def _prepare_image(image: np.ndarray, channel_selector: Optional[str]) -> np.ndarray:
    arr = np.asarray(image)

    channel_axis = _identify_channel_axis(arr)
    if channel_selector:
        if channel_axis is None:
            if arr.ndim == 2:
                pass
            else:
                raise ValueError("Channel selector provided but channel axis could not be identified")
        else:
            arr = np.moveaxis(arr, channel_axis, -1)
            idx_map = {"r": 0, "g": 1, "b": 2}
            idx = idx_map[channel_selector]
            if idx >= arr.shape[-1]:
                raise ValueError(
                    f"Requested channel '{channel_selector}' but image only has {arr.shape[-1]} channels"
                )
            arr = arr[..., idx]
    else:
        if channel_axis is not None:
            arr = np.moveaxis(arr, channel_axis, -1)
            arr = _drop_empty_channels(arr)
    arr = _squeeze_simple(arr)
    return arr


def _identify_channel_axis(arr: np.ndarray) -> int | None:
    if arr.ndim < 3:
        return None
    if arr.shape[-1] <= 4 and (arr.ndim == 3 or arr.shape[-1] != arr.shape[-2]):
        if arr.ndim == 3 and arr.shape[-1] == arr.shape[-2]:
            pass
        else:
            return -1
    if arr.shape[0] <= 4 and arr.shape[0] != arr.shape[1]:
        return 0
    return None


def _drop_empty_channels(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        return arr

    channels = [arr[..., idx] for idx in range(arr.shape[-1])]
    keep = [ch for ch in channels if not np.all(ch == 0)]

    if not keep:
        keep = [channels[0]]

    if len(keep) == 1:
        return keep[0]

    stacked = np.stack(keep, axis=-1)
    if stacked.shape[-1] == 1:
        return stacked[..., 0]
    return stacked


def _squeeze_simple(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 2:
        return arr
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 0:
        return arr
    return squeezed


def _filter_consistent_shapes(
    images_by_source: List[List[np.ndarray]],
    source_names: List[str],
    subject_names: List[str],
) -> Tuple[List[str], List[List[np.ndarray]]]:
    """
    Filter out subjects with inconsistent shapes within each source.
    Returns only subjects that have consistent shapes across all images in their source.
    """
    if not images_by_source or not subject_names:
        return subject_names, images_by_source
    
    # Track which subjects to keep
    subjects_to_keep = set(range(len(subject_names)))
    
    for src_idx, images in enumerate(images_by_source):
        if not images:
            continue
        
        # Find the most common shape for this source
        shape_counts: Dict[tuple, List[int]] = {}
        for subj_idx, img in enumerate(images):
            shape = img.shape
            if shape not in shape_counts:
                shape_counts[shape] = []
            shape_counts[shape].append(subj_idx)
        
        # Use the most common shape as reference
        if not shape_counts:
            continue
        
        most_common_shape = max(shape_counts.keys(), key=lambda s: len(shape_counts[s]))
        reference_indices = set(shape_counts[most_common_shape])
        
        # Remove subjects that don't match the most common shape for this source
        mismatched = set(range(len(images))) - reference_indices
        if mismatched:
            for subj_idx in mismatched:
                subject = subject_names[subj_idx] if subj_idx < len(subject_names) else subj_idx
                print(f"  Note: Excluding '{subject}' - inconsistent shape for source '{source_names[src_idx]}': "
                      f"expected {most_common_shape}, got {images[subj_idx].shape}")
            subjects_to_keep -= mismatched
    
    # Filter all lists to keep only consistent subjects
    if len(subjects_to_keep) < len(subject_names):
        kept_indices = sorted(subjects_to_keep)
        filtered_names = [subject_names[i] for i in kept_indices]
        filtered_images = [[images[i] for i in kept_indices] for images in images_by_source]
        return filtered_names, filtered_images
    
    return subject_names, images_by_source


def _pick_best_subdir_label(records: Sequence[SourceImage]) -> str:
    labels = [record.subdir_label for record in records if record.subdir_label]
    if not labels:
        return ""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]



def find_tif_images_by_keys(
    root_dir: Path,
    keys: List[str],
) -> Dict[str, List[Path]]:
    """
    Recursively search for .tif images and sort by matching keys.
    
    Each image is assigned to the first matching key only (no duplicates).
    
    Parameters
    ----------
    root_dir : Path
        Directory to search recursively
    keys : List[str]
        List of string keys to match (e.g., ["_DAPI", "_RFP", ""])
        Order matters - first match wins
    
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping each key to list of matching image paths
        
    Example
    -------
    >>> sorted_paths = find_tif_images_by_keys(
    ...     Path("./images"),
    ...     keys=["_DAPI", "_RFP", ""]
    ... )
    >>> print(sorted_paths["_DAPI"])  # All DAPI images
    >>> print(sorted_paths["_RFP"])   # All RFP images (excluding any that matched DAPI)
    >>> print(sorted_paths[""])       # All remaining .tif images
    """
    import pandas as pd
    
    # Find all .tif images recursively
    tif_paths = list(root_dir.rglob("*.tif")) + list(root_dir.rglob("*.TIF"))
    
    if not tif_paths:
        return {key: [] for key in keys}
    
    # Create dataframe with paths and names
    df = pd.DataFrame({
        "path": tif_paths,
        "name": [p.name for p in tif_paths],
    })
    
    # Assign each image to first matching key
    df["matched_key"] = None
    
    for key in keys:
        # Find unassigned images that match this key
        mask = (df["matched_key"].isna()) & (df["name"].str.contains(key, case=False, regex=False))
        df.loc[mask, "matched_key"] = key
    
    # Group by matched key and extract paths
    sorted_paths = {key: [] for key in keys}
    
    for key in keys:
        matched = df[df["matched_key"] == key]["path"].tolist()
        sorted_paths[key] = matched
    
    return sorted_paths