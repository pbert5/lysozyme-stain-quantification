"""Utilities for locating multi-source scientific image sets."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple


_ALLOWED_SUFFIXES = (".tif", ".tiff")
_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SourceSpec:
    """Specification for an image source to locate."""

    index: int
    name: str
    search_key: str


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
    sources: Sequence[Tuple[str, str]],
    *,
    allowed_suffixes: Sequence[str] | None = None,
) -> Tuple[List[str], List[List[Path]], List[str]]:
    """
    Discover aligned image sets for each subject across multiple sources.

    Args:
        img_dir: Root directory to search.
        sources: Sequence of (source_name, search_key) pairs.
        allowed_suffixes: Optional tuple of file suffixes to accept; defaults to TIFF variants.

    Returns:
        Tuple of (subject_names, images_by_source, source_names) where images_by_source[i]
        corresponds to the list of paths for sources[i].
    """
    if not sources:
        raise ValueError("sources must not be empty")

    suffixes = tuple(sfx.lower() for sfx in (allowed_suffixes or _ALLOWED_SUFFIXES))
    specs = [SourceSpec(idx, name, key) for idx, (name, key) in enumerate(sources)]
    grouped = _group_images_by_base(img_dir, specs, suffixes)

    subject_names: List[str] = []
    images_by_source: List[List[Path]] = [[] for _ in specs]
    source_names = [spec.name for spec in specs]
    existing_labels: set[str] = set()

    for base_name in sorted(grouped):
        base_matches = _build_matches_for_base(base_name, grouped[base_name], specs, existing_labels)
        for label, records in base_matches:
            subject_names.append(label)
            existing_labels.add(label)
            for record in records:
                images_by_source[record.spec.index].append(record.path)

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
    token = f"_{search_key.lower()}."
    lower_name = file_name.lower()
    idx = lower_name.find(token)
    if idx == -1:
        return None
    return file_name[:idx]


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


def _pick_best_subdir_label(records: Sequence[SourceImage]) -> str:
    labels = [record.subdir_label for record in records if record.subdir_label]
    if not labels:
        return ""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
