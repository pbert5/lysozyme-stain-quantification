from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ChannelMatchRule:
    column: str
    tokens: Tuple[str, ...]


def _iter_image_files(root: Path, include_extensions: Sequence[str], recursive: bool) -> Iterable[Path]:
    allowed = {ext.lower() for ext in include_extensions if ext}
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at the top level")
    return data


def _normalize_subject_token(text: str) -> str:
    text = text.strip().upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9_-]", "", text)
    return text


def _subject_from_path(path: Path, root: Path, strategy: str) -> str:
    rel_parent = path.parent.relative_to(root)
    if strategy == "parent_dir_name":
        return rel_parent.name
    if strategy == "two_level_dir":
        parts = rel_parent.parts
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        if len(parts) == 1:
            return parts[0]
        return path.stem
    if strategy == "relative_dir":
        return str(rel_parent) if str(rel_parent) != "." else path.stem
    return path.stem


def _discover_structured_rows(
    dataset_name: str,
    root: Path,
    subject_strategy: str,
    channel_file_names: Dict[str, str],
    microns_per_pixel: Optional[float],
    recursive: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lyso_name = channel_file_names.get("lysozyme")
    tissue_name = channel_file_names.get("tissue")
    if not lyso_name or not tissue_name:
        return rows

    folders = root.rglob("*") if recursive else root.glob("*")
    for folder in sorted(folders):
        if not folder.is_dir():
            continue
        lyso_path = folder / lyso_name
        tissue_path = folder / tissue_name
        if not lyso_path.exists() or not tissue_path.exists():
            continue
        subject_id = _subject_from_path(lyso_path, root=root, strategy=subject_strategy)
        rows.append(
            {
                "subject_id": subject_id,
                "lysozyme_path": str(lyso_path.resolve()),
                "tissue_path": str(tissue_path.resolve()),
                "microns_per_pixel": "" if microns_per_pixel is None else f"{float(microns_per_pixel):.6f}",
                "source_dataset": dataset_name,
                "source_label": "structured_dir",
                "notes": "",
            }
        )
    return rows


def _extract_base_name(file_name: str, search_token: str) -> Optional[str]:
    token = (search_token or "").lower().strip()
    if not token:
        return None

    name_lower = file_name.lower()
    separators = ("_", "-", " ", "")
    for sep in separators:
        needle = f"{sep}{token}."
        idx = name_lower.find(needle)
        if idx != -1:
            return file_name[:idx].rstrip(" _-")

    needle = f"{token}."
    idx = name_lower.find(needle)
    if idx != -1:
        return file_name[:idx].rstrip(" _-")

    return None


def _subject_from_file(path: Path, root: Path) -> str:
    # Prefer explicit subject prefix like G3ER, G2FB_01, etc., then append parent folder for uniqueness.
    stem = _normalize_subject_token(path.stem)
    match = re.search(r"(G\d+[A-Z]{1,2}(?:_\d+)?)", stem)
    base = match.group(1) if match else path.stem
    rel_parent = path.parent.relative_to(root)
    parent_key = str(rel_parent) if str(rel_parent) != "." else "root"
    return f"{base} [{parent_key}]"


def _discover_token_rows(
    dataset_name: str,
    root: Path,
    rules: Sequence[ChannelMatchRule],
    include_extensions: Sequence[str],
    exclude_name_tokens: Sequence[str],
    allow_combined: bool,
    microns_per_pixel: Optional[float],
    recursive: bool,
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, Path]] = {}
    excluded = tuple(token.lower() for token in exclude_name_tokens if token)

    for path in _iter_image_files(root, include_extensions=include_extensions, recursive=recursive):
        lower_name = path.name.lower()
        if excluded and any(token in lower_name for token in excluded):
            continue

        matched = False
        for rule in rules:
            for token in rule.tokens:
                base = _extract_base_name(path.name, token)
                if base is None:
                    continue
                parent_rel = str(path.parent.relative_to(root))
                key = (parent_rel, base)
                grouped.setdefault(key, {})[rule.column] = path
                matched = True
                break
            if matched:
                break

        if not matched and allow_combined:
            # Combined-channel fallback where one file is used for both channels.
            key = (str(path.parent.relative_to(root)), path.stem)
            grouped.setdefault(key, {})["lysozyme_path"] = path
            grouped.setdefault(key, {})["tissue_path"] = path

    rows: List[Dict[str, str]] = []
    for _, channels in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        lyso = channels.get("lysozyme_path")
        tissue = channels.get("tissue_path")
        if lyso is None or tissue is None:
            continue
        subject_id = _subject_from_file(lyso, root=root)
        rows.append(
            {
                "subject_id": subject_id,
                "lysozyme_path": str(lyso.resolve()),
                "tissue_path": str(tissue.resolve()),
                "microns_per_pixel": "" if microns_per_pixel is None else f"{float(microns_per_pixel):.6f}",
                "source_dataset": dataset_name,
                "source_label": "token_match",
                "notes": "",
            }
        )
    return rows


def _selected_datasets(config: Dict) -> Optional[Set[str]]:
    selected = config.get("pipeline_config", {}).get("discovery_datasets_to_use", [])
    if selected is None:
        return None
    if not isinstance(selected, list):
        raise ValueError("pipeline_config.discovery_datasets_to_use must be a list")
    cleaned = {str(name).strip() for name in selected if str(name).strip()}
    return cleaned if cleaned else None


def discover_rows(config: Dict) -> List[Dict[str, str]]:
    dataset_cfg = config.get("dataset_config", {})
    discovery_cfg = dataset_cfg.get("discovery", {})

    rows: List[Dict[str, str]] = []
    datasets = discovery_cfg.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("dataset_config.discovery.datasets must be a list")
    selected = _selected_datasets(config)

    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        dataset_name = str(dataset.get("name", "")).strip()
        if selected is not None and dataset_name not in selected:
            continue
        root = Path(str(dataset.get("root_dir", ""))).expanduser()
        if not root.exists():
            print(f"[skip] missing root_dir: {root}")
            continue

        mode = str(dataset.get("mode", "token_match")).strip().lower()
        subject_strategy = str(dataset.get("subject_from", "parent_dir_name")).strip()
        mpp = dataset.get("microns_per_pixel")
        include_extensions = dataset.get("include_extensions", list(ALLOWED_EXTENSIONS))
        recursive = bool(dataset.get("recursive", True))

        if mode == "structured_dir":
            channel_file_names = dataset.get("channel_file_names", {})
            rows.extend(
                _discover_structured_rows(
                    dataset_name=dataset_name or "unnamed_dataset",
                    root=root,
                    subject_strategy=subject_strategy,
                    channel_file_names=channel_file_names,
                    microns_per_pixel=mpp,
                    recursive=recursive,
                )
            )
            continue

        channel_tokens_cfg = dataset.get("channel_tokens", {})
        lyso_tokens = tuple(channel_tokens_cfg.get("lysozyme", []))
        tissue_tokens = tuple(channel_tokens_cfg.get("tissue", []))
        rules = (
            ChannelMatchRule(column="lysozyme_path", tokens=lyso_tokens),
            ChannelMatchRule(column="tissue_path", tokens=tissue_tokens),
        )
        rows.extend(
            _discover_token_rows(
                dataset_name=dataset_name or "unnamed_dataset",
                root=root,
                rules=rules,
                include_extensions=include_extensions,
                exclude_name_tokens=dataset.get("exclude_name_tokens", []),
                allow_combined=bool(dataset.get("allow_combined_single_file", True)),
                microns_per_pixel=mpp,
                recursive=recursive,
            )
        )

    # Deduplicate by exact pair path to avoid duplicate rows across datasets.
    dedup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["lysozyme_path"], row["tissue_path"])
        dedup[key] = row
    return sorted(dedup.values(), key=lambda r: r["subject_id"].lower())


def write_csv(csv_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "lysozyme_path",
        "tissue_path",
        "microns_per_pixel",
        "source_dataset",
        "source_label",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover lysozyme/tissue image pairs and rebuild input CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("lysozyme_pipeline_config.yaml"),
        help="Path to the YAML config created by init script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)

    dataset_cfg = config.get("dataset_config", {})
    csv_path = Path(str(dataset_cfg.get("input_csv", "lysozyme_input_data.csv")))
    if not csv_path.is_absolute():
        csv_path = (config_path.parent / csv_path).resolve()

    rows = discover_rows(config)
    write_csv(csv_path, rows)

    print(f"Discovered {len(rows)} image rows")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
