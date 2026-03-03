from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

LOGGER = logging.getLogger(__name__)

# Default set of stages we bother persisting when no custom whitelist is provided.
DEFAULT_DEBUG_STAGE_WHITELIST: tuple[str, ...] = (
    # Core channel inputs
    "rfp_input",
    "dapi_input",
    # Preprocessed caps components (support wildcard for tissue/crypt variants)
    "crypt_preprocessed",
    "tissue_preprocessed",
    "crypt_caps*",
    "tissue_caps*",
    # Crypt separation intermediates
    "crypt_clean",
    "crypt_troughs",
    "tissue_clean",
    "tissue_troughs",
    "thinned_crypts",
    "split_crypts",
    "split_times_thinned",
    "opened_split_times_thinned",
    "good_crypts",
    "good_crypts_smoothed",
    "distance_image",
    "local_maxima_mask",
    "watershed_mask",
    "watershed_labels",
    "seed_labels",
    "legacy_seed_labels",
    "new_seed_labels",
    "selected_label_set",
    "old_like_combined_labels",
    "old_like_expanded_labels",
    "old_like_watershed",
    # Segmenter book-keeping
    "potential_crypts",
    "cleaned_crypts",
    "best_crypts_candidate",
    "base_labels",
    "final_crypt_labels",
    # Effective count pipeline
    "best_crypts_labels",
    "medium_crypts_labels",
    "match_stack_*",
    "match_stack_max_projection",
    "match_stack_mean_projection",
    "match_stack_weights",
    "collapsed_results",
    "collapsed_caps_clean",
    "otsu_binary_mask",
    "labeled_binary_clean",
    "selected_labels_mask",
    "selected_labels",
    "selected_labels_intersection",
    "best_crypt_bounds",
    "medium_crypt_bounds",
)


def _safe_slug(value: Any, fallback: str = "value") -> str:
    """Return a filesystem-friendly token."""
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        elif ch in (" ", "/"):
            safe_chars.append("_")
        else:
            safe_chars.append("-")
    slug = "".join(safe_chars).strip("_-.")
    return slug or fallback


def _normalize_color(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False))
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _normalize_dynamic_range(arr: np.ndarray, *, use_log: bool = False) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False))
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32)
    data = arr[finite_mask]
    lo, hi = np.percentile(data, [1.0, 99.0])
    if hi <= lo:
        lo, hi = data.min(), data.max()
    arr = np.clip(arr, lo, hi)
    if use_log:
        arr = np.log1p(arr - lo)
        hi = np.log1p(hi - lo)
        lo = 0.0
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def _prepare_image_for_save(image: Any, stage: Optional[str] = None) -> Optional[np.ndarray]:
    """Return a float array in [0, 1] suitable for saving as PNG."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)

    use_log = False
    if stage:
        sl = stage.lower()
        if "match_stack" in sl or "collapsed" in sl:
            use_log = True

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return _normalize_color(arr)

    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        max_val = int(arr.max(initial=0))
        unique = np.unique(arr)
        if max_val > 1 and unique.size > 2:
            rgb = label2rgb(arr, bg_label=0)
            return np.clip(rgb.astype(np.float32), 0.0, 1.0)

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1])

    return _normalize_dynamic_range(arr, use_log=use_log)


@dataclass
class DebugImageSession:
    """Per-subject helper that persists intermediate images and records metadata."""

    root_dir: Path
    subject_name: str
    whitelist: Optional[set[str]] = None
    source_paths: Optional[Sequence[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.subject_slug = _safe_slug(self.subject_name or "subject", fallback="subject")
        self.session_id = uuid.uuid4().hex[:6]
        self.subject_dir = self.root_dir / f"{self.subject_slug}_{self.session_id}"
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._counter = 0
        if self.whitelist:
            cleaned_tokens = [token.strip().lower() for token in self.whitelist if token and token.strip()]
            self._whitelist_tokens = tuple(sorted(dict.fromkeys(cleaned_tokens)))
            self._whitelist_exact = {token for token in self._whitelist_tokens if not token.endswith("*")}
            self._whitelist_prefixes = [token[:-1] for token in self._whitelist_tokens if token.endswith("*")]
        else:
            self._whitelist_tokens = None
            self._whitelist_exact = None
            self._whitelist_prefixes = None
        self._stage_counts: Dict[Tuple[str, str], int] = {}

    def save_image(
        self,
        image: Any,
        stage: str,
        *,
        source: Optional[str] = None,
        description: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """Persist `image` to disk when stage passes the whitelist."""
        if not self.enabled:
            return None
        stage_norm = (stage or "").strip()
        if not stage_norm:
            raise ValueError("stage must be a non-empty string")

        if not force and self._whitelist_tokens is not None:
            stage_lower = stage_norm.lower()
            allowed = stage_lower in (self._whitelist_exact or set())
            if not allowed and self._whitelist_prefixes:
                allowed = any(stage_lower.startswith(prefix) for prefix in self._whitelist_prefixes)
            if not allowed:
                return None

        prepared = _prepare_image_for_save(image, stage_norm)
        if prepared is None:
            return None

        source_slug = _safe_slug(source or "root", fallback="root")
        stage_slug = _safe_slug(stage_norm, fallback="stage")
        out_dir = self.subject_dir / source_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        key = (source_slug, stage_slug)
        idx = self._stage_counts.get(key, 0)
        self._stage_counts[key] = idx + 1
        suffix = "" if idx == 0 else f"_{idx:02d}"

        filename = f"{self._counter:03d}_{stage_slug}{suffix}.png"
        out_path = out_dir / filename

        try:
            plt.imsave(out_path, prepared, cmap="gray" if prepared.ndim == 2 else None)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to save debug image %s: %s", out_path, exc)
            return None

        record: Dict[str, Any] = {
            "stage": stage_norm,
            "source": source or "root",
            "path": str(out_path.relative_to(self.root_dir)),
            "shape": list(np.asarray(image).shape),
            "dtype": str(np.asarray(image).dtype),
        }
        if description:
            record["description"] = description

        self._entries.append(record)
        self._counter += 1
        return out_path

    def to_summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of saved intermediates."""
        return {
            "subject": self.subject_name,
            "subject_slug": self.subject_slug,
            "session_id": self.session_id,
            "root": str(self.subject_dir.relative_to(self.root_dir)),
            "source_paths": list(self.source_paths or []),
            "metadata": dict(self.metadata),
            "entries": list(self._entries),
            "whitelist": list(self._whitelist_tokens) if self._whitelist_tokens else None,
        }


@dataclass
class DebugImageManager:
    """Factory that hands out DebugImageSession objects."""

    root_dir: Path
    whitelist: Optional[Sequence[str]] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.whitelist:
            cleaned = {token.strip().lower() for token in self.whitelist if token}
        else:
            cleaned = None
        self._whitelist = cleaned

    def create_session(
        self,
        subject_name: str,
        *,
        source_paths: Optional[Sequence[Path | str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DebugImageSession:
        paths: Optional[List[str]] = None
        if source_paths is not None:
            paths = [str(Path(p)) for p in source_paths]
        return DebugImageSession(
            root_dir=self.root_dir,
            subject_name=subject_name,
            whitelist=self._whitelist,
            source_paths=paths,
            metadata=metadata or {},
            enabled=self.enabled,
        )
