"""
Quantify per-crypt fluorescence and geometry for lysozyme staining experiments.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Order of the metrics produced by summarize_crypt_fluorescence.
SUMMARY_FIELD_ORDER: tuple[str, ...] = (
    "crypt_count",
    "crypt_area_um2_sum",
    "crypt_area_um2_mean",
    "crypt_area_um2_std",
    "rfp_sum_total",
    "rfp_sum_mean",
    "rfp_sum_std",
    "rfp_intensity_mean",
    "rfp_intensity_std",
    "rfp_intensity_min",
    "rfp_intensity_max",
    "rfp_max_intensity_mean",
    "rfp_max_intensity_std",
    "effective_full_intensity_um2_sum",
    "effective_full_intensity_um2_mean",
    "effective_full_intensity_um2_std",
)

PER_CRYPT_FIELD_ORDER: tuple[str, ...] = (
    "subject_name",
    "crypt_label",
    "crypt_index",
    "pixel_area",
    "pixel_value_sum",
    "pixel_value_mean",
    "pixel_value_std",
    "pixel_value_min",
    "pixel_value_max",
    "um_area",
    "um_value_sum",
    "um_value_mean",
    "um_value_std",
    "um_value_min",
    "um_value_max",
    "microns_per_px",
)


def _as_scalar_float(value: Any) -> float:
    """Best-effort conversion of an AnalysisStackXR channel element to float."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(()))
    raise ValueError(f"Expected scalar microns-per-pixel value, got shape {array.shape}.")


def _safe_std(values: np.ndarray, *, count: int) -> float:
    """Return the population standard deviation for a 1D array."""
    if count == 0:
        return float("nan")
    if count == 1:
        return 0.0
    return float(np.std(values, ddof=0))


def _as_image(value: Any, *, dtype: type | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim > 2 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)
    return array


def _resolve_primary_inputs(
    normalized_rfp: Any | None,
    crypt_labels: Any | None,
    microns_per_px: Any | None,
    channels: Sequence[Any] | None,
) -> tuple[Any, Any, Any]:
    """Return the three primary channels, preferring explicit kwargs over channels."""
    if channels is not None:
        if len(channels) < 3:
            raise ValueError("channels must provide [normalized_rfp, crypt_labels, microns_per_px].")
        if normalized_rfp is None:
            normalized_rfp = channels[0]
        if crypt_labels is None:
            crypt_labels = channels[1]
        if microns_per_px is None:
            microns_per_px = channels[2]

    if normalized_rfp is None or crypt_labels is None or microns_per_px is None:
        raise ValueError(
            "normalized_rfp, crypt_labels, and microns_per_px must be provided either as "
            "keyword arguments or as the first three entries in channels."
        )
    return normalized_rfp, crypt_labels, microns_per_px


def _coerce_subject_name(subject_name: Any | None) -> str:
    """Return a readable subject name from whatever metadata is supplied."""
    if subject_name is None:
        return "subject"
    if isinstance(subject_name, str):
        return subject_name
    if isinstance(subject_name, bytes):
        return subject_name.decode("utf-8", errors="replace")
    if hasattr(subject_name, "item"):
        try:
            return _coerce_subject_name(subject_name.item())
        except Exception:
            pass
    return str(subject_name)


def _summarize_single_subject(
    normalized_rfp: np.ndarray,
    crypt_labels: np.ndarray,
    microns_per_px: float,
    intensity_upper_bound: float | None,
) -> np.ndarray:
    if normalized_rfp.shape != crypt_labels.shape:
        raise ValueError("normalized_rfp and crypt_labels must share the same shape.")
    if microns_per_px <= 0.0:
        raise ValueError("microns_per_px must be a positive scalar.")

    if not np.any(crypt_labels):
        return np.array(
            [
                0.0,
                0.0,
                float("nan"),
                float("nan"),
                0.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            dtype=np.float64,
        )

    label_image = np.asarray(crypt_labels, dtype=np.int64)
    max_label = int(label_image.max())
    if max_label <= 0:
        raise ValueError("crypt_labels does not contain positive label values.")

    flattened_labels = label_image.ravel()
    flattened_intensity = np.asarray(normalized_rfp, dtype=np.float64).ravel()

    pixel_counts = np.bincount(flattened_labels, minlength=max_label + 1)[1:]
    if not np.any(pixel_counts):
        raise ValueError("No labeled crypt pixels found.")

    valid_mask = pixel_counts > 0
    pixel_counts = pixel_counts[valid_mask]
    label_ids = np.nonzero(valid_mask)[0] + 1

    intensity_sums = np.bincount(
        flattened_labels,
        weights=flattened_intensity,
        minlength=max_label + 1,
    )[1:]
    intensity_sums = intensity_sums[valid_mask]

    intensity_sq_sums = np.bincount(
        flattened_labels,
        weights=flattened_intensity * flattened_intensity,
        minlength=max_label + 1,
    )[1:]
    intensity_sq_sums = intensity_sq_sums[valid_mask]

    pixel_area_um2 = microns_per_px * microns_per_px
    areas_um2 = pixel_counts.astype(np.float64) * pixel_area_um2

    crypt_count = int(pixel_counts.size)
    area_sum = float(np.sum(areas_um2))
    area_mean = float(np.mean(areas_um2)) if crypt_count else float("nan")
    area_std = _safe_std(areas_um2, count=crypt_count)

    mean_intensity = np.divide(intensity_sums, pixel_counts, out=np.zeros_like(intensity_sums), where=pixel_counts > 0)
    variance = np.divide(
        intensity_sq_sums,
        pixel_counts,
        out=np.zeros_like(intensity_sq_sums),
        where=pixel_counts > 0,
    ) - mean_intensity * mean_intensity
    variance = np.clip(variance, a_min=0.0, a_max=None)
    intensity_std = np.sqrt(variance, dtype=np.float64)

    max_intensity = np.empty_like(mean_intensity)
    for idx, label_id in enumerate(label_ids):
        crypt_mask = flattened_labels == label_id
        max_intensity[idx] = float(np.max(flattened_intensity[crypt_mask]))

    total_intensity_sum = float(np.sum(intensity_sums))
    intensity_sum_mean = float(np.mean(intensity_sums))
    intensity_sum_std = _safe_std(intensity_sums, count=crypt_count)

    intensity_mean_mean = float(np.mean(mean_intensity))
    intensity_mean_std = _safe_std(mean_intensity, count=crypt_count)
    intensity_mean_min = float(np.min(mean_intensity))
    intensity_mean_max = float(np.max(mean_intensity))

    max_intensity_mean = float(np.mean(max_intensity))
    max_intensity_std = _safe_std(max_intensity, count=crypt_count)

    positive_intensity = np.clip(flattened_intensity, a_min=0.0, a_max=None)
    positive_intensity_sums = np.bincount(
        flattened_labels,
        weights=positive_intensity,
        minlength=max_label + 1,
    )[1:]
    positive_intensity_sums = positive_intensity_sums[valid_mask]

    effective_bound = intensity_upper_bound
    if effective_bound is None:
        effective_bound = float(np.max(positive_intensity))
        if effective_bound == 0.0:
            effective_bound = 1.0
    if effective_bound <= 0.0:
        raise ValueError("intensity_upper_bound must be positive.")

    effective_full_um2 = (positive_intensity_sums / effective_bound) * pixel_area_um2
    effective_full_sum = float(np.sum(effective_full_um2))
    effective_full_mean = float(np.mean(effective_full_um2))
    effective_full_std = _safe_std(effective_full_um2, count=crypt_count)

    return np.array(
        [
            float(crypt_count),
            area_sum,
            area_mean,
            area_std,
            total_intensity_sum,
            intensity_sum_mean,
            intensity_sum_std,
            intensity_mean_mean,
            intensity_mean_std,
            intensity_mean_min,
            intensity_mean_max,
            max_intensity_mean,
            max_intensity_std,
            effective_full_sum,
            effective_full_mean,
            effective_full_std,
        ],
        dtype=np.float64,
    )


def summarize_crypt_fluorescence(
    normalized_rfp: np.ndarray | None = None,
    crypt_labels: np.ndarray | None = None,
    microns_per_px: float | np.ndarray | None = None,
    *,
    channels: Sequence[Any] | None = None,
    masks: Sequence[Any] | None = None,
    intensity_upper_bound: float | None = None,
) -> dict[str, float]:
    """
    Summarize crypt-level fluorescence and geometric properties.

    Parameters
    ----------
    normalized_rfp, crypt_labels, microns_per_px
        Core data arrays describing fluorescence intensity, labeled crypt
        regions, and the spatial scale. Provide them directly or via
        ``channels`` as ``[normalized_rfp, crypt_labels, microns_per_px]``.
    channels
        Optional container holding the three core inputs (and potentially
        additional metadata). This keeps compatibility with the legacy
        Analysis Stack contract.
    masks
        Present for signature compatibility with the Analysis Stack contract;
        unused.
    intensity_upper_bound
        Optional maximum intensity representing 100% fluorescence.

    Returns
    -------
    dict
        Mapping from ``SUMMARY_FIELD_ORDER`` names to Python ``float`` values.
        ``float('nan')`` is used where a statistic is undefined.
    """
    del masks  # unused but required by the stack contract

    normalized_rfp, crypt_labels, microns_per_px = _resolve_primary_inputs(
        normalized_rfp, crypt_labels, microns_per_px, channels
    )

    normalized_rfp = _as_image(normalized_rfp, dtype=np.float64)
    crypt_labels = _as_image(crypt_labels)
    microns_per_px = _as_scalar_float(microns_per_px)

    metrics_array = _summarize_single_subject(
        normalized_rfp=normalized_rfp,
        crypt_labels=crypt_labels,
        microns_per_px=microns_per_px,
        intensity_upper_bound=intensity_upper_bound,
    )
    return {
        field: float(metrics_array[idx])
        for idx, field in enumerate(SUMMARY_FIELD_ORDER)
    }


def _compute_per_crypt_records(
    *,
    normalized_rfp: np.ndarray,
    crypt_labels: np.ndarray,
    microns_per_px: float,
    subject_name: str,
) -> list[dict[str, Any]]:
    """Compute per-crypt statistics for a single subject."""
    if normalized_rfp.shape != crypt_labels.shape:
        raise ValueError("normalized_rfp and crypt_labels must have matching shapes.")

    label_image = np.asarray(crypt_labels, dtype=np.int64)
    if not np.any(label_image):
        return []

    max_label = int(label_image.max())
    if max_label <= 0:
        return []

    flattened_labels = label_image.ravel()
    flattened_intensity = np.asarray(normalized_rfp, dtype=np.float64).ravel()

    pixel_counts = np.bincount(flattened_labels, minlength=max_label + 1)[1:]
    valid_mask = pixel_counts > 0
    if not np.any(valid_mask):
        return []

    label_ids = np.nonzero(valid_mask)[0] + 1
    pixel_counts = pixel_counts[valid_mask].astype(np.int64)

    intensity_sums = np.bincount(
        flattened_labels,
        weights=flattened_intensity,
        minlength=max_label + 1,
    )[1:]
    intensity_sums = intensity_sums[valid_mask]

    intensity_sq_sums = np.bincount(
        flattened_labels,
        weights=flattened_intensity * flattened_intensity,
        minlength=max_label + 1,
    )[1:]
    intensity_sq_sums = intensity_sq_sums[valid_mask]

    mean_intensity = np.divide(
        intensity_sums,
        pixel_counts,
        out=np.zeros_like(intensity_sums),
        where=pixel_counts > 0,
    )

    variance = np.divide(
        intensity_sq_sums,
        pixel_counts,
        out=np.zeros_like(intensity_sq_sums),
        where=pixel_counts > 0,
    ) - mean_intensity * mean_intensity
    variance = np.clip(variance, a_min=0.0, a_max=None)
    intensity_std = np.sqrt(variance, dtype=np.float64)

    min_intensity = np.full_like(mean_intensity, np.inf, dtype=np.float64)
    max_intensity = np.full_like(mean_intensity, -np.inf, dtype=np.float64)
    for idx, label_id in enumerate(label_ids):
        mask = flattened_labels == label_id
        if not np.any(mask):
            min_intensity[idx] = float("nan")
            max_intensity[idx] = float("nan")
            continue
        values = flattened_intensity[mask]
        min_intensity[idx] = float(np.min(values))
        max_intensity[idx] = float(np.max(values))

    microns_per_px = float(microns_per_px)
    pixel_area_um2 = (microns_per_px * microns_per_px) * pixel_counts.astype(np.float64)

    records: list[dict[str, Any]] = []
    for idx, label_id in enumerate(label_ids):
        record = {
            "subject_name": subject_name,
            "crypt_label": int(label_id),
            "crypt_index": idx,
            "pixel_area": int(pixel_counts[idx]),
            "pixel_value_sum": float(intensity_sums[idx]),
            "pixel_value_mean": float(mean_intensity[idx]),
            "pixel_value_std": float(intensity_std[idx]),
            "pixel_value_min": float(min_intensity[idx]),
            "pixel_value_max": float(max_intensity[idx]),
            "um_area": float(pixel_area_um2[idx]),
            "um_value_sum": float(intensity_sums[idx] * microns_per_px * microns_per_px),
            "um_value_mean": float(mean_intensity[idx]),
            "um_value_std": float(intensity_std[idx]),
            "um_value_min": float(min_intensity[idx]),
            "um_value_max": float(max_intensity[idx]),
            "microns_per_px": microns_per_px,
        }
        records.append(record)

    return records


def summarize_crypt_fluorescence_per_crypt(
    normalized_rfp: np.ndarray | None = None,
    crypt_labels: np.ndarray | None = None,
    microns_per_px: float | np.ndarray | None = None,
    *,
    subject_name: str | None = None,
    channels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """
    Summarize per-crypt fluorescence metrics for a single subject.

    Parameters
    ----------
    normalized_rfp, crypt_labels, microns_per_px
        Primary inputs describing fluorescence, labels, and spatial scale.
        Provide them directly or include them in ``channels``.
    channels
        Optional legacy container ``[normalized_rfp, crypt_labels, microns_per_px, subject_name?]``.
    subject_name
        Optional explicit subject identifier; falls back to ``channels[3]`` or ``"subject"``.

    Returns
    -------
    dict
        Dictionary with keys ``subject_name``, ``records`` (list of dicts following
        ``PER_CRYPT_FIELD_ORDER``), ``field_order`` (tuple of field names), and
        ``record_count``.
    """
    normalized_rfp, crypt_labels, microns_per_px = _resolve_primary_inputs(
        normalized_rfp, crypt_labels, microns_per_px, channels
    )

    if subject_name is None and channels is not None and len(channels) >= 4:
        subject_name = channels[3]
    subject_name = _coerce_subject_name(subject_name)

    normalized_rfp = _as_image(normalized_rfp)
    crypt_labels = _as_image(crypt_labels)
    microns_per_px = _as_scalar_float(microns_per_px)

    records = _compute_per_crypt_records(
        normalized_rfp=normalized_rfp,
        crypt_labels=crypt_labels,
        microns_per_px=microns_per_px,
        subject_name=subject_name,
    )
    return {
        "subject_name": subject_name,
        "records": records,
        "field_order": PER_CRYPT_FIELD_ORDER,
        "record_count": len(records),
    }


__all__ = [
    "SUMMARY_FIELD_ORDER",
    "PER_CRYPT_FIELD_ORDER",
    "summarize_crypt_fluorescence",
    "summarize_crypt_fluorescence_per_crypt",
]
