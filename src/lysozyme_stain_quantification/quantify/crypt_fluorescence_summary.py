"""
Quantify per-crypt fluorescence and geometry for lysozyme staining experiments.
"""

from __future__ import annotations

from typing import Any, Sequence, Iterable

import numpy as np
import xarray as xr

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
    channels: Sequence[Any],
    *,
    masks: Sequence[Any] | None = None,
    intensity_upper_bound: float | None = None,
) -> np.ndarray:
    """
    Summarize crypt-level fluorescence and geometric properties.

    Parameters
    ----------
    channels
        Sequence containing [normalized_rfp, crypt_labels, microns_per_px].
        ``normalized_rfp`` should contain the fluorescence intensity values,
        ``crypt_labels`` is the integer label image (0 indicates background),
        and ``microns_per_px`` is a scalar conveying the pixel scale.
    masks
        Present for signature compatibility with the AnalysisStack contract;
        unused.
    intensity_upper_bound
        Optional maximum intensity value representing the 100% fluorescence
        reference. When omitted the maximum non-negative pixel intensity from
        ``normalized_rfp`` is used. This value is used when computing the
        ``effective_full_intensity`` metrics (interpretable as the number of
        square microns that would be fully saturated at the reference level).

    Returns
    -------
    numpy.ndarray
        A one-dimensional array with values ordered according to
        ``SUMMARY_FIELD_ORDER``. Consumers can call ``.tolist()`` to obtain
        a plain Python list suitable for spreadsheet export.

    Raises
    ------
    ValueError
        If the provided channels are missing, mismatched in shape, or lack
        crypt annotations.
    """
    del masks  # unused but required by the stack contract

    if len(channels) < 3:
        raise ValueError(
            "summarize_crypt_fluorescence expects channels=[normalized_rfp, crypt_labels, microns_per_px]."
        )

    normalized_input = channels[0]
    crypt_input = channels[1]
    microns_input = channels[2]

    if isinstance(normalized_input, xr.DataArray) and "subject" in normalized_input.dims:
        if not isinstance(crypt_input, xr.DataArray) or "subject" not in crypt_input.dims:
            raise ValueError("crypt_labels must be an xarray DataArray with a 'subject' dimension.")
        if not isinstance(microns_input, xr.DataArray) or "subject" not in microns_input.dims:
            raise ValueError("microns_per_px must be an xarray DataArray with a 'subject' dimension.")

        subjects = list(_ensure_subject_order(normalized_input))
        metrics_list: list[np.ndarray] = []
        for subject_coord in subjects:
            rfp = _as_image(normalized_input.sel(subject=subject_coord).values, dtype=np.float64)
            labels = _as_image(crypt_input.sel(subject=subject_coord).values)
            microns_value = _as_scalar_float(microns_input.sel(subject=subject_coord).values)
            metrics = _summarize_single_subject(
                normalized_rfp=rfp,
                crypt_labels=labels,
                microns_per_px=microns_value,
                intensity_upper_bound=intensity_upper_bound,
            )
            metrics_list.append(metrics)

        metric_count = len(SUMMARY_FIELD_ORDER)
        data = (
            np.stack(metrics_list, axis=0)
            if metrics_list
            else np.empty((len(subjects), metric_count), dtype=np.float64)
        )
        return xr.DataArray(
            data,
            dims=("subject", "metric"),
            coords={
                "subject": subjects,
                "metric": np.asarray(SUMMARY_FIELD_ORDER, dtype=object),
            },
            name="crypt_fluorescence_summary",
        )

    normalized_rfp = _as_image(normalized_input, dtype=np.float64)
    crypt_labels = _as_image(crypt_input)
    microns_per_px = _as_scalar_float(microns_input)

    metrics = _summarize_single_subject(
        normalized_rfp=normalized_rfp,
        crypt_labels=crypt_labels,
        microns_per_px=microns_per_px,
        intensity_upper_bound=intensity_upper_bound,
    )
    return xr.DataArray(
        metrics,
        dims=("metric",),
        coords={"metric": np.asarray(SUMMARY_FIELD_ORDER, dtype=object)},
        name="crypt_fluorescence_summary",
    )




def _ensure_subject_order(subject_array: xr.DataArray) -> Iterable:
    """Return iterable of subject coordinate values in analysis stack order."""
    if "subject" not in subject_array.dims:
        raise ValueError("Expected a 'subject' dimension in the provided DataArray.")
    return list(subject_array.coords["subject"].values)


def _has_subject_dim(value: Any) -> bool:
    """Return True when value is an xarray DataArray that includes a subject dimension."""
    return isinstance(value, xr.DataArray) and "subject" in value.dims


def _extract_subject_name(subject_coord_value: Any, fallback: Any) -> str:
    """Return the best-available subject name string."""
    candidate = subject_coord_value if subject_coord_value is not None else fallback
    if isinstance(candidate, (str, bytes)):
        return candidate.decode("utf-8") if isinstance(candidate, bytes) else candidate
    if hasattr(candidate, "item"):
        try:
            item = candidate.item()
            return _extract_subject_name(item, fallback=None)
        except Exception:
            pass
    return str(candidate)


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
    *,
    channels: Sequence[Any],
    masks: Sequence[Any] | None = None,
) -> xr.DataArray:
    """
    Collect per-crypt statistics for every subject as an object-valued DataArray.

    Parameters
    ----------
    channels
        Expected sequence: [normalized_rfp, crypt_labels, microns_per_px, subject_name?].
        When the optional ``subject_name`` channel is omitted the subject coordinate
        is used for labeling.
    masks
        Present for signature compatibility; unused.

    Returns
    -------
    xarray.DataArray
        A one-dimensional array with ``subject`` dimension. Each element contains a
        ``list`` of ``dict`` records with fields defined by ``PER_CRYPT_FIELD_ORDER``.
    """
    del masks  # unused

    if len(channels) < 3:
        raise ValueError(
            "summarize_crypt_fluorescence_per_crypt expects channels="
            "[normalized_rfp, crypt_labels, microns_per_px, (optional) subject_name]."
        )

    normalized_rfp_input = channels[0]
    crypt_labels_input = channels[1]
    microns_per_px_input = channels[2]
    subject_name_input = channels[3] if len(channels) > 3 else None

    if _has_subject_dim(normalized_rfp_input):
        normalized_rfp_da = normalized_rfp_input
        crypt_labels_da = crypt_labels_input
        microns_per_px_da = microns_per_px_input
        subject_name_da = subject_name_input

        if not isinstance(crypt_labels_da, xr.DataArray) or "subject" not in crypt_labels_da.dims:
            raise ValueError("crypt_labels channel must be an xarray DataArray with a 'subject' dimension.")
        if not isinstance(microns_per_px_da, xr.DataArray) or "subject" not in microns_per_px_da.dims:
            raise ValueError("microns_per_px channel must be an xarray DataArray with a 'subject' dimension.")

        subjects = _ensure_subject_order(normalized_rfp_da)
        numeric_fields = [field for field in PER_CRYPT_FIELD_ORDER if field != "subject_name"]
        subject_names: list[str] = []
        records_per_subject: list[list[dict[str, Any]]] = []

        for idx, subject_coord in enumerate(subjects):
            rfp_sel = normalized_rfp_da.sel(subject=subject_coord)
            labels_sel = crypt_labels_da.sel(subject=subject_coord)
            microns_sel = microns_per_px_da.sel(subject=subject_coord)

            rfp = np.asarray(rfp_sel.values)
            labels = np.asarray(labels_sel.values)
            microns_arr = np.asarray(microns_sel.values).reshape(-1)
            if microns_arr.size == 0:
                raise ValueError("microns_per_px channel yielded no values for a subject.")
            microns_value = float(microns_arr[0])

            if subject_name_da is not None:
                subject_value = subject_name_da.sel(subject=subject_coord).values
                subject_name = _extract_subject_name(subject_value, subject_coord)
            else:
                subject_name = _extract_subject_name(subject_coord, fallback=None)

            records = _compute_per_crypt_records(
                normalized_rfp=rfp,
                crypt_labels=labels,
                microns_per_px=microns_value,
                subject_name=subject_name,
            )
            subject_names.append(subject_name)
            records_per_subject.append(records)

        record_counts = np.array([len(records) for records in records_per_subject], dtype=np.int32)
        max_records = int(record_counts.max(initial=0))
        if max_records == 0:
            data = np.empty((len(subjects), 0, len(numeric_fields)), dtype=np.float64)
        else:
            data = np.full((len(subjects), max_records, len(numeric_fields)), np.nan, dtype=np.float64)
            for subj_idx, records in enumerate(records_per_subject):
                for rec_idx, record in enumerate(records):
                    for field_idx, field in enumerate(numeric_fields):
                        data[subj_idx, rec_idx, field_idx] = float(record[field])

        coords = {
            "subject": subjects,
            "crypt": np.arange(max_records, dtype=np.int32),
            "field": numeric_fields,
            "record_count": ("subject", record_counts),
            "subject_display_name": ("subject", subject_names),
        }
        return xr.DataArray(
            data,
            dims=("subject", "crypt", "field"),
            coords=coords,
            name="crypt_fluorescence_per_crypt",
        )

    # When use_apply_ufunc=True the function receives single-subject numpy arrays.
    rfp = _as_image(normalized_rfp_input)
    labels = _as_image(crypt_labels_input)
    microns_arr = np.asarray(microns_per_px_input).reshape(-1)
    if microns_arr.size == 0:
        raise ValueError("microns_per_px channel yielded no values for a subject.")
    microns_value = float(microns_arr[0])

    if subject_name_input is not None:
        subject_name = _extract_subject_name(subject_name_input, fallback="subject")
    else:
        subject_name = "subject"

    numeric_fields = [field for field in PER_CRYPT_FIELD_ORDER if field != "subject_name"]
    records = _compute_per_crypt_records(
        normalized_rfp=rfp,
        crypt_labels=labels,
        microns_per_px=microns_value,
        subject_name=subject_name,
    )
    data = np.empty((len(records), len(numeric_fields)), dtype=np.float64)
    for rec_idx, record in enumerate(records):
        for field_idx, field in enumerate(numeric_fields):
            data[rec_idx, field_idx] = float(record[field])
    return data


__all__ = [
    "SUMMARY_FIELD_ORDER",
    "PER_CRYPT_FIELD_ORDER",
    "summarize_crypt_fluorescence",
    "summarize_crypt_fluorescence_per_crypt",
]
