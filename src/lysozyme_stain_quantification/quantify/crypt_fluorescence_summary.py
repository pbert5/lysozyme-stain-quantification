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

    normalized_rfp = np.asarray(channels[0], dtype=np.float64)
    crypt_labels = np.asarray(channels[1])
    microns_per_px = _as_scalar_float(channels[2])

    if normalized_rfp.shape != crypt_labels.shape:
        raise ValueError("normalized_rfp and crypt_labels must share the same shape.")
    if microns_per_px <= 0.0:
        raise ValueError("microns_per_px must be a positive scalar.")

    if not np.any(crypt_labels):
        # No crypts means we can return zero counts and NaNs for per-crypt statistics.
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
    flattened_intensity = normalized_rfp.ravel()

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

    if intensity_upper_bound is None:
        intensity_upper_bound = float(np.max(positive_intensity))
        if intensity_upper_bound == 0.0:
            intensity_upper_bound = 1.0
    if intensity_upper_bound <= 0.0:
        raise ValueError("intensity_upper_bound must be positive.")

    effective_full_um2 = (positive_intensity_sums / intensity_upper_bound) * pixel_area_um2
    effective_full_sum = float(np.sum(effective_full_um2))
    effective_full_mean = float(np.mean(effective_full_um2))
    effective_full_std = _safe_std(effective_full_um2, count=crypt_count)

    summary = np.array(
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

    return summary


__all__ = ["SUMMARY_FIELD_ORDER", "summarize_crypt_fluorescence"]
