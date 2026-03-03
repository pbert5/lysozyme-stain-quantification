from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import xarray as xr


def render_label_overlay(
    *,
    channels: Sequence[np.ndarray],
    masks: Sequence[np.ndarray] | None = None,
    fill_alpha: float = 0.35,
    outline_alpha: float = 1.0,
    outline_width: int = 1,
    normalize_scalar: bool = True,
    color_seed: int = 12345,
) -> xr.DataArray:
    """Render label overlays on top of an RGB background derived from input channels.

    Parameters
    ----------
    channels:
        Expects either (scalar, labels) or (rfp, dapi, labels). When two channels are
        provided, the first is used as a grayscale background replicated across RGB. When
        three channels are given, the first two are interpreted as RFP and DAPI signals,
        blended into an RGB background before overlays are applied.
    masks:
        Present for contract compatibility; currently unused.
    fill_alpha:
        Blend factor applied to the filled interior of labeled regions. Set to zero to
        disable interior tinting.
    outline_alpha:
        Blend factor applied to the label outlines.
    outline_width:
        Thickness of the outline in pixels (>= 1).
    normalize_scalar:
        When ``True``, input scalar channels are normalized to ``[0, 1]`` per subject
        before rendering. When ``False`` the values are clipped to ``[0, 1]``.
    color_seed:
        Seed controlling the deterministic pseudo-random colors assigned per label.

    Returns
    -------
    xr.DataArray
        A ``(3, Y, X)`` float32 array representing an RGB visualization.
    """
    if len(channels) < 2:
        raise ValueError("render_label_overlay expects at least two channels.")

    if masks:
        _ = masks  # masks are unused but keep signature compatibility

    def _normalize_plane(plane: np.ndarray) -> np.ndarray:
        finite = np.isfinite(plane)
        if not finite.any():
            return np.zeros_like(plane, dtype=np.float32)
        data = plane[finite]
        vmin = float(data.min())
        vmax = float(data.max())
        if vmax > vmin:
            normalized = (plane - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(plane, dtype=np.float32)
        normalized[~finite] = 0.0
        return normalized

    def _squeeze_if_needed(arr: np.ndarray) -> np.ndarray:
        if arr.ndim > 2 and arr.shape[0] == 1:
            return np.squeeze(arr, axis=0)
        return arr

    def _prep_scalar(arr: np.ndarray) -> np.ndarray:
        data = np.asarray(arr, dtype=np.float32)
        data = _squeeze_if_needed(data)
        if normalize_scalar:
            return _normalize_plane(data)
        return np.clip(data, 0.0, 1.0)

    if len(channels) >= 3:
        rfp = _prep_scalar(channels[0])
        dapi = _prep_scalar(channels[1])
        labels = np.asarray(channels[2])
        labels = _squeeze_if_needed(labels)
        if rfp.ndim != 2 or dapi.ndim != 2:
            raise ValueError("rfp and dapi channels must be 2D arrays.")
        if rfp.shape != dapi.shape:
            raise ValueError(f"rfp shape {rfp.shape} does not match dapi shape {dapi.shape}.")
    else:
        rfp = _prep_scalar(channels[0])
        dapi = rfp
        labels = np.asarray(channels[1])
        labels = _squeeze_if_needed(labels)

    if labels.ndim != 2:
        raise ValueError(f"Label channel must be 2D; received shape {labels.shape}.")
    if labels.shape != rfp.shape:
        raise ValueError(f"Label channel shape {labels.shape} does not match image {rfp.shape}.")
    if outline_width < 1:
        raise ValueError("outline_width must be at least 1 pixel.")

    base_rgb = np.stack([rfp, np.zeros_like(rfp), dapi], axis=0)

    def _erode(binary: np.ndarray) -> np.ndarray:
        pad = np.pad(binary, 1, mode="constant", constant_values=False)
        return (
            pad[1:-1, 1:-1]
            & pad[:-2, 1:-1]
            & pad[2:, 1:-1]
            & pad[1:-1, :-2]
            & pad[1:-1, 2:]
            & pad[:-2, :-2]
            & pad[:-2, 2:]
            & pad[2:, :-2]
            & pad[2:, 2:]
        )

    def _dilate(binary: np.ndarray) -> np.ndarray:
        pad = np.pad(binary, 1, mode="constant", constant_values=False)
        return (
            pad[1:-1, 1:-1]
            | pad[:-2, 1:-1]
            | pad[2:, 1:-1]
            | pad[1:-1, :-2]
            | pad[1:-1, 2:]
            | pad[:-2, :-2]
            | pad[:-2, 2:]
            | pad[2:, :-2]
            | pad[2:, 2:]
        )

    def _outline_edges(binary_mask: np.ndarray, width: int) -> np.ndarray:
        base_outline = binary_mask & ~_erode(binary_mask)
        if width <= 1:
            return base_outline
        current = base_outline
        for _ in range(width - 1):
            current = _dilate(current)
        return current

    def _color_for_label(label_value: int) -> np.ndarray:
        seed = (int(label_value) * 1315423911 + color_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        color = rng.random(3, dtype=np.float32)
        return 0.25 + 0.75 * color  # avoid very dark colors

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    for label_value in unique_labels:
        component_mask = labels == label_value
        if not np.any(component_mask):
            continue
        color = _color_for_label(label_value)

        if fill_alpha > 0.0:
            blend = (1.0 - fill_alpha) * base_rgb + fill_alpha * color[:, np.newaxis, np.newaxis]
            base_rgb = np.where(
                component_mask[np.newaxis, ...],
                blend,
                base_rgb,
            )

        outline_mask = _outline_edges(component_mask, outline_width)
        if outline_alpha > 0.0 and np.any(outline_mask):
            blend_outline = (
                (1.0 - outline_alpha) * base_rgb + outline_alpha * color[:, np.newaxis, np.newaxis]
            )
            base_rgb = np.where(outline_mask[np.newaxis, ...], blend_outline, base_rgb)

    output = np.clip(base_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    return xr.DataArray(
        output,
        dims=("channel", "y", "x"),
        coords={
            "channel": np.asarray(["r", "g", "b"], dtype=object),
            "y": np.arange(output.shape[1]),
            "x": np.arange(output.shape[2]),
        },
        name="label_overlay",
    )
