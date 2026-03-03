from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import numpy as np


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Return a 2D view of arr, selecting a channel if needed.

    - If arr is 2D, returns as-is.
    - If arr is 3D with a small channel dim (<= 4), selects the first channel.
    - Otherwise, squeezes singleton dims and attempts to return a 2D view.
    """
    if arr.ndim == 2:
        return arr
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        # Try channels-last first
        if squeezed.shape[-1] <= 4:
            return squeezed[..., 0]
        # Fallback to channels-first
        if squeezed.shape[0] <= 4:
            return squeezed[0]
    # As a last resort, flatten spatial and reshape square-ish if possible
    return np.squeeze(arr)


def _normalize_radii(
    start_radius: int,
    frame_count: int,
    min_radius: int = 2,
    decreasing: bool = True,
) -> List[int]:
    """Generate a list of integer radii across frames.

    Ensures:
    - start_radius >= min_radius and min_radius >= 2
    - Exactly frame_count entries are returned
    - Monotonic sequence (decreasing by default)
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if start_radius is None:
        raise ValueError("start_radius is required")
    if start_radius < 2:
        raise ValueError("Kernel radius must be at least 2 pixels")
    if min_radius < 2:
        raise ValueError("min_radius must be at least 2 pixels")

    start = int(start_radius)
    stop = int(min_radius)
    if not decreasing:
        start, stop = stop, start
    # linspace inclusive ends; cast to int; clip to [min_radius, start_radius]
    seq = np.linspace(start, stop, num=frame_count)
    radii = np.clip(np.rint(seq).astype(int), min(start_radius, min_radius), max(start_radius, min_radius))
    return radii.tolist()


def generate_morph_sequence(
    image: np.ndarray,
    op: Callable[..., np.ndarray],
    kernel_fn: Callable[[int], np.ndarray],
    radius: int,
    frame_count: int,
    *,
    min_radius: int = 2,
    decreasing: bool = True,
    op_kwargs: dict | None = None,
) -> List[np.ndarray]:
    """Generate a sequence of frames by applying a morphology op with shrinking kernels.

    Parameters
    - image: 2D or 3D numpy array (grayscale or multi-channel); reduced to 2D.
    - op: skimage morphology op, e.g., skimage.morphology.dilation/erosion.
    - kernel_fn: function that takes integer radius -> footprint array.
    - radius: starting kernel radius (pixels). Must be >= 2.
    - frame_count: number of frames to generate.
    - min_radius: minimum allowed radius (inclusive). Defaults to 2.
    - decreasing: if True, radii go from radius down to min_radius; otherwise increasing.
    - op_kwargs: extra kwargs for op (e.g., behavior flags); footprint is injected.

    Returns
    - List of 2D numpy arrays (same shape as input image reduced to 2D).

    Notes
    - Works for binary and grayscale images. For boolean, result stays boolean.
    - Prevents 1px (or smaller) kernels: raises if radius < 2 or min_radius < 2.
    """
    if op is None or not callable(op):
        raise ValueError("op must be a callable skimage morphology operation")
    if kernel_fn is None or not callable(kernel_fn):
        raise ValueError("kernel_fn must be callable and accept an integer radius")

    base = _to_2d(np.asarray(image))
    radii = _normalize_radii(radius, frame_count, min_radius=min_radius, decreasing=decreasing)
    frames: List[np.ndarray] = []
    kwargs = dict(op_kwargs or {})

    for r in radii:
        footprint = kernel_fn(int(r))
        # some kernel functions may produce empty/1px results for small r, guard again
        if footprint is None or footprint.size == 0:
            raise ValueError("kernel_fn produced an empty footprint")
        if min(footprint.shape) <= 1:
            raise ValueError("kernel_fn produced a <=1px footprint; increase min_radius")
        result = op(base, footprint=footprint, **kwargs)
        frames.append(result)

    return frames

