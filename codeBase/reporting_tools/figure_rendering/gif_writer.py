from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == bool:
        return (arr.astype(np.uint8)) * 255
    # scale to 0-255 based on min/max per sequence could vary, but we do per-frame here
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if amax == amin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - amin) / (amax - amin)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def write_gif(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: int = 14,
    loop: int = 0,
) -> Path:
    """Write a sequence of 2D frames to a GIF using imageio.

    Parameters
    - frames: iterable of 2D numpy arrays (grayscale). Each will be normalized to uint8.
    - output_path: path to save the GIF.
    - fps: frames per second for the GIF.
    - loop: number of loops (0 = loop forever).
    """
    from imageio import v2 as imageio

    if not frames:
        raise ValueError("No frames provided to write_gif")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames_u8: List[np.ndarray] = [_to_uint8(f) for f in frames]
    duration = 1.0 / float(fps)
    imageio.mimsave(out, frames_u8, format="GIF", duration=duration, loop=loop)
    return out

