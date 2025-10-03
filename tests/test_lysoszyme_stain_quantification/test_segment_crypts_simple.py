
import pytest
import numpy as np
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts

def generate_test_crypts(
        image_size: tuple[int, int] = (100, 100),
        crypt_centers: list[tuple[int, int]] = [(40, 40), (70, 40)],
        tissue_box: tuple[tuple[int, int], tuple[int, int]] = ((20, 0), (80, 100)),
        crypt_radius: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two grayscale test images (red and blue channels represented as 2D arrays).
    - Background: 0.05
    - Red image: tissue box = 0.2, crypt circles = 0.8 (override tissue)
    - Blue image: tissue box = 0.5
    Returns (red_image, blue_image) as float32 arrays in range [0, 1].
    """
    h, w = image_size
    # initialize backgrounds
    red = np.full((h, w), 0.05, dtype=np.float32)
    blue = np.full((h, w), 0.05, dtype=np.float32)

    # unpack tissue box (assume ((x0, y0), (x1, y1)))
    (x0, y0), (x1, y1) = tissue_box
    # clamp to image bounds and convert to int indices
    x0i, x1i = max(0, int(x0)), min(w, int(x1))
    y0i, y1i = max(0, int(y0)), min(h, int(y1))

    # fill tissue box
    red[y0i:y1i, x0i:x1i] = 0.2
    blue[y0i:y1i, x0i:x1i] = 0.5

    # prepare coordinate grid for circles (centers given as (x, y))
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    rr2 = crypt_radius * crypt_radius
    for cx, cy in crypt_centers:
        # create circular mask and set red channel to crypt intensity
        mask = (xx - int(cx))**2 + (yy - int(cy))**2 <= rr2
        red[mask] = 0.8

    return red, blue










