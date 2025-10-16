from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.measure import label as sk_label
from skimage.morphology import (
    binary_erosion,
    binary_opening,
    disk,
    dilation,
    local_maxima,
    opening,
    remove_small_objects,
)
from skimage.segmentation import expand_labels, watershed
from skimage.util import invert


# ---------------------------- utilities ---------------------------- #


def to_float01(img: np.ndarray) -> np.ndarray:
    """Return an array scaled to [0, 1] float without modifying NaNs/Inf."""
    if img.dtype in (np.float32, np.float64):
        return np.clip(img, 0.0, 1.0)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return np.clip(img.astype(np.float32) / float(info.max), 0.0, 1.0)
    arr = img.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def minmax(x: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] (safe if constant)."""
    x = x.astype(np.float32, copy=False)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def remove_rectangles(
    image: NDArray[np.uint8],
    *,
    stacks: Optional[NDArray[np.uint8]] = None,
    white_thresh: int = 240,
    same_tol: int = 5,
    area_min: int = 10,
    aspect_low: float = 0.2,
    aspect_high: float = 5.0,
    dilation_kernel: Tuple[int, int] = (15, 15),
    inpaint_radius: int = 15,
) -> NDArray[np.uint8]:
    """
    Remove rectangular artifacts + flat-white text/graphics that are the same across stacks.
    - If `stacks` is provided (N, H, W), 'same across stacks' uses that.
    - Otherwise, uses across-channel flatness on the provided RGB image.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
    else:
        gray = image
        height, width = gray.shape

    _, binary_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect_ratio = w / h
        if aspect_ratio < aspect_low or aspect_ratio > aspect_high:
            cv2.rectangle(rect_mask, (x, y), (x + w, y + h), 255, -1)

    if stacks is not None:
        if stacks.ndim != 3 or stacks.shape[1:] != (height, width):
            raise ValueError("`stacks` must be shaped (N, H, W) and match image size.")
        stacks = stacks.astype(np.int16)
        flat = (stacks.max(axis=0) - stacks.min(axis=0)) <= same_tol
        bright = stacks.mean(axis=0) >= white_thresh
        flat_white_mask = (flat & bright).astype(np.uint8) * 255
    else:
        if image.ndim == 3 and image.shape[2] >= 3:
            channels = image[:, :, :3].astype(np.int16)
            flat = (channels.max(axis=2) - channels.min(axis=2)) <= same_tol
            bright = channels.mean(axis=2) >= white_thresh
            flat_white_mask = (flat & bright).astype(np.uint8) * 255
        else:
            flat_white_mask = np.zeros((height, width), dtype=np.uint8)

    if flat_white_mask.any():
        flat_white_mask = cv2.dilate(flat_white_mask, np.ones((3, 3), np.uint8), iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (flat_white_mask > 0).astype(np.uint8), connectivity=8
        )
        kept = np.zeros((height, width), dtype=np.uint8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= area_min:
                kept[labels == i] = 255
        flat_white_mask = kept

    mask = rect_mask.copy()
    if flat_white_mask.any():
        mask = cv2.bitwise_or(mask, flat_white_mask)

    if not mask.any():
        return image.copy()

    mask = cv2.dilate(mask, np.ones(dilation_kernel, np.uint8), iterations=1)
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ---------------------------- morphology helpers ---------------------------- #


def caps(image: np.ndarray, small_r: int, big_r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contrast-adaptive cap decomposition."""
    image = to_float01(image)
    image = equalize_adapthist(image, clip_limit=0.01)

    hats = dilation(image, disk(big_r)) - dilation(image, disk(small_r))
    hats = minmax(hats)

    clean = image - np.minimum(image, hats)
    clean = minmax(clean)

    troughs = np.maximum(image, hats) - image
    troughs = minmax(troughs)
    return hats, clean, troughs


def show_caps(image: np.ndarray, small_r: int, big_r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visualize the cap decomposition for debugging."""
    hats, clean, troughs = caps(image, small_r, big_r)

    fig, ax = plt.subplots(1, 4, figsize=(30, 5))
    for axis in ax:
        axis.axis("off")
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(hats, cmap="gray")
    ax[2].imshow(clean, cmap="gray")
    ax[3].imshow(troughs, cmap="gray")
    fig.colorbar(ax[1].images[0], ax=ax[1])
    fig.colorbar(ax[2].images[0], ax=ax[2])
    fig.colorbar(ax[3].images[0], ax=ax[3])
    fig.tight_layout()
    return hats, clean, troughs


def identify_crypt_seeds(
    crypt_img: NDArray,
    tissue_image: NDArray,
    blob_size_px: Optional[int] = None,
) -> NDArray:
    """Legacy seed generator kept for reference."""
    del blob_size_px  # legacy placeholder
    tissue_hats, tissue_clean, tissue_troughs = caps(equalize_adapthist(tissue_image), 1, 40)
    crypt_hats, crypt_clean, _ = caps(crypt_img, 2, 40)

    thinned_crypts = np.maximum(crypt_clean - tissue_clean, 0)
    split_crypts = np.maximum(crypt_clean - tissue_troughs, 0)

    good_crypts = minmax(opening(split_crypts * thinned_crypts, footprint=disk(5)) ** 0.5)
    distance = tissue_troughs - good_crypts
    maxi = local_maxima(good_crypts)
    crypt_seeds = watershed(distance, markers=maxi, mask=tissue_troughs < good_crypts)
    return sk_label(crypt_seeds > 0)


def limited_expansion(
    crypt_img: NDArray,
    tissue_image: NDArray,
    blob_size_px: Optional[int] = None,
) -> NDArray:
    """Limit vertical expansion of crypts by suppressing tissue spillover."""
    del blob_size_px  # not used but kept for interface parity
    _, _, outer_troughs = caps(minmax(tissue_image), 10, 50)
    outer_troughs = minmax(outer_troughs)

    adjusted = np.maximum(crypt_img - outer_troughs, 0)
    crypt_max = binary_opening(local_maxima(adjusted, indices=False), footprint=disk(1)).astype(bool)
    tissue_max = binary_opening(local_maxima(outer_troughs), disk(2))

    alt = invert(dilation(minmax(adjusted), footprint=disk(2)) + minmax(outer_troughs))
    spread = watershed(image=alt, markers=np.maximum(crypt_max * 1, tissue_max * 2))
    return spread


def identify_crypt_seeds_new(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
) -> np.ndarray:
    """Newer seed logic that prefers solid crypts."""
    crypt_img = to_float01(crypt_img)
    tissue_image = to_float01(tissue_image)

    _, tissue_clean, tissue_troughs = caps(tissue_image, 1, 40)
    _, crypt_clean, crypt_troughs = caps(crypt_img, 2, 40)

    thinned_crypts = np.maximum(crypt_clean - tissue_clean, 0)
    split_crypts = np.maximum(crypt_clean - tissue_troughs, 0)

    good_crypts = minmax(opening(split_crypts * thinned_crypts, footprint=disk(5)) ** 0.5)
    distance = tissue_troughs - good_crypts
    maxi = local_maxima(good_crypts)
    seeds = watershed(distance, markers=maxi, mask=tissue_troughs < good_crypts)
    return sk_label(seeds > 0)


def identify_potential_crypts_old_like(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    blob_size_px: Optional[int] = 30,
    external_crypt_seeds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Original watershed skeleton with optional external seeds."""
    crypt_img = to_float01(crypt_img)
    tissue_image = to_float01(tissue_image)

    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Image shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    effective_blob = float(blob_size_px) if blob_size_px else 1.0
    erosion_dim = _odd(max(3, int(round(effective_blob / 10.0))))
    erosion_footprint = np.ones((erosion_dim, erosion_dim), dtype=bool)

    if external_crypt_seeds is None:
        crypt_seeds_bool = crypt_img > np.minimum(tissue_image, crypt_img)
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seeds_bool = binary_erosion(crypt_seeds_bool, footprint=erosion_footprint)
        crypt_seeds_bool = remove_small_objects(crypt_seeds_bool, min_size=min_region_area)
        labeled_diff_r, _ = ndi_label(crypt_seeds_bool)
    else:
        external_crypt_seeds = external_crypt_seeds.astype(np.int32)
        crypt_seed_mask = external_crypt_seeds > 0
        min_region_area = max(20, int(round((effective_blob**2) / 16.0)))
        crypt_seed_mask = remove_small_objects(crypt_seed_mask, min_size=min_region_area)
        labeled_diff_r = sk_label(crypt_seed_mask)

    abs_diff = np.maximum(tissue_image - crypt_img, 0)
    mask_gt_red = abs_diff > crypt_img

    erosion_kernel_size = max(3, int(round(effective_blob * 0.15)))
    erosion_kernel_size = erosion_kernel_size if erosion_kernel_size % 2 == 0 else erosion_kernel_size + 1
    cv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    mask_u8 = (mask_gt_red.astype(np.uint8) * 255)
    erosion_iterations = max(1, int(round(effective_blob / 20.0)))
    mask_eroded_u8 = cv2.erode(mask_u8, cv_kernel, iterations=erosion_iterations)
    tissue_eroded = mask_eroded_u8.astype(bool)

    combined_labels = np.zeros_like(labeled_diff_r, dtype=int)
    combined_labels[tissue_eroded] = 2
    combined_labels[labeled_diff_r > 0] = 1

    expand_distance = max(1, int(round(effective_blob * 2.5)))
    expanded_labels = expand_labels(combined_labels, distance=expand_distance)

    reworked = np.zeros_like(expanded_labels, dtype=np.int32)
    reworked[expanded_labels == 2] = 1
    mask_copy = (expanded_labels != 2) & (labeled_diff_r != 0)
    reworked[mask_copy] = labeled_diff_r[mask_copy] + 1

    mask_ws = expanded_labels > 0

    elevation = minmax(distance_transform_edt(combined_labels == 2)) - minmax(
        distance_transform_edt(combined_labels == 1)
    )

    ws_labels = watershed(elevation, markers=reworked, mask=mask_ws).copy()
    ws_labels[ws_labels == 1] = 0
    ws_labels[ws_labels > 1] -= 1
    return ws_labels.astype(np.int32)


def _seed_health_metrics(seed_labels: np.ndarray) -> Dict[str, Any]:
    """Compute basic metrics to judge seed quality."""
    if seed_labels is None or seed_labels.size == 0:
        return dict(n_labels=0, coverage=0.0, mean_area=0.0)
    n_labels = int(seed_labels.max())
    coverage = float(np.count_nonzero(seed_labels)) / seed_labels.size
    if n_labels > 0:
        areas = np.bincount(seed_labels.ravel())[1:]
        mean_area = float(areas.mean()) if areas.size else 0.0
    else:
        mean_area = 0.0
    return dict(n_labels=n_labels, coverage=coverage, mean_area=mean_area)


def identify_potential_crypts_hybrid(
    crypt_img: np.ndarray,
    tissue_image: np.ndarray,
    blob_size_px: Optional[int] = 30,
    use_new_seeds: bool = True,
    auto_fallback: bool = True,
    min_seed_count: int = 10,
    min_coverage: float = 0.002,
    debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Hybrid pipeline:
      1) Try the new seed generator.
      2) Optionally fall back to original seed logic if new seeds are weak.
      3) Run the original watershed body with the chosen seeds.
    """
    crypt_img = to_float01(crypt_img)
    tissue_image = to_float01(tissue_image)

    dbg: Dict[str, Any] = {}

    new_seeds = identify_crypt_seeds_new(crypt_img, tissue_image) if use_new_seeds else None
    new_metrics = _seed_health_metrics(
        new_seeds if new_seeds is not None else np.zeros_like(crypt_img, dtype=np.int32)
    )
    dbg["new_seed_metrics"] = new_metrics

    use_new = use_new_seeds
    if auto_fallback and use_new_seeds:
        use_new = (new_metrics["n_labels"] >= min_seed_count) and (new_metrics["coverage"] >= min_coverage)

    dbg["using_new_seeds"] = bool(use_new)

    if use_new:
        labels = identify_potential_crypts_old_like(
            crypt_img,
            tissue_image,
            blob_size_px=blob_size_px,
            external_crypt_seeds=new_seeds,
        )
        dbg["mode"] = "old_body_with_new_seeds"
    else:
        labels = identify_potential_crypts_old_like(
            crypt_img,
            tissue_image,
            blob_size_px=blob_size_px,
            external_crypt_seeds=None,
        )
        dbg["mode"] = "old_body_with_old_seeds"

    if debug:
        dbg["labels_max"] = int(labels.max())
        dbg["labels_coverage"] = float(np.count_nonzero(labels)) / labels.size

    return labels, dbg


# ---------------------------- driver ---------------------------- #


def build_image_sets() -> list[Tuple[str, np.ndarray, np.ndarray]]:
    """Load source images and return [(subject, crypt_channel, tissue_channel), ...]."""
    jej2 = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c2.tif"
    )
    jej2_tissue = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2c1.tif"
    )
    jej3 = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c2.tif"
    )
    jej3_tissue = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-3c1.tif"
    )
    g3fr_rgb = tiff.imread("/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2.tif")
    g3fr_sep_rfp = tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_RFP.tif"
    )
    g3fr_sep_dapi = tiff.imread(
        "/home/phillip/documents/lysozyme/lysozyme images/Jej LYZ/G3/G3FR - 2_DAPI.tif"
    )

    evil_g3fr = remove_rectangles(g3fr_rgb)

    return [
        ("Jej-2", jej2[..., 1], jej2_tissue[..., 2]),
        ("Jej-3", jej3[..., 1], jej3_tissue[..., 2]),
        ("G3FR", evil_g3fr[..., 0], evil_g3fr[..., 2]),
        ("G3FR_sep", g3fr_sep_rfp[..., 0], g3fr_sep_dapi[..., 2]),
    ]


def _load_reference_image() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single reference RGB image (crypt, crypt channel, dapi channel)."""
    image = tiff.imread(
        "/home/phillip/documents/yen-lab-discussion/rfp/Lyz Fabp1/CDKO 158.1/Jej-2.tif"
    )
    crypt_channel = minmax(image[:, :, 1])
    dapi_channel = minmax(image[:, :, 2])
    first_channel = minmax(image[:, :, 0])
    return first_channel, crypt_channel, dapi_channel


def main() -> None:
    list_of_images_to_try = build_image_sets()
    crypt_img = minmax(list_of_images_to_try[0][1])
    tissue_image = minmax(list_of_images_to_try[0][2])

    _first_channel, _crypts, _dapi = _load_reference_image()
    del _first_channel, _crypts, _dapi  # kept for parity with notebook but unused

    scale = 5
    width = 4
    fig, ax = plt.subplots(len(list_of_images_to_try), width, figsize=(scale * width, len(list_of_images_to_try) * scale))

    for i, (subject, crypt, tissue) in enumerate(list_of_images_to_try):
        crypt = minmax(crypt)
        tissue = minmax(tissue)

        ax[i, 0].imshow(np.stack([crypt, np.zeros_like(crypt), tissue], axis=-1))
        ax[i, 0].axis("off")
        ax[i, 0].set_title(f"{subject} - Crypt")

        old_labels = identify_potential_crypts_old_like(crypt, tissue)
        ax[i, 1].imshow(label2rgb(old_labels))
        ax[i, 1].axis("off")
        ax[i, 1].set_title(f"{subject} - Old identify potential crypts")

        hybrid_labels, _ = identify_potential_crypts_hybrid(crypt, tissue)
        ax[i, 2].imshow(label2rgb(hybrid_labels), cmap="nipy_spectral")
        ax[i, 2].axis("off")
        ax[i, 2].set_title(f"{subject} - Hybrid identify potential crypts")

        new_seeds = identify_crypt_seeds_new(crypt, tissue)
        ax[i, 3].imshow(label2rgb(new_seeds), cmap="nipy_spectral")
        ax[i, 3].axis("off")
        ax[i, 3].set_title(f"{subject} - New")

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "crypt_segmentation_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
