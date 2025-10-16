import numpy as np
import cv2

from skimage import morphology
from skimage.segmentation import expand_labels, watershed
from scipy.ndimage import label as ndi_label, distance_transform_edt
from typing import Tuple
from numpy.typing import DTypeLike, ArrayLike



def minmax01(x, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(float, copy=False)
    lo = np.min(x)
    hi = np.max(x)
    return (x - lo) / max(hi - lo, eps)


def _build_rgb(red_gray: np.ndarray, blue_gray: np.ndarray) -> np.ndarray:
    def to_u8(arr):
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            lo = np.nanmin(arr)
            hi = np.nanmax(arr)
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = np.zeros_like(arr)
            return arr.astype(np.uint8)
        return arr

    r8 = to_u8(red_gray)
    b8 = to_u8(blue_gray)
    zeros = np.zeros_like(r8)
    return np.stack([r8, zeros, b8], axis=-1)


def _calculate_intensity_metrics(ws_labels: np.ndarray, red_img: np.ndarray, blue_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    background_mask = ws_labels == 1
    background_tissue_intensity = 0.0
    if np.any(background_mask):
        red_bg = red_img[background_mask]
        blue_bg = blue_img[background_mask]
        valid_mask = blue_bg > 1e-10
        if np.any(valid_mask):
            background_tissue_intensity = np.mean(red_bg[valid_mask] / blue_bg[valid_mask])

    crypt_mask = ws_labels > 1
    average_crypt_intensity = 0.0
    if np.any(crypt_mask):
        red_crypt = red_img[crypt_mask]
        blue_crypt = blue_img[crypt_mask]
        valid_mask = blue_crypt > 1e-10
        if np.any(valid_mask):
            average_crypt_intensity = np.mean(red_crypt[valid_mask] / blue_crypt[valid_mask])

    return background_tissue_intensity, average_crypt_intensity


def identify_potential_crypts(crypt_img, tissue_image, blob_size_px: int = 30, debug: bool = False) -> np.ndarray:
    """Identify potential crypt regions using the original watershed pipeline."""

    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Image shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")

    red_img = crypt_img.astype(np.float32, copy=False)
    blue_img = tissue_image.astype(np.float32, copy=False)

    debug_info = {} 
    if debug:
        print(
            f"[EXTRACTOR DEBUG] Input red: shape {red_img.shape}, range [{red_img.min():.2f}, {red_img.max():.2f}]"
        )
        print(
            f"[EXTRACTOR DEBUG] Input blue: shape {blue_img.shape}, range [{blue_img.min():.2f}, {blue_img.max():.2f}]"
        )

    disp = _build_rgb(red_img, blue_img)
    red = disp[..., 0].astype(np.float32)
    blue = disp[..., 2].astype(np.float32)

    if debug:
        print(f"[EXTRACTOR DEBUG] Red from display: range [{red.min():.2f}, {red.max():.2f}]")
        print(f"[EXTRACTOR DEBUG] Blue from display: range [{blue.min():.2f}, {blue.max():.2f}]")

    mask_r_dilation = np.maximum(blue, red)
    mask_r_erosion = np.minimum(blue, red)

    if debug:
        debug_info['mask_r_dilation'] = mask_r_dilation.copy()
        debug_info['mask_r_erosion'] = mask_r_erosion.copy()
        print(
            f"[EXTRACTOR DEBUG] Dilation mask range: [{mask_r_dilation.min():.2f}, {mask_r_dilation.max():.2f}]"
        )
        print(
            f"[EXTRACTOR DEBUG] Erosion mask range: [{mask_r_erosion.min():.2f}, {mask_r_erosion.max():.2f}]"
        )

    effective_blob = float(blob_size_px) if blob_size_px else 1.0
    erosion_dim = max(3, int(round(effective_blob / 10.0)))
    if erosion_dim % 2 == 0:
        erosion_dim += 1
    erosion_footprint = np.ones((erosion_dim, erosion_dim), dtype=bool)

    diff_r: np.ndarray = red > mask_r_erosion

    if debug:
        debug_info['diff_r_raw'] = diff_r.copy()
        print(f"[EXTRACTOR DEBUG] diff_r raw: {np.sum(diff_r)} pixels")

    min_region_area = max(20, int(round((effective_blob ** 2) / 16.0)))
    diff_r = morphology.binary_erosion(diff_r, footprint=erosion_footprint)
    diff_r = morphology.remove_small_objects(diff_r, min_size=min_region_area)

    if debug:
        debug_info['diff_r'] = diff_r.copy()
        print(
            f"[EXTRACTOR DEBUG] diff_r final: {np.sum(diff_r)} pixels after erosion and cleanup"
        )
        print(
            f"[EXTRACTOR DEBUG] erosion_dim={erosion_dim}, min_region_area={min_region_area}"
        )

    abs_diff = np.abs(mask_r_dilation - red)
    mask_gt_red = abs_diff > red

    if debug:
        debug_info['abs_diff'] = abs_diff.copy()
        debug_info['mask_gt_red'] = mask_gt_red.copy()
        print(f"[EXTRACTOR DEBUG] abs_diff range: [{abs_diff.min():.2f}, {abs_diff.max():.2f}]")
        print(f"[EXTRACTOR DEBUG] mask_gt_red: {np.sum(mask_gt_red)} pixels")

    erosion_kernel_size = max(3, int(round(effective_blob * 0.15)))
    if erosion_kernel_size % 2 == 1:
        erosion_kernel_size += 1
    cv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    mask_u8 = mask_gt_red.astype(np.uint8) * 255
    erosion_iterations = max(1, int(round(effective_blob / 20.0)))
    mask_eroded_u8 = cv2.erode(mask_u8, cv_kernel, iterations=erosion_iterations)
    mask_gt_red_eroded = mask_eroded_u8.astype(bool)

    if debug:
        debug_info['mask_gt_red_eroded'] = mask_gt_red_eroded.copy()
        print(f"[EXTRACTOR DEBUG] mask_gt_red_eroded: {np.sum(mask_gt_red_eroded)} pixels")
        print(
            f"[EXTRACTOR DEBUG] erosion_kernel_size={erosion_kernel_size}, iterations={erosion_iterations}"
        )

    combined_labels = np.zeros_like(diff_r, dtype=int)
    combined_labels[mask_gt_red_eroded] = 2
    combined_labels[diff_r] = 1

    if debug:
        debug_info['combined_labels'] = combined_labels.copy()
        unique_combined = np.unique(combined_labels)
        counts = [(label, np.sum(combined_labels == label)) for label in unique_combined]
        print(f"[EXTRACTOR DEBUG] Combined labels: {counts}")

    expand_distance = max(1, int(round(effective_blob * 2.5)))
    expanded_labels = expand_labels(combined_labels, distance=expand_distance)

    if debug:
        debug_info['expanded_labels'] = expanded_labels.copy()
        unique_expanded = np.unique(expanded_labels)
        print(f"[EXTRACTOR DEBUG] Expanded labels: {len(unique_expanded)} unique values")
        if len(unique_expanded) <= 20:
            print(f"[EXTRACTOR DEBUG] Expanded values: {unique_expanded}")
        print(f"[EXTRACTOR DEBUG] expand_distance={expand_distance}")

    labeled_diff_r, _ = ndi_label(diff_r != 0)

    if debug:
        debug_info['labeled_diff_r'] = labeled_diff_r.copy()
        print(f"[EXTRACTOR DEBUG] labeled_diff_r: max {labeled_diff_r.max()} regions")

    reworked = np.zeros_like(expanded_labels, dtype=np.int32)
    reworked[expanded_labels == 2] = 1
    mask_copy = (expanded_labels != 2) & (labeled_diff_r != 0)
    reworked[mask_copy] = labeled_diff_r[mask_copy] + 1

    if debug:
        debug_info['reworked'] = reworked.copy()
        unique_reworked = np.unique(reworked)
        print(
            f"[EXTRACTOR DEBUG] Reworked markers: {len(unique_reworked)} unique, max: {reworked.max()}"
        )

    mask_ws = expanded_labels > 0

    if debug:
        debug_info['mask_ws'] = mask_ws.copy()
        print(f"[EXTRACTOR DEBUG] Watershed mask: {np.sum(mask_ws)} pixels")

    elevation = (
        minmax01(distance_transform_edt(combined_labels == 2))
        - minmax01(distance_transform_edt(combined_labels == 1))
    )

    if debug:
        debug_info['elevation'] = elevation.copy()
        print(f"[EXTRACTOR DEBUG] Elevation range: [{elevation.min():.3f}, {elevation.max():.3f}]")

    ws_labels = watershed(elevation, markers=reworked, mask=mask_ws)

    if debug:
        debug_info['ws_labels'] = ws_labels.copy()
        unique_ws = np.unique(ws_labels)
        print(
            f"[EXTRACTOR DEBUG] Final watershed: {len(unique_ws)} regions, max label: {ws_labels.max()}"
        )
        if len(unique_ws) <= 20:
            print(f"[EXTRACTOR DEBUG] Final labels: {unique_ws}")

    background_tissue_intensity, average_crypt_intensity = _calculate_intensity_metrics(
        ws_labels, red_img, blue_img
    )

    ws_labels = ws_labels.copy()
    ws_labels[ws_labels == 1] = 0
    ws_labels[ws_labels > 1] = ws_labels[ws_labels > 1] - 1

    identify_potential_crypts.last_intensity_metrics = (
        background_tissue_intensity,
        average_crypt_intensity,
    )
    if debug:
        debug_info['background_tissue_intensity'] = background_tissue_intensity
        debug_info['average_crypt_intensity'] = average_crypt_intensity
        identify_potential_crypts.last_debug_info = debug_info
    else:
        identify_potential_crypts.last_debug_info = {}

    return ws_labels


identify_potential_crypts.last_intensity_metrics = (0.0, 0.0)
identify_potential_crypts.last_debug_info = {}
