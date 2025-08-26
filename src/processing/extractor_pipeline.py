"""
Extractor pipeline for blob detection using watershed segmentation.
Based on the exact working notebook algorithm.
"""

import numpy as np
from skimage.segmentation import expand_labels, watershed, find_boundaries
from scipy.ndimage import label as ndi_label, distance_transform_edt
from skimage import morphology
import cv2

from utils.image_utils import minmax01
from utils.file_utils import build_rgb


class ExtractorPipeline:
    """Pipeline for extracting labeled blobs from red/blue channel images."""
    
    def __init__(self, debug=False):
        """
        Initialize the extractor pipeline.
        
        Args:
            debug: Whether to enable debug mode
        """
        self.debug = debug
        self.debug_info = {}
    
    def extract(self, red_img, blue_img):
        """
        Extract labeled blobs from red and blue channel images.
        This exactly matches the working notebook algorithm.
        
        Args:
            red_img: Red channel image as float32 array
            blue_img: Blue channel image as float32 array
        
        Returns:
            Labeled array where each blob has a unique integer label
        """
        if red_img.shape != blue_img.shape:
            raise ValueError(f"Image shape mismatch: red {red_img.shape} vs blue {blue_img.shape}")
        
        # Store original images for debug
        if self.debug:
            self.debug_info['red_original'] = red_img.copy()
            self.debug_info['blue_original'] = blue_img.copy()
            print(f"[EXTRACTOR DEBUG] Input red: shape {red_img.shape}, range [{red_img.min():.2f}, {red_img.max():.2f}]")
            print(f"[EXTRACTOR DEBUG] Input blue: shape {blue_img.shape}, range [{blue_img.min():.2f}, {blue_img.max():.2f}]")
        
        # Build RGB display image to match notebook exactly
        disp = build_rgb(red_img, blue_img)
        
        # Extract red and blue channels from the RGB display (this is key!)
        # This matches the notebook: red = disp[...,0].astype(np.float32)
        red = disp[..., 0].astype(np.float32)
        blue = disp[..., 2].astype(np.float32)
        
        if self.debug:
            self.debug_info['disp_rgb'] = disp.copy()
            self.debug_info['red_from_disp'] = red.copy()
            self.debug_info['blue_from_disp'] = blue.copy()
            print(f"[EXTRACTOR DEBUG] Red from display: range [{red.min():.2f}, {red.max():.2f}]")
            print(f"[EXTRACTOR DEBUG] Blue from display: range [{blue.min():.2f}, {blue.max():.2f}]")
        
        # Simple morphological reconstruction style differences (matching notebook exactly)
        mask_r_dilation = np.maximum(blue, red)
        mask_r_erosion = np.minimum(blue, red)
        
        if self.debug:
            self.debug_info['mask_r_dilation'] = mask_r_dilation.copy()
            self.debug_info['mask_r_erosion'] = mask_r_erosion.copy()
            print(f"[EXTRACTOR DEBUG] Dilation mask range: [{mask_r_dilation.min():.2f}, {mask_r_dilation.max():.2f}]")
            print(f"[EXTRACTOR DEBUG] Erosion mask range: [{mask_r_erosion.min():.2f}, {mask_r_erosion.max():.2f}]")
        
        # diff_r: red stronger than min envelope (exact notebook logic)
        diff_r = red > mask_r_erosion
        
        if self.debug:
            self.debug_info['diff_r_raw'] = diff_r.copy()
            print(f"[EXTRACTOR DEBUG] diff_r raw: {np.sum(diff_r)} pixels")
        
        diff_r = morphology.binary_erosion(diff_r, footprint=np.ones((3, 3)))
        diff_r = morphology.remove_small_objects(diff_r, min_size=100)
        
        if self.debug:
            self.debug_info['diff_r'] = diff_r.copy()
            print(f"[EXTRACTOR DEBUG] diff_r final: {np.sum(diff_r)} pixels after erosion and cleanup")
        
        # Secondary mask using absolute difference (exact notebook logic)
        abs_diff = np.abs(mask_r_dilation - red)
        mask_gt_red = abs_diff > red
        
        if self.debug:
            self.debug_info['abs_diff'] = abs_diff.copy()
            self.debug_info['mask_gt_red'] = mask_gt_red.copy()
            print(f"[EXTRACTOR DEBUG] abs_diff range: [{abs_diff.min():.2f}, {abs_diff.max():.2f}]")
            print(f"[EXTRACTOR DEBUG] mask_gt_red: {np.sum(mask_gt_red)} pixels")
        
        # Erode the secondary mask (exact notebook parameters)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask_u8 = (mask_gt_red.astype(np.uint8) * 255)
        mask_eroded_u8 = cv2.erode(mask_u8, erosion_kernel, iterations=2)
        mask_gt_red_eroded = mask_eroded_u8.astype(bool)
        
        if self.debug:
            self.debug_info['mask_gt_red_eroded'] = mask_gt_red_eroded.copy()
            print(f"[EXTRACTOR DEBUG] mask_gt_red_eroded: {np.sum(mask_gt_red_eroded)} pixels")
        
        # Build combined_labels (0 bg, 1 diff_r, 2 mask_gt_red) - exact notebook
        combined_labels = np.zeros_like(diff_r, dtype=int)
        combined_labels[mask_gt_red_eroded] = 2
        combined_labels[diff_r] = 1
        
        if self.debug:
            self.debug_info['combined_labels'] = combined_labels.copy()
            unique_combined = np.unique(combined_labels)
            counts = [(label, np.sum(combined_labels == label)) for label in unique_combined]
            print(f"[EXTRACTOR DEBUG] Combined labels: {counts}")
        
        # Expand labels (exact notebook distance=100)
        expanded_labels = expand_labels(combined_labels, distance=100)
        
        if self.debug:
            self.debug_info['expanded_labels'] = expanded_labels.copy()
            unique_expanded = np.unique(expanded_labels)
            print(f"[EXTRACTOR DEBUG] Expanded labels: {len(unique_expanded)} unique values")
            if len(unique_expanded) <= 20:
                print(f"[EXTRACTOR DEBUG] Expanded values: {unique_expanded}")
        
        # Markers from diff_r (exact notebook logic)
        labeled_diff_r, _ = ndi_label(diff_r != 0)
        
        if self.debug:
            self.debug_info['labeled_diff_r'] = labeled_diff_r.copy()
            print(f"[EXTRACTOR DEBUG] labeled_diff_r: max {labeled_diff_r.max()} regions")
        
        # Reworked markers array (exact notebook logic)
        reworked = np.zeros_like(expanded_labels, dtype=np.int32)
        reworked[expanded_labels == 2] = 1  # entire class 2 region => marker 1
        mask_copy = (expanded_labels != 2) & (labeled_diff_r != 0)
        reworked[mask_copy] = labeled_diff_r[mask_copy] + 1
        
        if self.debug:
            self.debug_info['reworked'] = reworked.copy()
            unique_reworked = np.unique(reworked)
            print(f"[EXTRACTOR DEBUG] Reworked markers: {len(unique_reworked)} unique, max: {reworked.max()}")
        
        # Watershed mask (exact notebook logic)
        mask_ws = expanded_labels > 0
        
        if self.debug:
            self.debug_info['mask_ws'] = mask_ws.copy()
            print(f"[EXTRACTOR DEBUG] Watershed mask: {np.sum(mask_ws)} pixels")
        
        # Elevation: attract to class 2, repel from class 1 (exact notebook)
        elevation = (
            minmax01(distance_transform_edt(combined_labels == 2))
            - minmax01(distance_transform_edt(combined_labels == 1))
        )
        
        if self.debug:
            self.debug_info['elevation'] = elevation.copy()
            print(f"[EXTRACTOR DEBUG] Elevation range: [{elevation.min():.3f}, {elevation.max():.3f}]")
        
        # Apply watershed (exact notebook logic)
        ws_labels = watershed(elevation, markers=reworked, mask=mask_ws)
        
        if self.debug:
            self.debug_info['ws_labels'] = ws_labels.copy()
            unique_ws = np.unique(ws_labels)
            print(f"[EXTRACTOR DEBUG] Final watershed: {len(unique_ws)} regions, max label: {ws_labels.max()}")
            if len(unique_ws) <= 20:
                print(f"[EXTRACTOR DEBUG] Final labels: {unique_ws}")

        # Remove background label (1) and relabel others sequentially starting from 1
        ws_labels[ws_labels == 1] = 0
        ws_labels[ws_labels > 1] = ws_labels[ws_labels > 1] - 1
        return ws_labels

    def get_debug_info(self):
        """
        Get debug information from the last extraction.
        
        Returns:
            Dictionary containing debug information
        """
        return self.debug_info.copy()
    
    def clear_debug_info(self):
        """Clear stored debug information."""
        self.debug_info.clear()
