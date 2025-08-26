"""
Watershed refinement using expanded labels over RFP/DAPI pairs.
This module provides blob detection functionality using watershed segmentation
for lysozyme stain quantification.
"""

import numpy as np
from skimage.segmentation import expand_labels, watershed, find_boundaries
from scipy.ndimage import label as ndi_label, distance_transform_edt
import tifffile
import cv2
from pathlib import Path
from skimage import morphology
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def remove_rectangles(image, white_thresh=240, aspect_low=0.2, aspect_high=5.0,
                     dilation_kernel=(15, 15), inpaint_radius=15):
    """Remove rectangular artifacts from image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, bm = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0: 
            continue
        ar = w / h
        if ar < aspect_low or ar > aspect_high:
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            
    if not mask.any(): 
        return image.copy()
    mask = cv2.dilate(mask, np.ones(dilation_kernel, np.uint8), 1)
    bgr = image[..., ::-1]
    out = cv2.inpaint(bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return out[..., ::-1]


def load_as_gray(p: Path) -> np.ndarray:
    """Load an image file as grayscale float32 array."""
    arr = tifffile.imread(p)
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[..., 0]
    return arr.astype(np.float32)


def build_rgb(red_gray: np.ndarray, blue_gray: np.ndarray) -> np.ndarray:
    """Build RGB image from red and blue grayscale channels."""
    def to_u8(x):
        if x.dtype != np.uint8:
            lo, hi = np.nanmin(x), np.nanmax(x)
            if hi > lo:
                x = (x - lo)/(hi-lo)*255.0
            else:
                x = np.zeros_like(x)
            return x.astype(np.uint8)
        return x
    
    r8 = to_u8(red_gray)
    b8 = to_u8(blue_gray)
    zeros = np.zeros_like(r8)
    return np.stack([r8, zeros, b8], axis=-1)


def minmax01(x: np.ndarray, eps=1e-12) -> np.ndarray:
    """Normalize array to [0,1] range."""
    x = x.astype(float, copy=False)
    lo = np.min(x)
    hi = np.max(x)
    return (x - lo) / max(hi - lo, eps)


def get_rfp_dapi_pairs(images_root: Path, max_pairs: int = 30) -> List[Tuple[Path, Path]]:
    """Find RFP/DAPI image pairs in the given directory."""
    red_files = sorted(list(images_root.rglob('*_RFP.tif')) + list(images_root.rglob('*_RFP.tiff')))
    
    def _match_blue(r_path: Path) -> Optional[Path]:
        stem = r_path.name
        if '_RFP.' not in stem:
            return None
        base = stem.split('_RFP.')[0]
        for ext in ['tif', 'tiff', 'TIF', 'TIFF']:
            cand = r_path.with_name(f'{base}_DAPI.{ext}')
            if cand.exists():
                return cand
        return None
    
    pairs = []
    for r_file in red_files:
        b_file = _match_blue(r_file)
        if b_file is not None:
            pairs.append((r_file, b_file))
        if len(pairs) >= max_pairs:
            break
    
    return pairs


class BlobDetector:
    """Watershed-based blob detector for lysozyme stain quantification."""
    
    def __init__(self, channel: int = 0, debug: bool = False):
        self.channel = channel
        self.debug = debug
    
    def detect(self, image: np.ndarray, segmentation_low_thresh: int = 30,
               segmentation_high_thresh: int = 150, scale_bar_intensity_threshold: int = 240,
               singleton_penalty: int = 10) -> np.ndarray:
        """
        Detect blobs using watershed segmentation.
        
        Args:
            image: Input image array
            segmentation_low_thresh: Low threshold for segmentation
            segmentation_high_thresh: High threshold for segmentation
            scale_bar_intensity_threshold: Threshold for scale bar detection
            singleton_penalty: Penalty for singleton objects
            
        Returns:
            Label array with detected blobs
        """
        # For now, return a simple placeholder that maintains compatibility
        # This can be expanded to use the watershed method for single images
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Simple thresholding as placeholder
        _, binary = cv2.threshold(gray.astype(np.uint8), segmentation_low_thresh, 255, cv2.THRESH_BINARY)
        labels, _ = ndi_label(binary)
        
        return labels
    
    def process_rfp_dapi_pair(self, rfp_path: Path, dapi_path: Path) -> np.ndarray:
        """
        Process an RFP/DAPI image pair using watershed refinement.
        
        Args:
            rfp_path: Path to RFP image
            dapi_path: Path to DAPI image
            
        Returns:
            Watershed labels array
        """
        try:
            r = load_as_gray(rfp_path)
            b = load_as_gray(dapi_path)
        except Exception as e:
            print(f'[skip] read error {rfp_path.name}/{dapi_path.name}: {e}')
            return np.array([])
            
        if r.shape != b.shape:
            print(f'[skip] shape mismatch {r.shape} vs {b.shape}: {rfp_path.name}')
            return np.array([])
            
        disp = build_rgb(r, b)
        try:
            disp = remove_rectangles(disp)
        except Exception:
            pass

        # Red / Blue channels for logic
        red = disp[..., 0].astype(np.float32)
        blue = disp[..., 2].astype(np.float32)

        # Simple morphological reconstruction style differences
        mask_r_dilation = np.maximum(blue, red)
        mask_r_erosion = np.minimum(blue, red)
        
        # diff_r: red stronger than min envelope
        diff_r = red > mask_r_erosion
        diff_r = morphology.binary_erosion(diff_r, footprint=np.ones((3, 3)))
        diff_r = morphology.remove_small_objects(diff_r, min_size=100)

        # Secondary mask using absolute difference
        abs_diff = np.abs(mask_r_dilation - red)
        mask_gt_red = abs_diff > red
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask_u8 = (mask_gt_red.astype(np.uint8) * 255)
        mask_eroded_u8 = cv2.erode(mask_u8, erosion_kernel, iterations=2)
        mask_gt_red_eroded = mask_eroded_u8.astype(bool)

        # Build combined_labels (0 bg, 1 diff_r, 2 mask_gt_red)
        combined_labels = np.zeros_like(diff_r, dtype=int)
        combined_labels[mask_gt_red_eroded] = 2
        combined_labels[diff_r] = 1

        # Expand labels
        expanded_labels = expand_labels(combined_labels, distance=100)

        # Markers from diff_r
        labeled_diff_r, _ = ndi_label(diff_r != 0)

        # Reworked markers array
        reworked = np.zeros_like(expanded_labels, dtype=np.int32)
        reworked[expanded_labels == 2] = 1  # entire class 2 region => marker 1
        mask_copy = (expanded_labels != 2) & (labeled_diff_r != 0)
        reworked[mask_copy] = labeled_diff_r[mask_copy] + 1

        # Watershed mask
        mask_ws = expanded_labels > 0

        # Elevation: attract to class 2, repel from class 1
        elevation = (
            minmax01(distance_transform_edt(combined_labels == 2))
            - minmax01(distance_transform_edt(combined_labels == 1))
        )

        ws_labels = watershed(elevation, markers=reworked, mask=mask_ws)
        
        return ws_labels
    
    def visualize_watershed_results(self, rfp_path: Path, dapi_path: Path, 
                                  ws_labels: np.ndarray, project_root: Path,
                                  show_plot: bool = True) -> np.ndarray:
        """
        Create visualization overlay for watershed results.
        
        Args:
            rfp_path: Path to RFP image
            dapi_path: Path to DAPI image  
            ws_labels: Watershed labels array
            project_root: Root path of project
            show_plot: Whether to display the plot
            
        Returns:
            Overlay image with boundaries
        """
        # Load and build RGB display
        r = load_as_gray(rfp_path)
        b = load_as_gray(dapi_path)
        disp = build_rgb(r, b)
        
        try:
            disp = remove_rectangles(disp)
        except Exception:
            pass
        
        # Create overlay with boundaries
        boundaries = find_boundaries(ws_labels, mode='inner')
        overlay_ws = disp.copy()
        overlay_ws[boundaries] = [255, 0, 0]
        
        if show_plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(ws_labels, cmap='tab20')
            plt.title('Watershed labels')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(overlay_ws)
            plt.title('Overlay (boundaries)')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            # Show original for comparison
            plt.imshow(disp)
            plt.title('Original RFP/DAPI')
            plt.axis('off')
            
            rel_name = rfp_path.relative_to(project_root / 'lysozyme images').as_posix().replace('_RFP', '')
            plt.suptitle(f'Watershed Results: {rel_name}')
            plt.tight_layout()
            plt.show()
        
        return overlay_ws


def process_all_rfp_dapi_pairs(project_root: Path, max_show: int = 30, 
                              show_plots: bool = True) -> List[np.ndarray]:
    """
    Process all RFP/DAPI pairs in the project using watershed refinement.
    
    Args:
        project_root: Root directory of the project
        max_show: Maximum number of pairs to process
        show_plots: Whether to show visualization plots
        
    Returns:
        List of watershed label arrays
    """
    images_root = project_root / 'lysozyme images'
    pairs = get_rfp_dapi_pairs(images_root, max_pairs=max_show)
    
    if not pairs:
        print('No RFP/DAPI pairs found. Aborting watershed processing.')
        return []
    
    print(f"Found {len(pairs)} RFP/DAPI pairs to process")
    
    detector = BlobDetector(debug=True)
    results = []
    
    for idx, (r_fp, b_fp) in enumerate(pairs, 1):
        print(f"Processing pair {idx}/{len(pairs)}: {r_fp.name}")
        
        ws_labels = detector.process_rfp_dapi_pair(r_fp, b_fp)
        
        if ws_labels.size > 0:
            results.append(ws_labels)
            
            if show_plots:
                detector.visualize_watershed_results(
                    r_fp, b_fp, ws_labels, project_root, show_plot=True
                )
        else:
            print(f"  Skipped due to processing error")
    
    print(f"Successfully processed {len(results)} image pairs")
    return results


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    results = process_all_rfp_dapi_pairs(project_root, max_show=5, show_plots=True)
    print(f"Processed {len(results)} pairs successfully")
