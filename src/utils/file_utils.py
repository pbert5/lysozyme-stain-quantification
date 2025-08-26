"""
Utility functions for file operations and validation.
"""

from pathlib import Path
from typing import List, Tuple, Union
import tifffile


def validate_directories(img_dir: Path, results_dir: Path) -> bool:
    """
    Validate that directories exist and are accessible.
    
    Args:
        img_dir: Path to image directory
        results_dir: Path to results directory
    
    Returns:
        True if validation passes, False otherwise
    """
    if not img_dir.exists():
        print(f"Error: Image directory does not exist: {img_dir}")
        return False
    
    if not img_dir.is_dir():
        print(f"Error: Image path is not a directory: {img_dir}")
        return False
    
    # Create results directory if it doesn't exist
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create results directory {results_dir}: {e}")
        return False
    
    return True


def find_image_pairs(img_dir: Path, red_channel: str, blue_channel: str) -> List[Tuple[Path, Path]]:
    """
    Find paired red and blue channel TIFF files in the image directory.
    
    Args:
        img_dir: Directory to search for images
        red_channel: String identifier for red channel (e.g., "RFP")
        blue_channel: String identifier for blue channel (e.g., "DAPI")
    
    Returns:
        List of tuples containing (red_path, blue_path)
    """
    # Find all red channel files
    red_pattern_tif = f"*_{red_channel}.tif"
    red_pattern_tiff = f"*_{red_channel}.tiff"
    
    red_files = (
        list(img_dir.rglob(red_pattern_tif)) +
        list(img_dir.rglob(red_pattern_tiff)) +
        list(img_dir.rglob(red_pattern_tif.upper())) +
        list(img_dir.rglob(red_pattern_tiff.upper()))
    )
    
    pairs = []
    
    for red_path in red_files:
        blue_path = _find_matching_blue(red_path, red_channel, blue_channel)
        if blue_path:
            pairs.append((red_path, blue_path))
    
    return sorted(pairs, key=lambda x: x[0].name)


def _find_matching_blue(red_path: Path, red_channel: str, blue_channel: str) -> Union[Path, None]:
    """
    Find the matching blue channel file for a red channel file.
    
    Args:
        red_path: Path to red channel file
        red_channel: String identifier for red channel
        blue_channel: String identifier for blue channel
    
    Returns:
        Path to matching blue file or None if not found
    """
    stem = red_path.name
    
    # Extract base name
    if f"_{red_channel}." not in stem:
        return None
    
    base = stem.split(f"_{red_channel}.")[0]
    
    # Try different extensions
    for ext in ['tif', 'tiff', 'TIF', 'TIFF']:
        candidate = red_path.with_name(f"{base}_{blue_channel}.{ext}")
        if candidate.exists():
            return candidate
    
    return None


def load_as_gray(path: Path):
    """
    Load a TIFF file and convert to grayscale float32 array.
    
    Args:
        path: Path to TIFF file
    
    Returns:
        Grayscale image as float32 numpy array
    """
    import numpy as np
    import cv2
    
    arr = tifffile.imread(path)
    
    if arr.ndim == 3:
        # Handle channel-first format
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        
        # Convert RGB/RGBA to grayscale
        if arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[..., 0]
    
    return arr.astype(np.float32)


def build_rgb(red_gray, blue_gray):
    """
    Build RGB image from red and blue grayscale channels.
    
    Args:
        red_gray: Red channel as grayscale array
        blue_gray: Blue channel as grayscale array
    
    Returns:
        RGB image as uint8 numpy array
    """
    import numpy as np
    
    def to_u8(x):
        if x.dtype != np.uint8:
            lo, hi = np.nanmin(x), np.nanmax(x)
            if hi > lo:
                x = (x - lo) / (hi - lo) * 255.0
            else:
                x = np.zeros_like(x)
            return x.astype(np.uint8)
        return x
    
    r8 = to_u8(red_gray)
    b8 = to_u8(blue_gray)
    zeros = np.zeros_like(r8)
    
    return np.stack([r8, zeros, b8], axis=-1)


def remove_rectangles(image, white_thresh=240, aspect_low=0.2, aspect_high=5.0,
                     dilation_kernel=(15, 15), inpaint_radius=15):
    """
    Remove rectangular artifacts from image.
    
    Args:
        image: Input image array
        white_thresh: Threshold for detecting white regions
        aspect_low: Minimum aspect ratio to consider
        aspect_high: Maximum aspect ratio to consider
        dilation_kernel: Kernel size for dilation
        inpaint_radius: Radius for inpainting
    
    Returns:
        Image with rectangles removed
    """
    import cv2
    import numpy as np
    
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
