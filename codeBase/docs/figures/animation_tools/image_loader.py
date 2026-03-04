"""
Image loading utilities for pipeline data and test images.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import tifffile
from skimage import draw
import sys

# Add codeBase to path for pipeline imports
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from pipeline_implementations.dask_implementation.pipeline import find_tif_images_by_keys
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: Pipeline imports not available")
    PIPELINE_AVAILABLE = False


def load_pipeline_images(max_subjects: int = 1) -> Optional[Tuple[np.ndarray, str]]:
    """
    Load RFP images from the pipeline data directory.
    
    Parameters:
    -----------
    max_subjects : int, default=1
        Maximum number of subjects to search
        
    Returns:
    --------
    Optional[Tuple[np.ndarray, str]]
        Tuple of (image_array, image_path) or None if not found
    """
    if not PIPELINE_AVAILABLE:
        print("Error: Pipeline functions not available")
        return None
    
    image_base_dir = Path("/home/phillip/documents/experimental_data/inputs/karen/lysozyme")
    
    if not image_base_dir.exists():
        print(f"Error: Image directory not found: {image_base_dir}")
        return None
    
    print(f"Searching for RFP images in: {image_base_dir}")
    
    try:
        # Use pipeline function to find images
        unmatched, pairs, paired_names, unmatched_names = find_tif_images_by_keys(
            image_base_dir,
            keys=["_RFP", "_DAPI"],
            max_subjects=max_subjects
        )
        
        # Try paired images first
        if pairs:
            rfp_path = pairs[0][0]  # First RFP image from pair
            print(f"Found paired RFP image: {rfp_path}")
            image = _load_tiff_image(rfp_path)
            return image, str(rfp_path)
        
        # Try unmatched images for RFP-like files
        if unmatched:
            for path in unmatched:
                if 'rfp' in path.name.lower():
                    print(f"Found unmatched RFP image: {path}")
                    image = _load_tiff_image(path)
                    return image, str(path)
        
        # Manual search fallback
        rfp_files = list(image_base_dir.rglob("*RFP*.tif")) + list(image_base_dir.rglob("*rfp*.tif"))
        if rfp_files:
            rfp_path = rfp_files[0]
            print(f"Found RFP image (manual search): {rfp_path}")
            image = _load_tiff_image(rfp_path)
            return image, str(rfp_path)
        
        print("No RFP images found")
        return None
        
    except Exception as e:
        print(f"Error loading pipeline images: {e}")
        return None


def _load_tiff_image(path: Path) -> np.ndarray:
    """
    Load and process TIFF image.
    
    Parameters:
    -----------
    path : Path
        Path to TIFF file
        
    Returns:
    --------
    np.ndarray
        Processed image array
    """
    # Load TIFF
    image = tifffile.imread(path)
    
    # Handle multi-dimensional images
    if image.ndim > 2:
        # Extract first channel if multi-channel
        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            # Channels first (C, H, W)
            image = image[0]
        elif image.ndim == 3 and image.shape[-1] < image.shape[0]:
            # Channels last (H, W, C)
            image = image[:, :, 0]
        elif image.ndim == 4:
            # 4D image (e.g., time, channel, height, width)
            image = image[0, 0] if image.shape[0] < image.shape[-1] else image[0, :, :, 0]
        else:
            # Default: flatten to 2D
            image = np.squeeze(image)
    
    # Ensure 2D
    if image.ndim != 2:
        raise ValueError(f"Could not process image to 2D: shape {image.shape}")
    
    # Convert to float and normalize
    image = image.astype(np.float64)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    
    print(f"Loaded image: shape {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")
    
    return image


def create_test_star(
    image_size: int = 512,
    radius: int = 100,
    num_points: int = 5,
    inner_ratio: float = 0.4
) -> np.ndarray:
    """
    Create a binary star shape for testing.
    
    Parameters:
    -----------
    image_size : int, default=512
        Size of square image
    radius : int, default=100
        Outer radius of star
    num_points : int, default=5
        Number of star points
    inner_ratio : float, default=0.4
        Ratio of inner to outer radius
        
    Returns:
    --------
    np.ndarray
        Binary star image
    """
    center = image_size // 2
    star_image = np.zeros((image_size, image_size), dtype=bool)
    
    # Calculate star points
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    radii = np.array([radius, radius * inner_ratio] * num_points)
    
    # Convert to cartesian coordinates
    x_coords = center + radii * np.cos(angles)
    y_coords = center + radii * np.sin(angles)
    
    # Create polygon
    rr, cc = draw.polygon(y_coords, x_coords, star_image.shape)
    star_image[rr, cc] = True
    
    print(f"Created {num_points}-pointed star: {image_size}x{image_size}, radius {radius}")
    
    return star_image


def create_test_shapes() -> dict:
    """
    Create a collection of test shapes for animation testing.
    
    Returns:
    --------
    dict
        Dictionary of test shapes
    """
    shapes = {}
    
    # Star shape
    shapes['star'] = create_test_star(512, 100, 5)
    
    # Circle
    center = 256
    radius = 80
    rr, cc = draw.disk((center, center), radius, shape=(512, 512))
    circle = np.zeros((512, 512), dtype=bool)
    circle[rr, cc] = True
    shapes['circle'] = circle
    
    # Rectangle
    rectangle = np.zeros((512, 512), dtype=bool)
    rectangle[200:300, 150:350] = True
    shapes['rectangle'] = rectangle
    
    # Cross shape
    cross = np.zeros((512, 512), dtype=bool)
    cross[200:300, :] = True  # Horizontal bar
    cross[:, 200:300] = True  # Vertical bar
    shapes['cross'] = cross
    
    print(f"Created {len(shapes)} test shapes: {list(shapes.keys())}")
    
    return shapes
