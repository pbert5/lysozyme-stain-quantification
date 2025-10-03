"""
Utility functions for image processing and mathematical operations.
"""

import numpy as np


def minmax01(x, eps=1e-12):
    """
    Normalize array to 0-1 range.
    
    Args:
        x: Input array
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Normalized array in 0-1 range
    """
    x = x.astype(float, copy=False)
    lo = np.min(x)
    hi = np.max(x)
    return (x - lo) / max(hi - lo, eps)


def create_output_directories(base_output_dir, debug=False):
    """
    Create standard output directory structure.
    
    Args:
        base_output_dir: Base output directory path
        debug: Whether to create debug directories
    
    Returns:
        Dictionary of created directory paths
    """
    from pathlib import Path
    
    dirs = {
        'base': Path(base_output_dir),
        'results': Path(base_output_dir) / 'results',
        'summaries': Path(base_output_dir) / 'summaries',
        'visualizations': Path(base_output_dir) / 'visualizations'  # Always create this
    }
    
    if debug:
        dirs.update({
            'debug': Path(base_output_dir) / 'debug',
            'debug_individual': Path(base_output_dir) / 'debug' / 'individual',
            'debug_merged': Path(base_output_dir) / 'debug' / 'merged',
            'quick_check': Path(base_output_dir) / 'debug' / 'quick_check'
        })
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_debug_image(image, filepath, title=None):
    """
    Save debug image with optional title.
    
    Args:
        image: Image array to save
        filepath: Output file path
        title: Optional title for the image
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def save_standard_visualization(rgb_img, labels, filepath, title=None):
    """
    Save standard visualization with detected regions overlaid on RGB image.
    This is the main visualization for non-debug mode.
    
    Args:
        rgb_img: Original RGB image
        labels: Labeled regions array
        filepath: Output file path
        title: Optional title for the image
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    import numpy as np
    
    # Create overlay with detected region boundaries
    boundaries = find_boundaries(labels, mode='inner')
    overlay = rgb_img.copy()
    overlay[boundaries] = [255, 0, 0]  # Red boundaries
    
    plt.figure(figsize=(12, 8))
    plt.imshow(overlay)
    if title:
        plt.title(title, fontsize=14)
    plt.axis('off')
    
    # Add text annotation with region count
    num_regions = len(np.unique(labels)) - 1  # Subtract 1 for background
    plt.text(10, 30, f'{num_regions} regions detected', 
             color='white', fontsize=12, weight='bold',
             bbox=dict(facecolor='black', alpha=0.7, pad=5))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_pixel_dimensions(filename, pixel_dims_dict):
    """
    Determine pixel dimensions based on filename and lookup dictionary.
    
    Args:
        filename: Name of the image file
        pixel_dims_dict: Dictionary mapping patterns to pixel dimensions
    
    Returns:
        Pixel dimension in micrometers
    """
    for pattern, dims in pixel_dims_dict.items():
        if pattern != "default" and pattern in filename:
            return dims
    
    return pixel_dims_dict.get("default", 0.4476)
