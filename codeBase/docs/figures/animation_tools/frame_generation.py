"""
Core frame generation functions for morphological operation animations.
"""

import numpy as np
from typing import Callable, List, Union, Any
from skimage import morphology


def morphological_frame_sequence(
    image: np.ndarray,
    operation: Callable,
    kernel_func: Callable,
    max_radius: int,
    frame_count: int,
    min_radius: int = 1,
    **operation_kwargs: Any
) -> List[np.ndarray]:
    """
    Generate a sequence of frames showing progressive morphological operations.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (binary or grayscale)
    operation : Callable
        Morphological operation function (e.g., morphology.dilation, morphology.erosion)
    kernel_func : Callable
        Function to create structuring element (e.g., morphology.disk, morphology.square)
    max_radius : int
        Maximum kernel radius
    frame_count : int
        Number of frames to generate
    min_radius : int, default=1
        Minimum kernel radius (prevents kernels smaller than 1px)
    **operation_kwargs : Any
        Additional keyword arguments for the operation
    
    Returns:
    --------
    List[np.ndarray]
        List of image frames
    
    Raises:
    -------
    ValueError
        If frame_count would result in kernels smaller than min_radius
    
    Examples:
    ---------
    >>> # Binary dilation
    >>> frames = morphological_frame_sequence(
    ...     binary_image, 
    ...     morphology.binary_dilation,
    ...     morphology.disk,
    ...     max_radius=20,
    ...     frame_count=10
    ... )
    
    >>> # Grayscale dilation
    >>> frames = morphological_frame_sequence(
    ...     grayscale_image,
    ...     morphology.dilation,
    ...     morphology.disk, 
    ...     max_radius=15,
    ...     frame_count=8
    ... )
    """
    # Input validation
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    
    if max_radius < min_radius:
        raise ValueError(f"max_radius ({max_radius}) must be >= min_radius ({min_radius})")
    
    # Calculate radius step size
    if frame_count == 1:
        radii = [max_radius]
    else:
        # Create sequence from min_radius to max_radius
        radii = np.linspace(min_radius, max_radius, frame_count)
        radii = np.round(radii).astype(int)
        radii = np.clip(radii, min_radius, max_radius)
    
    # Check if any radius would be too small
    if np.any(radii < min_radius):
        raise ValueError(f"Frame count {frame_count} would create kernels smaller than {min_radius}px. "
                        f"Reduce frame_count or increase max_radius.")
    
    # Check if the effective range is too small
    if max_radius < min_radius + frame_count - 1:
        raise ValueError(f"Not enough radius range for {frame_count} frames. "
                        f"Need at least {min_radius + frame_count - 1} max_radius for {frame_count} frames.")
    
    print(f"Generating {frame_count} frames with radii: {list(radii)}")
    
    # Generate frames
    frames = []
    current_image = image.copy()
    
    for i, radius in enumerate(radii):
        if i == 0:
            # First frame: original image
            frames.append(current_image.copy())
        else:
            # Apply operation with current kernel
            try:
                kernel = kernel_func(radius)
                current_image = operation(current_image, kernel, **operation_kwargs)
                frames.append(current_image.copy())
            except Exception as e:
                print(f"Warning: Failed to apply operation with radius {radius}: {e}")
                # Use previous frame as fallback
                frames.append(frames[-1].copy())
    
    return frames


def incremental_morphological_sequence(
    image: np.ndarray,
    operation: Callable,
    kernel_func: Callable,
    radius_sequence: List[int],
    **operation_kwargs: Any
) -> List[np.ndarray]:
    """
    Generate frames by applying operation incrementally with different kernel sizes.
    Each frame builds on the previous one.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    operation : Callable
        Morphological operation
    kernel_func : Callable
        Kernel generation function
    radius_sequence : List[int]
        Sequence of radii to apply
    **operation_kwargs : Any
        Additional operation arguments
        
    Returns:
    --------
    List[np.ndarray]
        List of progressively modified images
    """
    frames = [image.copy()]
    current_image = image.copy()
    
    for radius in radius_sequence:
        if radius >= 1:  # Safety check
            kernel = kernel_func(radius)
            current_image = operation(current_image, kernel, **operation_kwargs)
            frames.append(current_image.copy())
    
    return frames


def reset_morphological_sequence(
    image: np.ndarray,
    operation: Callable,
    kernel_func: Callable,
    radius_sequence: List[int],
    **operation_kwargs: Any
) -> List[np.ndarray]:
    """
    Generate frames by applying operation to original image with different kernel sizes.
    Each frame is independent, starting from the original.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    operation : Callable
        Morphological operation
    kernel_func : Callable
        Kernel generation function 
    radius_sequence : List[int]
        Sequence of radii to apply
    **operation_kwargs : Any
        Additional operation arguments
        
    Returns:
    --------
    List[np.ndarray]
        List of independently processed images
    """
    frames = [image.copy()]  # Start with original
    
    for radius in radius_sequence:
        if radius >= 1:  # Safety check
            kernel = kernel_func(radius)
            result = operation(image.copy(), kernel, **operation_kwargs)
            frames.append(result.copy())
    
    return frames