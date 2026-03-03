"""
GIF creation and frame handling utilities.
"""

import numpy as np
from PIL import Image
from typing import List, Union, Optional
import os


def frames_to_gif(
    frames: List[np.ndarray],
    output_path: str,
    duration_seconds: float = 3.0,
    fps: Optional[int] = None,
    loop: int = 0,
    optimize: bool = True
) -> None:
    """
    Convert a sequence of frames to an animated GIF.
    
    Parameters:
    -----------
    frames : List[np.ndarray]
        List of image frames (2D grayscale or 3D RGB)
    output_path : str
        Path for output GIF file
    duration_seconds : float, default=3.0
        Total duration of animation in seconds
    fps : Optional[int], default=None
        Target frames per second. If None, calculated from duration_seconds
    loop : int, default=0
        Number of loops (0 = infinite loop)
    optimize : bool, default=True
        Whether to optimize GIF file size
    
    Returns:
    --------
    None
        Saves GIF to output_path
    
    Examples:
    ---------
    >>> frames = [frame1, frame2, frame3]
    >>> frames_to_gif(frames, "animation.gif", duration_seconds=2.0, fps=15)
    """
    if not frames:
        raise ValueError("frames list cannot be empty")
    
    # Calculate frame duration
    if fps is not None:
        frame_duration_ms = int(1000 / fps)
    else:
        frame_duration_ms = int((duration_seconds * 1000) / len(frames))
    
    # Ensure minimum frame duration (some viewers have issues with very fast frames)
    frame_duration_ms = max(frame_duration_ms, 20)  # Minimum 20ms (50 FPS max)
    
    print(f"Creating GIF with {len(frames)} frames")
    print(f"Frame duration: {frame_duration_ms}ms ({1000/frame_duration_ms:.1f} FPS)")
    print(f"Total duration: {len(frames) * frame_duration_ms / 1000:.2f}s")
    
    # Convert frames to PIL Images
    pil_images = []
    
    for i, frame in enumerate(frames):
        # Ensure frame is numpy array
        frame_array = np.asarray(frame)
        
        # Handle different frame types
        if frame_array.ndim == 2:
            # Grayscale: convert to RGB
            if frame_array.dtype == bool:
                # Binary image
                frame_uint8 = (frame_array.astype(np.uint8) * 255)
            else:
                # Grayscale image - normalize to [0, 255]
                frame_normalized = _normalize_to_uint8(frame_array)
                frame_uint8 = frame_normalized
            
            # Convert to RGB
            frame_rgb = np.stack([frame_uint8] * 3, axis=-1)
            
        elif frame_array.ndim == 3 and frame_array.shape[2] == 3:
            # Already RGB
            if frame_array.dtype != np.uint8:
                frame_rgb = _normalize_to_uint8(frame_array)
            else:
                frame_rgb = frame_array
                
        elif frame_array.ndim == 3 and frame_array.shape[2] == 1:
            # Single channel in 3D array
            frame_2d = frame_array[:, :, 0]
            frame_uint8 = _normalize_to_uint8(frame_2d)
            frame_rgb = np.stack([frame_uint8] * 3, axis=-1)
            
        else:
            raise ValueError(f"Unsupported frame shape: {frame_array.shape}")
        
        # Convert to PIL Image
        pil_img = Image.fromarray(frame_rgb.astype(np.uint8), 'RGB')
        pil_images.append(pil_img)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as animated GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=frame_duration_ms,
        loop=loop,
        optimize=optimize
    )
    
    print(f"GIF saved: {output_path}")


def save_frame_sequence(
    frames: List[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
    format: str = "png"
) -> List[str]:
    """
    Save individual frames as separate image files.
    
    Parameters:
    -----------
    frames : List[np.ndarray]
        List of image frames
    output_dir : str
        Directory to save frames
    prefix : str, default="frame"
        Prefix for frame filenames
    format : str, default="png"
        Image format (png, jpg, etc.)
        
    Returns:
    --------
    List[str]
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, frame in enumerate(frames):
        filename = f"{prefix}_{i:03d}.{format}"
        filepath = os.path.join(output_dir, filename)
        
        # Convert frame to appropriate format
        frame_array = np.asarray(frame)
        
        if frame_array.ndim == 2:
            if frame_array.dtype == bool:
                save_array = (frame_array.astype(np.uint8) * 255)
            else:
                save_array = _normalize_to_uint8(frame_array)
            
            # Save as grayscale
            Image.fromarray(save_array, 'L').save(filepath)
            
        elif frame_array.ndim == 3:
            if frame_array.dtype != np.uint8:
                save_array = _normalize_to_uint8(frame_array)
            else:
                save_array = frame_array
            
            Image.fromarray(save_array.astype(np.uint8), 'RGB').save(filepath)
        
        saved_paths.append(filepath)
    
    print(f"Saved {len(saved_paths)} frames to {output_dir}")
    return saved_paths


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 255] uint8 range.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array
        
    Returns:
    --------
    np.ndarray
        Normalized uint8 array
    """
    array = np.asarray(array, dtype=np.float64)
    
    # Handle empty or invalid arrays
    if array.size == 0:
        return array.astype(np.uint8)
    
    # Get finite values for normalization
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.uint8)
    
    finite_values = array[finite_mask]
    min_val = np.min(finite_values)
    max_val = np.max(finite_values)
    
    # Handle constant images
    if max_val == min_val:
        if min_val == 0:
            return np.zeros_like(array, dtype=np.uint8)
        else:
            return np.full_like(array, 255, dtype=np.uint8)
    
    # Normalize to [0, 255]
    normalized = (array - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    uint8_array = (normalized * 255).astype(np.uint8)
    
    return uint8_array