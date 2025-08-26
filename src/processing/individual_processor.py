"""
Individual processor for handling single image pairs.
"""

import numpy as np
from pathlib import Path
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

from processing.extractor_pipeline import ExtractorPipeline
from processing.merge_pipeline import MergePipeline
from utils.file_utils import load_as_gray, build_rgb, remove_rectangles
from utils.image_utils import calculate_pixel_dimensions


class IndividualProcessor:
    """Processor for individual image pairs."""
    
    def __init__(self, pixel_dims_dict, debug=False):
        """
        Initialize the individual processor.
        
        Args:
            pixel_dims_dict: Dictionary mapping filename patterns to pixel dimensions
            debug: Whether to enable debug mode
        """
        self.pixel_dims_dict = pixel_dims_dict
        self.debug = debug
        self.debug_info = {}
    
    def process_pair(self, red_path, blue_path):
        """
        Process a single red/blue image pair.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Tuple of (merged_labels, label_summary, debug_info)
        """
        try:
            # Load images
            red_img = load_as_gray(red_path)
            blue_img = load_as_gray(blue_path)
            
            if red_img.shape != blue_img.shape:
                raise ValueError(f"Shape mismatch: red {red_img.shape} vs blue {blue_img.shape}")
            
            # Build RGB display image
            rgb_img = build_rgb(red_img, blue_img)
            
            # Try to remove rectangles using the proper function
            try:
                rgb_img = remove_rectangles(rgb_img)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Rectangle removal failed: {e}")
                pass  # Continue if rectangle removal fails
            
            # Store for debug
            if self.debug:
                self.debug_info['rgb_combined'] = rgb_img.copy()
                self.debug_info['red_channel'] = red_img.copy()
                self.debug_info['blue_channel'] = blue_img.copy()
                print(f"[DEBUG] Loaded images: red {red_img.shape} range [{red_img.min():.2f}, {red_img.max():.2f}]")
                print(f"[DEBUG] Loaded images: blue {blue_img.shape} range [{blue_img.min():.2f}, {blue_img.max():.2f}]")
            
            # Extract labels using extractor pipeline
            extractor = ExtractorPipeline(debug=self.debug)
            initial_labels = extractor.extract(red_img, blue_img)
            
            if self.debug:
                self.debug_info['extractor_debug'] = extractor.get_debug_info()
                self.debug_info['initial_labels'] = initial_labels.copy()
                unique_labels = np.unique(initial_labels)
                print(f"[DEBUG] Initial extraction found {len(unique_labels)-1} regions (labels: {unique_labels})")
            
            # SKIP MERGING FOR LYSOZYME DETECTION - The extractor works perfectly!
            # The merge pipeline is too aggressive and merges individual lysozyme stains
            print(f"[INFO] Using extractor results directly (no merging) - found {len(np.unique(initial_labels))-1} regions")
            merged_labels = initial_labels.copy()
            
            if self.debug:
                self.debug_info['merged_labels'] = merged_labels.copy()
                self.debug_info['merge_stage1'] = {}  # Empty since no merge
                self.debug_info['merge_stage2'] = {}  # Empty since no merge
                print(f"[DEBUG] Skipping merge - extractor found {len(np.unique(initial_labels))-1} lysozyme regions")
                print(f"[DEBUG] This prevents over-aggressive merging of individual stains")
            
            # Calculate pixel dimensions
            pixel_dim = calculate_pixel_dimensions(red_path.name, self.pixel_dims_dict)
            
            if self.debug:
                print(f"[DEBUG] Using pixel dimension: {pixel_dim} μm/pixel for {red_path.name}")
            
            # Generate label summary
            label_summary = self._generate_label_summary(
                merged_labels, red_img, pixel_dim
            )
            
            if self.debug:
                if label_summary is not None and len(label_summary) > 0:
                    print(f"[DEBUG] Generated summary for {len(label_summary)} regions")
                    print(f"[DEBUG] Area range: [{label_summary[:, 3].min():.2f}, {label_summary[:, 3].max():.2f}] μm²")
                    print(f"[DEBUG] Intensity range: [{label_summary[:, 5].min():.2f}, {label_summary[:, 5].max():.2f}]")
                else:
                    print(f"[DEBUG] No regions found in summary")
            
            # Generate debug visualizations if requested
            debug_visuals = {}
            if self.debug:
                debug_visuals = self._generate_debug_visuals(
                    rgb_img, initial_labels, merged_labels, red_path.stem
                )
            
            return merged_labels, label_summary, debug_visuals
            
        except Exception as e:
            print(f"Error processing {red_path.name}/{blue_path.name}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, None, {}
    
    def _generate_label_summary(self, labels, red_img, pixel_dim):
        """
        Generate summary statistics for labeled regions.
        
        Args:
            labels: Labeled image array
            red_img: Red channel image for intensity measurements
            pixel_dim: Pixel dimension in micrometers
        
        Returns:
            Numpy array with columns: [id, pos_x, pos_y, area, red_sum, red_intensity]
        """
        props = regionprops(labels, intensity_image=red_img)
        
        if self.debug:
            unique_labels = np.unique(labels)
            print(f"[SUMMARY DEBUG] Input labels: {len(unique_labels)} unique ({unique_labels})")
            print(f"[SUMMARY DEBUG] Region props found: {len(props)} regions")
            print(f"[SUMMARY DEBUG] Red image for intensity: range [{red_img.min():.2f}, {red_img.max():.2f}]")
        
        summary_data = []
        for i, prop in enumerate(props):
            if prop.label == 0:  # Skip background
                if self.debug:
                    print(f"[SUMMARY DEBUG] Skipping background region (label 0)")
                continue
            
            # Calculate statistics
            region_id = prop.label
            pos_y, pos_x = prop.centroid  # Note: centroid returns (row, col)
            area_pixels = prop.area
            area_um2 = area_pixels * (pixel_dim ** 2)
            
            # Get intensity values for this region
            mask = labels == prop.label
            red_values = red_img[mask]
            red_sum = red_values.sum()
            red_intensity = red_sum / area_pixels if area_pixels > 0 else 0
            
            if self.debug and i < 5:  # Only show debug for first 5 regions
                print(f"[SUMMARY DEBUG] Region {region_id}: area={area_pixels}px ({area_um2:.2f}μm²), "
                      f"centroid=({pos_x:.1f},{pos_y:.1f}), red_sum={red_sum:.0f}, intensity={red_intensity:.2f}")
            
            summary_data.append([
                region_id, pos_x * pixel_dim, pos_y * pixel_dim, 
                area_um2, red_sum, red_intensity
            ])
        
        # Convert to numpy array
        if summary_data:
            result = np.array(summary_data)
            if self.debug:
                print(f"[SUMMARY DEBUG] Final summary shape: {result.shape}")
            return result
        else:
            if self.debug:
                print(f"[SUMMARY DEBUG] No regions to summarize, returning empty array")
            # Return empty array with correct shape
            return np.empty((0, 6))
    
    def _generate_debug_visuals(self, rgb_img, initial_labels, merged_labels, base_name):
        """
        Generate debug visualization images.
        
        Args:
            rgb_img: Combined RGB image
            initial_labels: Initial labeled regions
            merged_labels: Final merged labeled regions
            base_name: Base name for the image pair
        
        Returns:
            Dictionary of debug images
        """
        debug_visuals = {}
        
        # Initial labels overlay
        initial_boundaries = find_boundaries(initial_labels, mode='inner')
        initial_overlay = rgb_img.copy()
        initial_overlay[initial_boundaries] = [255, 255, 0]  # Yellow boundaries
        debug_visuals['initial_overlay'] = initial_overlay
        
        # Merged labels overlay
        merged_boundaries = find_boundaries(merged_labels, mode='inner')
        merged_overlay = rgb_img.copy()
        merged_overlay[merged_boundaries] = [255, 0, 0]  # Red boundaries
        debug_visuals['merged_overlay'] = merged_overlay
        
        # Combined view with both overlays
        combined_overlay = rgb_img.copy()
        combined_overlay[initial_boundaries] = [255, 255, 0]  # Yellow for initial
        combined_overlay[merged_boundaries] = [255, 0, 0]     # Red for merged (overwrites)
        debug_visuals['combined_overlay'] = combined_overlay
        
        return debug_visuals
    
    def run(self, red_path, blue_path):
        """
        Default run method that returns final merged labels.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Final merged labels array
        """
        merged_labels, _, _ = self.process_pair(red_path, blue_path)
        return merged_labels
    
    def debug_run(self, red_path, blue_path):
        """
        Debug run method that returns comprehensive debug information.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Dictionary containing debug renderings and information
        """
        merged_labels, label_summary, debug_visuals = self.process_pair(red_path, blue_path)
        
        debug_output = {
            'merged_labels': merged_labels,
            'label_summary': label_summary,
            'visuals': debug_visuals,
            'processing_info': self.debug_info.copy()
        }
        
        return debug_output
