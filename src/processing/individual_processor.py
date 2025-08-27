"""
Individual processor for handling single image pairs.
"""

import numpy as np
from pathlib import Path
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

from processing.extractor_pipeline import ExtractorPipeline
from processing.scoring_selector import ScoringSelector
from utils.file_utils import load_as_gray, build_rgb, remove_rectangles
from utils.image_utils import calculate_pixel_dimensions


class IndividualProcessor:
    """Processor for individual image pairs."""
    
    def __init__(self, pixel_dims_dict, debug=False, scoring_weights=None, max_regions=5):
        """
        Initialize the individual processor.
        
        Args:
            pixel_dims_dict: Dictionary mapping filename patterns to pixel dimensions
            debug: Whether to enable debug mode
            scoring_weights: Dictionary of scoring weights for region selection
            max_regions: Maximum number of regions to select per image
        """
        self.pixel_dims_dict = pixel_dims_dict
        self.debug = debug
        self.scoring_weights = scoring_weights
        self.max_regions = max_regions
        self.debug_info = {}
    
    def process_pair(self, red_path, blue_path):
        """
        Process a single red/blue image pair.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Tuple of (selected_labels, label_summary, debug_info)
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
            
            # Get intensity metrics before they are lost
            background_tissue_intensity, average_crypt_intensity = extractor.get_intensity_metrics()
            
            # Remove regions that touch the image edges
            initial_labels = self._remove_edge_touching_regions(initial_labels)
            
            if self.debug:
                self.debug_info['extractor_debug'] = extractor.get_debug_info()
                self.debug_info['initial_labels'] = initial_labels.copy()
                unique_labels = np.unique(initial_labels)
                print(f"[DEBUG] Initial extraction found {len(unique_labels)-1} regions after edge removal (labels: {unique_labels})")
            
            # Apply NEW SCORING STRATEGY - Quality-based region selection
            print(f"[INFO] Applying new quality-based scoring strategy...")
            
            # Use scoring weights passed from configuration, or defaults
            scoring_weights = self.scoring_weights if self.scoring_weights is not None else {
                'circularity': 0.35,    # Most important - want circular regions
                'area': 0.25,           # Second - want consistent sizes
                'line_fit': 0.15,       # Moderate - want aligned regions
                'red_intensity': 0.15,  # Moderate - want bright regions
                'com_consistency': 0.10 # Least - center consistency
            }
            
            selector = ScoringSelector(initial_labels, red_img, debug=self.debug, max_regions=self.max_regions, weights=scoring_weights)
            selected_labels = selector.select()
            
            if self.debug:
                selection_debug = selector.get_debug_info()
                self.debug_info['selected_labels'] = selected_labels.copy()
                self.debug_info['selection_properties'] = selection_debug.get('properties_df', {})
                self.debug_info['selected_label_ids'] = selection_debug.get('selected_labels', [])
                print(f"[DEBUG] Selection: {selection_debug['original_regions']} -> {selection_debug['selected_regions']} regions")
                print(f"[DEBUG] Selected label IDs: {selection_debug['selected_labels']}")
                if hasattr(selector, 'properties_df') and selector.properties_df is not None:
                    print(f"[DEBUG] Quality score range: {selector.properties_df['quality_score'].min():.3f} - {selector.properties_df['quality_score'].max():.3f}")
                
                # Save scoring plot if we have multiple regions
                if len(np.unique(initial_labels)) > 3:  # More than just background + 1-2 regions
                    try:
                        scoring_plot_path = self.debug_info.get('debug_dir', Path(".")) / f"{red_path.stem}_scoring_breakdown.png"
                        selector.plot_scoring_results(save_path=scoring_plot_path)
                        print(f"[DEBUG] Saved scoring breakdown plot to {scoring_plot_path}")
                    except Exception as e:
                        print(f"[DEBUG] Could not save scoring plot: {e}")
            else:
                print(f"[INFO] Selection complete: {len(np.unique(initial_labels))-1} -> {len(np.unique(selected_labels))-1} regions")
            
            # Calculate pixel dimensions
            pixel_dim = calculate_pixel_dimensions(red_path.name, self.pixel_dims_dict)
            
            if self.debug:
                print(f"[DEBUG] Using pixel dimension: {pixel_dim} μm/pixel for {red_path.name}")
            
            # Generate label summary
            label_summary = self._generate_label_summary(
                selected_labels, red_img, pixel_dim, background_tissue_intensity, average_crypt_intensity
            )
            
            if self.debug:
                if label_summary is not None and len(label_summary) > 0:
                    print(f"[DEBUG] Generated summary for {len(label_summary)} regions")
                    print(f"[DEBUG] Area range: [{label_summary[:, 3].min():.2f}, {label_summary[:, 3].max():.2f}] μm²")
                    print(f"[DEBUG] Intensity range: [{label_summary[:, 5].min():.2f}, {label_summary[:, 5].max():.2f}]")
                else:
                    print(f"[DEBUG] No regions found in summary")
            
            # Generate visualizations
            # Always generate standard visualization (regardless of debug mode)
            standard_visual = self._generate_standard_visualization(rgb_img, selected_labels, red_path.stem)
            
            # Save selected labels for inspection tool
            self._save_labels_array(selected_labels, red_path.stem)
            
            # Generate additional debug visualizations if requested
            debug_visuals = {}
            if self.debug:
                debug_visuals = self._generate_debug_visuals(
                    rgb_img, initial_labels, selected_labels, red_path.stem
                )
            
            return selected_labels, label_summary, standard_visual, debug_visuals
            
        except Exception as e:
            print(f"Error processing {red_path.name}/{blue_path.name}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, None, {}, {}
    
    def _save_labels_array(self, labels, image_stem):
        """
        Save the selected labels array for later inspection.
        
        Args:
            labels: Selected labels array
            image_stem: Image stem for filename
        """
        try:
            # Create npy directory if it doesn't exist
            npy_dir = Path("results") / "npy"
            npy_dir.mkdir(parents=True, exist_ok=True)
            
            # Save labels array
            labels_path = npy_dir / f"{image_stem}_selected_labels.npy"
            np.save(labels_path, labels)
            
            if self.debug:
                print(f"[DEBUG] Saved labels array to {labels_path}")
                
        except Exception as e:
            print(f"Warning: Could not save labels array: {e}")

    def _generate_label_summary(self, labels, red_img, pixel_dim, background_tissue_intensity=0.0, average_crypt_intensity=0.0):
        """
        Generate summary statistics for labeled regions - VECTORIZED VERSION.
        
        Args:
            labels: Labeled image array
            red_img: Red channel image for intensity measurements
            pixel_dim: Pixel dimension in micrometers
            background_tissue_intensity: Background tissue intensity (red/blue ratio)
            average_crypt_intensity: Average crypt intensity (red/blue ratio)
        
        Returns:
            Numpy array with columns: [id, pos_x, pos_y, area_um2, red_sum, red_intensity, 
                                     fluorescence, background_tissue_intensity, average_crypt_intensity]
        """
        props = regionprops(labels, intensity_image=red_img)
        
        if self.debug:
            unique_labels = np.unique(labels)
            print(f"[SUMMARY DEBUG] Input labels: {len(unique_labels)} unique ({unique_labels})")
            print(f"[SUMMARY DEBUG] Region props found: {len(props)} regions")
            print(f"[SUMMARY DEBUG] Red image for intensity: range [{red_img.min():.2f}, {red_img.max():.2f}]")
        
        if not props:
            if self.debug:
                print(f"[SUMMARY DEBUG] No regions to summarize, returning empty array")
            return np.empty((0, 9))  # Updated to 9 columns
        
        # Vectorized calculation of properties
        summary_data = []
        for i, prop in enumerate(props):
            if prop.label == 0:  # Skip background
                if self.debug:
                    print(f"[SUMMARY DEBUG] Skipping background region (label 0)")
                continue
            
            # Calculate statistics using regionprops data directly
            region_id = prop.label
            pos_y, pos_x = prop.centroid  # Note: centroid returns (row, col)
            area_pixels = prop.area
            area_um2 = area_pixels * (pixel_dim ** 2)
            
            # Use regionprops intensity data - more efficient than manual masking
            red_sum = prop.mean_intensity * area_pixels  # Total intensity
            red_intensity = prop.mean_intensity  # Mean intensity
            
            # Calculate fluorescence as red_sum_pixels / area_pixels * area_um2
            # This is equivalent to: red_intensity * area_um2
            fluorescence = red_intensity * area_um2
            
            if self.debug and i < 5:  # Only show debug for first 5 regions
                print(f"[SUMMARY DEBUG] Region {region_id}: area={area_pixels}px ({area_um2:.2f}μm²), "
                      f"centroid=({pos_x:.1f},{pos_y:.1f}), red_sum={red_sum:.0f}, intensity={red_intensity:.2f}, "
                      f"fluorescence={fluorescence:.2f}")
            
            summary_data.append([
                region_id, pos_x * pixel_dim, pos_y * pixel_dim, 
                area_um2, red_sum, red_intensity, fluorescence,
                background_tissue_intensity, average_crypt_intensity
            ])
        
        # Convert to numpy array
        if summary_data:
            result = np.array(summary_data)
            if self.debug:
                print(f"[SUMMARY DEBUG] Final summary shape: {result.shape}")
                print(f"[SUMMARY DEBUG] Background tissue intensity: {background_tissue_intensity:.3f}")
                print(f"[SUMMARY DEBUG] Average crypt intensity: {average_crypt_intensity:.3f}")
            return result
        else:
            if self.debug:
                print(f"[SUMMARY DEBUG] No regions to summarize, returning empty array")
            # Return empty array with correct shape
            return np.empty((0, 9))  # Updated to 9 columns
    
    def _generate_standard_visualization(self, rgb_img, labels, base_name):
        """
        Generate standard visualization with detected regions overlaid.
        This is always generated regardless of debug mode.
        
        Args:
            rgb_img: Combined RGB image
            labels: Labeled regions array
            base_name: Base name for the image pair
        
        Returns:
            Dictionary with standard visualization
        """
        from skimage.segmentation import find_boundaries
        import numpy as np
        
        # Create overlay with detected region boundaries
        boundaries = find_boundaries(labels, mode='inner')
        overlay = rgb_img.copy()
        overlay[boundaries] = [255, 0, 0]  # Red boundaries
        
        return {
            'standard_overlay': overlay,
            'num_regions': len(np.unique(labels)) - 1,  # Subtract background
            'base_name': base_name
        }
    
    def _generate_debug_visuals(self, rgb_img, initial_labels, selected_labels, base_name):
        """
        Generate debug visualization images.
        
        Args:
            rgb_img: Combined RGB image
            initial_labels: Initial labeled regions
            selected_labels: Final selected labeled regions
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
        
        # Selected labels overlay
        selected_boundaries = find_boundaries(selected_labels, mode='inner')
        selected_overlay = rgb_img.copy()
        selected_overlay[selected_boundaries] = [255, 0, 0]  # Red boundaries
        debug_visuals['selected_overlay'] = selected_overlay
        
        # Combined view with both overlays
        combined_overlay = rgb_img.copy()
        combined_overlay[initial_boundaries] = [255, 255, 0]  # Yellow for initial
        combined_overlay[selected_boundaries] = [255, 0, 0]     # Red for selected (overwrites)
        debug_visuals['combined_overlay'] = combined_overlay
        
        return debug_visuals
    
    def run(self, red_path, blue_path):
        """
        Default run method that returns final selected labels.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Final selected labels array
        """
        selected_labels, _, _, _ = self.process_pair(red_path, blue_path)
        return selected_labels
    
    def debug_run(self, red_path, blue_path):
        """
        Debug run method that returns comprehensive debug information.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
        
        Returns:
            Dictionary containing debug renderings and information
        """
        selected_labels, label_summary, standard_visual, debug_visuals = self.process_pair(red_path, blue_path)
        
        debug_output = {
            'selected_labels': selected_labels,
            'label_summary': label_summary,
            'standard_visual': standard_visual,
            'visuals': debug_visuals,
            'processing_info': self.debug_info.copy()
        }
        
        return debug_output
    
    def _remove_edge_touching_regions(self, labels):
        """
        Remove any labeled regions that touch the edges of the image - VECTORIZED VERSION.
        
        Args:
            labels: Labeled image array
            
        Returns:
            Filtered label array with edge-touching regions removed
        """
        height, width = labels.shape
        
        # Get unique labels excluding background
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        
        # Vectorized edge detection - get all edge pixels at once
        edge_mask = np.zeros_like(labels, dtype=bool)
        edge_mask[0, :] = True    # Top edge
        edge_mask[-1, :] = True   # Bottom edge
        edge_mask[:, 0] = True    # Left edge
        edge_mask[:, -1] = True   # Right edge
        
        # Find all labels that touch edges in one operation
        edge_labels = np.unique(labels[edge_mask])
        edge_labels = edge_labels[edge_labels != 0]  # Remove background
        
        if self.debug:
            print(f"[DEBUG] Found {len(edge_labels)} edge-touching regions to remove: {sorted(edge_labels)}")
        
        # Vectorized removal - create mask for non-edge labels
        keep_mask = ~np.isin(labels, edge_labels)
        filtered_labels = labels * keep_mask  # Removes edge labels, sets them to 0
        
        # Vectorized relabeling - get remaining labels and create lookup table
        remaining_labels = np.unique(filtered_labels)
        remaining_labels = remaining_labels[remaining_labels != 0]
        
        if len(remaining_labels) > 0:
            # Create vectorized lookup table for relabeling
            max_label = remaining_labels.max()
            lookup = np.zeros(max_label + 1, dtype=labels.dtype)
            
            for new_id, old_id in enumerate(sorted(remaining_labels), start=1):
                lookup[old_id] = new_id
            
            # Apply vectorized relabeling
            final_labels = np.where(filtered_labels > 0, lookup[filtered_labels], 0)
        else:
            final_labels = filtered_labels
        
        if self.debug:
            original_count = len(unique_labels)
            final_count = len(np.unique(final_labels)) - 1
            print(f"[DEBUG] Edge removal: {original_count} -> {final_count} regions")
        
        return final_labels
