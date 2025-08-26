"""
Bulk processor for handling multiple image pairs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from processing.individual_processor import IndividualProcessor
from utils.image_utils import create_output_directories, save_debug_image
from utils.file_utils import build_rgb, load_as_gray


class BulkProcessor:
    """Processor for handling multiple image pairs in bulk."""
    
    def __init__(self, output_dir, pixel_dims, debug=False, max_workers=None, scoring_weights=None, max_regions=5):
        """
        Initialize the bulk processor.
        
        Args:
            output_dir: Output directory for results
            pixel_dims: Dictionary mapping filename patterns to pixel dimensions
            debug: Whether to enable debug mode
            max_workers: Maximum number of worker processes (None for auto)
            scoring_weights: Dictionary of scoring weights for region selection
            max_regions: Maximum number of regions to select per image
        """
        self.output_dir = Path(output_dir)
        self.pixel_dims = pixel_dims
        self.debug = debug
        self.max_workers = max_workers
        self.scoring_weights = scoring_weights
        self.max_regions = max_regions
        
        # Create output directories
        self.dirs = create_output_directories(self.output_dir, debug=debug)
        
        # Results storage
        self.results = []
        self.summaries = []
        self.failed_pairs = []
    
    def process_pairs(self, image_pairs):
        """
        Process multiple image pairs.
        
        Args:
            image_pairs: List of (red_path, blue_path) tuples
        
        Returns:
            List of processing results
        """
        print(f"Processing {len(image_pairs)} image pairs...")
        
        # For debugging, limit to first few pairs
        if self.debug and len(image_pairs) > 3:
            print(f"[DEBUG MODE] Limiting to first 3 pairs for detailed debugging")
            image_pairs = image_pairs[:3]
        
        # Process pairs (sequentially for now to avoid memory issues)
        for i, (red_path, blue_path) in enumerate(image_pairs):
            print(f"Processing pair {i+1}/{len(image_pairs)}: {red_path.stem}")
            
            if self.debug:
                print(f"[BULK DEBUG] Processing: {red_path.name} + {blue_path.name}")
            
            try:
                result = self._process_single_pair(red_path, blue_path, i)
                if result:
                    self.results.append(result)
                    
                    # Add to consolidated summary
                    if result['summary'] is not None and len(result['summary']) > 0:
                        summary_df = pd.DataFrame(
                            result['summary'], 
                            columns=['id', 'pos_x_um', 'pos_y_um', 'area_um2', 'red_sum', 'red_intensity']
                        )
                        summary_df['image_name'] = red_path.stem
                        summary_df['red_file'] = red_path.name
                        summary_df['blue_file'] = blue_path.name
                        self.summaries.append(summary_df)
                        
                        if self.debug:
                            print(f"[BULK DEBUG] Added {len(result['summary'])} regions to summary")
                    else:
                        if self.debug:
                            print(f"[BULK DEBUG] No regions found for {red_path.stem}")
                else:
                    print(f"[BULK DEBUG] Processing failed for {red_path.stem}")
                    
            except Exception as e:
                print(f"Failed to process {red_path.stem}: {e}")
                self.failed_pairs.append((red_path, blue_path, str(e)))
                if self.debug:
                    import traceback
                    traceback.print_exc()
        
        # Save consolidated results
        self._save_consolidated_results()
        
        # Generate quick check visualization
        if self.debug:
            self._generate_quick_check()
        
        print(f"Completed processing. {len(self.results)} successful, {len(self.failed_pairs)} failed.")
        return self.results
    
    def _process_single_pair(self, red_path, blue_path, index):
        """
        Process a single image pair.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
            index: Index of the pair in the batch
        
        Returns:
            Dictionary containing processing results
        """
        processor = IndividualProcessor(self.pixel_dims, debug=self.debug, scoring_weights=self.scoring_weights, max_regions=self.max_regions)
        
        if self.debug:
            # Get full debug information
            debug_result = processor.debug_run(red_path, blue_path)
            
            if debug_result['selected_labels'] is None:
                return None
            
            # Save debug visualizations
            self._save_debug_visuals(debug_result, red_path.stem, index)
            
            # Save standard visualization
            standard_visual = debug_result.get('standard_visual', {})
            if standard_visual:
                self._save_standard_visualization(standard_visual, red_path.stem)
            
            return {
                'red_path': red_path,
                'blue_path': blue_path,
                'selected_labels': debug_result['selected_labels'],
                'summary': debug_result['label_summary'],
                'debug_info': debug_result
            }
        else:
            # Get just the essential results + standard visualization
            selected_labels, summary, standard_visual, _ = processor.process_pair(red_path, blue_path)
            
            if selected_labels is None:
                return None
            
            # Save standard visualization
            if standard_visual:
                self._save_standard_visualization(standard_visual, red_path.stem)
            
            return {
                'red_path': red_path,
                'blue_path': blue_path,
                'selected_labels': selected_labels,
                'summary': summary,
                'debug_info': None
            }
    
    def _save_standard_visualization(self, standard_visual, base_name):
        """
        Save standard visualization to the visualizations directory.
        
        Args:
            standard_visual: Dictionary containing standard visualization data
            base_name: Base name for the image pair
        """
        from utils.image_utils import save_standard_visualization
        
        if 'standard_overlay' in standard_visual:
            overlay_img = standard_visual['standard_overlay']
            output_path = self.dirs['visualizations'] / f"{base_name}_detected_regions.png"
            
            title = f"{base_name} - {standard_visual.get('num_regions', 0)} regions detected"
            
            # Save using the regular save_debug_image since we already have the overlay
            from utils.image_utils import save_debug_image
            save_debug_image(overlay_img, output_path, title)
    
    def _save_debug_visuals(self, debug_result, base_name, index):
        """
        Save debug visualizations to files.
        
        Args:
            debug_result: Debug result dictionary
            base_name: Base name for the image pair
            index: Index of the pair in the batch
        """
        visuals = debug_result.get('visuals', {})
        processing_info = debug_result.get('processing_info', {})
        
        # Save individual debug images
        for visual_name, image in visuals.items():
            output_path = self.dirs['debug_individual'] / f"{base_name}_{visual_name}.png"
            save_debug_image(image, output_path, f"{base_name} - {visual_name}")
        
        # Save merged overlay specifically
        if 'merged_overlay' in visuals:
            output_path = self.dirs['debug_merged'] / f"{base_name}_merged.png"
            save_debug_image(visuals['merged_overlay'], output_path, f"{base_name} - Merged Labels")
        
        # Save intermediate processing steps if available
        if processing_info:
            # Create subdirectory for this image's detailed debug info
            detailed_dir = self.dirs['debug_individual'] / f"{base_name}_detailed"
            detailed_dir.mkdir(exist_ok=True)
            
            # Save original channels
            if 'red_channel' in processing_info:
                save_debug_image(processing_info['red_channel'], 
                               detailed_dir / f"{base_name}_01_red_channel.png",
                               f"{base_name} - Red Channel")
            
            if 'blue_channel' in processing_info:
                save_debug_image(processing_info['blue_channel'], 
                               detailed_dir / f"{base_name}_02_blue_channel.png",
                               f"{base_name} - Blue Channel")
            
            if 'rgb_combined' in processing_info:
                save_debug_image(processing_info['rgb_combined'], 
                               detailed_dir / f"{base_name}_03_rgb_combined.png",
                               f"{base_name} - RGB Combined")
            
            # Save extractor debug info
            extractor_debug = processing_info.get('extractor_debug', {})
            step_num = 4
            
            for step_name, step_data in extractor_debug.items():
                if isinstance(step_data, np.ndarray) and step_data.ndim in [2, 3]:
                    save_debug_image(step_data, 
                                   detailed_dir / f"{base_name}_{step_num:02d}_{step_name}.png",
                                   f"{base_name} - {step_name}")
                    step_num += 1
            
            # Save initial and merged labels
            if 'initial_labels' in processing_info:
                from skimage.color import label2rgb
                initial_img = label2rgb(processing_info['initial_labels'], bg_label=0)
                save_debug_image(initial_img, 
                               detailed_dir / f"{base_name}_{step_num:02d}_initial_labels.png",
                               f"{base_name} - Initial Labels")
                step_num += 1
            
            if 'selected_labels' in processing_info:
                from skimage.color import label2rgb
                selected_img = label2rgb(processing_info['selected_labels'], bg_label=0)
                save_debug_image(selected_img, 
                               detailed_dir / f"{base_name}_{step_num:02d}_final_selected_labels.png",
                               f"{base_name} - Final Selected Labels")
    
    def _save_consolidated_results(self):
        """Save consolidated results to CSV files."""
        if self.summaries:
            # Combine all summaries
            all_summaries = pd.concat(self.summaries, ignore_index=True)
            
            # Save to CSV
            summary_path = self.dirs['summaries'] / 'consolidated_summary.csv'
            all_summaries.to_csv(summary_path, index=False)
            print(f"Saved consolidated summary to {summary_path}")
            
            # Save by-image summary
            by_image = all_summaries.groupby('image_name').agg({
                'area_um2': ['count', 'sum', 'mean'],
                'red_sum': ['sum', 'mean'],
                'red_intensity': ['mean', 'std']
            }).round(4)
            
            by_image.columns = ['_'.join(col).strip() for col in by_image.columns]
            by_image_path = self.dirs['summaries'] / 'by_image_summary.csv'
            by_image.to_csv(by_image_path)
            print(f"Saved by-image summary to {by_image_path}")
        
        # Save failed pairs log
        if self.failed_pairs:
            failed_df = pd.DataFrame(self.failed_pairs, columns=['red_path', 'blue_path', 'error'])
            failed_path = self.dirs['summaries'] / 'failed_pairs.csv'
            failed_df.to_csv(failed_path, index=False)
            print(f"Saved failed pairs log to {failed_path}")
    
    def _generate_quick_check(self):
        """Generate quick check visualization with all overlays."""
        if not self.results:
            return
        
        # Create a grid of quick check images
        n_results = min(len(self.results), 16)  # Limit to 16 for visualization
        n_cols = 4
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_results):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            result = self.results[i]
            
            try:
                # Load original images
                red_img = load_as_gray(result['red_path'])
                blue_img = load_as_gray(result['blue_path'])
                rgb_img = build_rgb(red_img, blue_img)
                
                # Add selected labels overlay
                selected_boundaries = find_boundaries(result['selected_labels'], mode='inner')
                overlay = rgb_img.copy()
                overlay[selected_boundaries] = [255, 0, 0]  # Red boundaries
                
                ax.imshow(overlay)
                ax.set_title(result['red_path'].stem, fontsize=8)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(result['red_path'].stem, fontsize=8)
                ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_results, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        quick_check_path = self.dirs['quick_check'] / 'consolidated_quick_check.png'
        plt.savefig(quick_check_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved quick check visualization to {quick_check_path}")
