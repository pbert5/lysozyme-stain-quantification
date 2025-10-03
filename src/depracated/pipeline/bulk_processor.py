"""
Bulk processor for handling multiple image pairs.
"""
#TODO: this is sinfull, must implement parralel proc, am using less then 10% of my cpu at a time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import re
from collections import defaultdict, Counter

from ...lysozyme_stain_quantification.segment_crypts import segment_crypts_full
from ..debug import generate_label_summary
from ..utils.image_utils import (
    calculate_pixel_dimensions,
    create_output_directories,
    save_debug_image,
)
from ..utils.file_utils import load_as_gray


class BulkProcessor:
    """Processor for handling multiple image pairs in bulk."""
    
    def __init__(
        self,
        output_dir,
        pixel_dims,
        debug=False,
        max_workers=None,
        scoring_weights=None,
        max_regions=5,
        blob_size_px=15,
    ):
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
        self.blob_size_px = blob_size_px
        
        # Create output directories
        self.dirs = create_output_directories(self.output_dir, debug=debug)
        
        # Results storage
        self.results = []
        self.summaries = []
        self.failed_pairs = []
        
        # Quality tracking
        self.quality_issues = {
            'insufficient_regions': [],  # Images with fewer than max_regions
            'duplicate_names': defaultdict(list),  # Names that appear multiple times
            'retake_images': [],  # Images from retake directories
            'red_outliers': [],  # Regions with unusual red intensity values
            'processing_warnings': []  # General processing warnings
        }
    
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
            # Generate clean name and track quality info
            clean_name, is_retake, subdir_info = self._generate_clean_name(red_path, blue_path)
            
            print(f"Processing pair {i+1}/{len(image_pairs)}: {clean_name}")
            if is_retake:
                print(f"  [RETAKE] from {subdir_info}")
            
            if self.debug:
                print(f"[BULK DEBUG] Processing: {red_path.name} + {blue_path.name}")
                print(f"[BULK DEBUG] Clean name: {clean_name} (subdir: {subdir_info})")
            
            try:
                result = self._process_single_pair(red_path, blue_path, i)
                if result:
                    self.results.append(result)
                    
                    # Check region quality
                    self._check_region_quality(result['summary'], clean_name, self.max_regions)
                    
                    # Add to consolidated summary
                    if result['summary'] is not None and len(result['summary']) > 0:
                        summary_df = pd.DataFrame(
                            result['summary'], 
                            columns=['id', 'pos_x_um', 'pos_y_um', 'area_um2', 'red_sum', 'red_intensity', 
                                   'fluorescence', 'background_tissue_intensity', 'average_crypt_intensity']
                        )
                        # Use clean name instead of raw stem
                        summary_df['image_name'] = clean_name
                        summary_df['is_retake'] = is_retake
                        summary_df['subdir'] = subdir_info
                        summary_df['red_file'] = red_path.name
                        summary_df['blue_file'] = blue_path.name
                        summary_df['red_path'] = str(red_path)  # Full path for reference
                        self.summaries.append(summary_df)
                        
                        if self.debug:
                            print(f"[BULK DEBUG] Added {len(result['summary'])} regions to summary")
                    else:
                        if self.debug:
                            print(f"[BULK DEBUG] No regions found for {clean_name}")
                else:
                    print(f"[BULK DEBUG] Processing failed for {clean_name}")
                    
            except Exception as e:
                print(f"Failed to process {clean_name}: {e}")
                self.failed_pairs.append((red_path, blue_path, str(e)))
                if self.debug:
                    import traceback
                    traceback.print_exc()
        
        # Save consolidated results
        self._save_consolidated_results()
        
        # Generate quick check visualization
        if self.debug:
            self._generate_quick_check()
        
        # Generate quality report
        self._generate_quality_report()
        
        print(f"Completed processing. {len(self.results)} successful, {len(self.failed_pairs)} failed.")
        
        return self.results
    
    def _generate_clean_name(self, red_path, blue_path):
        """
        Generate a clean image name, removing channel suffixes and handling duplicates.
        
        Args:
            red_path: Path to red channel image
            blue_path: Path to blue channel image
            
        Returns:
            Tuple of (clean_name, is_retake, subdir_info)
        """
        # Remove channel suffix (_RFP, _DAPI) from filename
        red_name = red_path.stem
        clean_name = re.sub(r'_(?:RFP|DAPI)$', '', red_name, flags=re.IGNORECASE)
        
        # Check if this is from a retake directory or Jej LYZ directory
        parent_path_str = str(red_path.parent).lower()
        is_retake = 'retake' in parent_path_str or 'jej lyz' in parent_path_str
        
        # Get subdirectory info for duplicate handling
        # Use relative path from the common image directory
        try:
            # Try to find a common root directory
            image_dir_parts = Path(red_path).parts
            # Find the part that contains "lysozyme" or use the parent of the file
            subdir_parts = red_path.parent.parts[-2:]  # Last 2 directory levels
            subdir_info = '/'.join(subdir_parts) if subdir_parts else 'root'
        except:
            # Fallback to just the immediate parent directory
            subdir_info = red_path.parent.name
        
        # Track duplicate names using clean_name as key
        if clean_name not in self.quality_issues['duplicate_names']:
            self.quality_issues['duplicate_names'][clean_name] = []
        
        self.quality_issues['duplicate_names'][clean_name].append({
            'path': red_path,
            'subdir': subdir_info,
            'is_retake': is_retake
        })
        
        # Track retakes
        if is_retake:
            self.quality_issues['retake_images'].append({
                'clean_name': clean_name,
                'path': red_path,
                'subdir': subdir_info
            })
        
        return clean_name, is_retake, subdir_info
    
    def _check_region_quality(self, summary_data, clean_name, max_regions):
        """
        Check for quality issues in the processed regions.
        
        Args:
            summary_data: Region summary array
            clean_name: Clean image name
            max_regions: Expected maximum regions
        """
        if summary_data is None or len(summary_data) == 0:
            self.quality_issues['insufficient_regions'].append({
                'name': clean_name,
                'found_regions': 0,
                'expected_regions': max_regions,
                'issue': 'no_regions_detected'
            })
            return
        
        n_regions = len(summary_data)
        
        # Check for insufficient regions
        if n_regions < max_regions:
            self.quality_issues['insufficient_regions'].append({
                'name': clean_name,
                'found_regions': n_regions,
                'expected_regions': max_regions,
                'issue': 'fewer_than_expected'
            })
        
        # Check for red intensity outliers
        if len(summary_data) > 0:
            red_intensities = summary_data[:, 5]  # red_intensity column
            
            # Calculate outlier thresholds (using IQR method)
            q1 = np.percentile(red_intensities, 25)
            q3 = np.percentile(red_intensities, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = np.where((red_intensities < lower_bound) | (red_intensities > upper_bound))[0]
            
            if len(outlier_indices) > 0:
                for idx in outlier_indices:
                    self.quality_issues['red_outliers'].append({
                        'name': clean_name,
                        'region_id': int(summary_data[idx, 0]),
                        'red_intensity': float(red_intensities[idx]),
                        'outlier_type': 'low' if red_intensities[idx] < lower_bound else 'high',
                        'q1': q1,
                        'q3': q3,
                        'threshold': lower_bound if red_intensities[idx] < lower_bound else upper_bound
                    })
    
    def _generate_quality_report(self):
        """Generate and save a comprehensive quality report."""
        report_lines = []
        report_lines.append("LYSOZYME STAIN QUANTIFICATION - QUALITY REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Processing summary
        report_lines.append(f"PROCESSING SUMMARY:")
        report_lines.append(f"  Successfully processed: {len(self.results)} images")
        report_lines.append(f"  Failed processing: {len(self.failed_pairs)} images")
        report_lines.append("")
        
        # Duplicate names
        actual_duplicates = {name: entries for name, entries in self.quality_issues['duplicate_names'].items() if len(entries) > 1}
        if actual_duplicates:
            report_lines.append(f"DUPLICATE IMAGE NAMES ({len(actual_duplicates)} cases):")
            for name, entries in actual_duplicates.items():
                report_lines.append(f"  '{name}' appears in {len(entries)} locations:")
                for entry in entries:
                    retake_note = " [RETAKE]" if entry['is_retake'] else ""
                    report_lines.append(f"    - {entry['subdir']}{retake_note}")
            report_lines.append("")
        
        # Retake images
        if self.quality_issues['retake_images']:
            report_lines.append(f"RETAKE IMAGES ({len(self.quality_issues['retake_images'])} found):")
            for retake in self.quality_issues['retake_images']:
                report_lines.append(f"  {retake['clean_name']} (from {retake['subdir']})")
            report_lines.append("")
        
        # Insufficient regions
        if self.quality_issues['insufficient_regions']:
            report_lines.append(f"INSUFFICIENT REGIONS ({len(self.quality_issues['insufficient_regions'])} images):")
            for issue in self.quality_issues['insufficient_regions']:
                report_lines.append(f"  {issue['name']}: {issue['found_regions']}/{issue['expected_regions']} regions ({issue['issue']})")
            report_lines.append("")
        
        # Red intensity outliers
        if self.quality_issues['red_outliers']:
            report_lines.append(f"RED INTENSITY OUTLIERS ({len(self.quality_issues['red_outliers'])} regions):")
            outlier_by_image = defaultdict(list)
            for outlier in self.quality_issues['red_outliers']:
                outlier_by_image[outlier['name']].append(outlier)
            
            for image_name, outliers in outlier_by_image.items():
                report_lines.append(f"  {image_name}:")
                for outlier in outliers:
                    report_lines.append(f"    Region {outlier['region_id']}: {outlier['red_intensity']:.2f} ({outlier['outlier_type']} outlier, threshold: {outlier['threshold']:.2f})")
            report_lines.append("")
        
        # Processing warnings
        if self.quality_issues['processing_warnings']:
            report_lines.append(f"PROCESSING WARNINGS ({len(self.quality_issues['processing_warnings'])}):")
            for warning in self.quality_issues['processing_warnings']:
                report_lines.append(f"  {warning}")
            report_lines.append("")
        
        # Summary statistics
        if self.summaries:
            all_summaries = pd.concat(self.summaries, ignore_index=True)
            report_lines.append("OVERALL STATISTICS:")
            report_lines.append(f"  Total regions detected: {len(all_summaries)}")
            report_lines.append(f"  Average regions per image: {len(all_summaries) / len(self.results):.1f}")
            report_lines.append(f"  Red intensity range: {all_summaries['red_intensity'].min():.2f} - {all_summaries['red_intensity'].max():.2f}")
            report_lines.append(f"  Area range: {all_summaries['area_um2'].min():.2f} - {all_summaries['area_um2'].max():.2f} um²")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.dirs['summaries'] / 'quality_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Quality report saved to {report_path}")
        
        # Print key issues to console
        if actual_duplicates or self.quality_issues['retake_images'] or self.quality_issues['insufficient_regions']:
            print("\nQUALITY ISSUES DETECTED:")
            if actual_duplicates:
                print(f"  - {len(actual_duplicates)} duplicate image names")
            if self.quality_issues['retake_images']:
                print(f"  - {len(self.quality_issues['retake_images'])} retake images")
            if self.quality_issues['insufficient_regions']:
                print(f"  - {len(self.quality_issues['insufficient_regions'])} images with insufficient regions")
            if self.quality_issues['red_outliers']:
                print(f"  - {len(self.quality_issues['red_outliers'])} red intensity outliers")
            print(f"  See {report_path} for details")
    
    def _process_single_pair(self, red_path, blue_path, index):
        """Process a single image pair."""
        red_img = load_as_gray(red_path)
        blue_img = load_as_gray(blue_path)

        seg_result = segment_crypts_full(
            (red_img, blue_img),
            blob_size_px=self.blob_size_px,
            debug=self.debug,
            scoring_weights=self.scoring_weights,
            max_regions=self.max_regions,
            image_name=red_path.stem,
        )

        pixel_dim = None
        summary = None
        if isinstance(self.pixel_dims, dict) and self.pixel_dims:
            pixel_dim = calculate_pixel_dimensions(red_path.name, self.pixel_dims)
        elif isinstance(self.pixel_dims, (int, float)):
            pixel_dim = float(self.pixel_dims)

        if pixel_dim is not None:
            summary = generate_label_summary(
                seg_result.labels,
                seg_result.red_channel,
                pixel_dim,
                background_tissue_intensity=seg_result.background_tissue_intensity,
                average_crypt_intensity=seg_result.average_crypt_intensity,
                debug=self.debug,
            )

        try:
            labels_dir = self.dirs.get('results', self.output_dir) / 'npy'
            seg_result.save_labels(red_path.stem, labels_dir)
        except Exception as exc:
            if self.debug:
                print(f"[BULK DEBUG] Could not save labels array for {red_path.stem}: {exc}")

        result_payload = {
            'red_path': red_path,
            'blue_path': blue_path,
            'selected_labels': seg_result.labels,
            'summary': summary,
            'standard_visual': seg_result.standard_visual,
            'rgb_image': seg_result.rgb_image,
            'red_channel': seg_result.red_channel,
            'blue_channel': seg_result.blue_channel,
            'metrics': {
                'background_tissue_intensity': seg_result.background_tissue_intensity,
                'average_crypt_intensity': seg_result.average_crypt_intensity,
                'pixel_dim': pixel_dim,
            },
            'debug_info': None,
        }

        if self.debug:
            debug_result = {
                'selected_labels': seg_result.labels,
                'label_summary': summary,
                'standard_visual': seg_result.standard_visual,
                'visuals': seg_result.debug_visuals,
                'processing_info': {
                    'red_channel': seg_result.red_channel,
                    'blue_channel': seg_result.blue_channel,
                    'rgb_combined': seg_result.rgb_image,
                    'extractor_debug': seg_result.extractor_debug,
                    'initial_labels': seg_result.initial_labels,
                    'selected_labels': seg_result.labels,
                    'selection_properties': seg_result.selection_debug.get('properties_df')
                    if seg_result.selection_debug
                    else None,
                    'selected_label_ids': seg_result.selection_debug.get('selected_labels')
                    if seg_result.selection_debug
                    else None,
                },
            }

            self._save_debug_visuals(debug_result, red_path.stem, index)
            self._save_standard_visualization(seg_result.standard_visual, red_path.stem)
            result_payload['debug_info'] = debug_result
        else:
            self._save_standard_visualization(seg_result.standard_visual, red_path.stem)

        return result_payload
    
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
            
            # Save by-image summary with metadata
            groupby_cols = ['image_name', 'is_retake', 'subdir']
            by_image = all_summaries.groupby(groupby_cols).agg({
                'area_um2': ['count', 'sum', 'mean', 'std'],
                'red_sum': ['sum', 'mean'],
                'red_intensity': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            by_image.columns = ['_'.join(col).strip() for col in by_image.columns]
            by_image = by_image.reset_index()
            by_image_path = self.dirs['summaries'] / 'by_image_summary.csv'
            by_image.to_csv(by_image_path, index=False)
            print(f"Saved by-image summary to {by_image_path}")
            
            # Save retake-only summary if any retakes exist
            retake_summaries = all_summaries[all_summaries['is_retake'] == True]
            if len(retake_summaries) > 0:
                retake_path = self.dirs['summaries'] / 'retake_images_summary.csv'
                retake_summaries.to_csv(retake_path, index=False)
                print(f"Saved retake images summary to {retake_path}")
        
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
                rgb_img = result.get('rgb_image')
                if rgb_img is None:
                    raise ValueError("Missing RGB image for quick check")

                selected_boundaries = find_boundaries(result['selected_labels'], mode='inner')
                overlay = rgb_img.copy()
                overlay[selected_boundaries] = [255, 0, 0]

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
