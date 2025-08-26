"""
New merge pipeline for combining adjacent labeled regions based on quality assessment and boundary sharing.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.linear_model import LinearRegression
from skimage.measure import regionprops, perimeter
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt


class MergePipeline:
    """Pipeline for merging adjacent labeled regions based on quality assessment and boundary sharing."""
    
    def __init__(self, label_img, raw_img=None, debug=False):
        """
        Initialize the merge pipeline.
        
        Args:
            label_img: Labeled image array
            raw_img: Raw red channel image for quality assessment
            debug: Whether to enable debug output
        """
        self.label_img = label_img.copy()
        self.raw_img = raw_img
        self.debug = debug
        
        # Results storage
        self.properties_df = None
        self.merge_history = []
        self.merged_label_array = None

    def calculate_region_properties(self):
        """Calculate comprehensive properties for each detected region"""
        properties = []
        
        for label_id in np.unique(self.label_img):
            if label_id == 0:  # Skip background
                continue
                
            # Get mask for this region
            mask = self.label_img == label_id
            
            # Basic properties
            area = np.sum(mask)
            if area == 0:
                continue
                
            # Physical center of mass
            physical_com = center_of_mass(mask)
            
            # Red intensity weighted center of mass (if raw image provided)
            if self.raw_img is not None:
                red_values = self.raw_img[mask]
                if np.sum(red_values) > 0:
                    # Get coordinates of pixels in this region
                    y_coords, x_coords = np.where(mask)
                    # Weight by red intensity
                    red_com_y = np.sum(y_coords * red_values) / np.sum(red_values)
                    red_com_x = np.sum(x_coords * red_values) / np.sum(red_values)
                    red_com = (red_com_y, red_com_x)
                else:
                    red_com = physical_com
                    
                # Red intensity per area
                total_red_intensity = np.sum(red_values)
                red_intensity_per_area = total_red_intensity / area if area > 0 else 0
            else:
                red_com = physical_com
                total_red_intensity = 0
                red_intensity_per_area = 0
            
            # Distance between physical and red-weighted center of mass
            com_distance = np.sqrt((physical_com[0] - red_com[0])**2 + (physical_com[1] - red_com[1])**2)
            com_distance_normalized = com_distance / np.sqrt(area)
            
            # Circularity measure
            region_perimeter = perimeter(mask.astype(int))
            circularity = 4 * np.pi * area / (region_perimeter**2) if region_perimeter > 0 else 0
            
            properties.append({
                'label_id': label_id,
                'area': area,
                'physical_com': physical_com,
                'red_com': red_com,
                'com_distance': com_distance,
                'com_distance_normalized': com_distance_normalized,
                'red_intensity_per_area': red_intensity_per_area,
                'total_red_intensity': total_red_intensity,
                'circularity': circularity
            })
        
        return pd.DataFrame(properties)

    def calculate_circularity_from_line_fit(self, properties_df):
        """Calculate how far each detection center is from line of best fit"""
        if len(properties_df) < 2:
            properties_df['distance_from_line'] = 0
            return properties_df
        
        # Get physical centers
        centers = np.array([prop for prop in properties_df['physical_com']])
        
        # Fit line through centers
        X = centers[:, 1].reshape(-1, 1)  # x coordinates
        y = centers[:, 0]  # y coordinates
        
        reg = LinearRegression().fit(X, y)
        
        # Calculate distance from line for each point
        distances = []
        for center in centers:
            x, y = center[1], center[0]
            # Distance from point to line ax + by + c = 0
            # Line: y = mx + b -> mx - y + b = 0
            m = reg.coef_[0]
            b = reg.intercept_
            distance = abs(m * x - y + b) / np.sqrt(m**2 + 1)
            distances.append(distance)
        
        properties_df['distance_from_line'] = distances
        return properties_df

    def quality_score_regions(self, properties_df):
        """Calculate quality scores and identify best regions"""
        if len(properties_df) == 0:
            return properties_df
        
        # Normalize metrics (smaller distance from median area is better)
        median_area = properties_df['area'].median()
        properties_df['area_deviation'] = np.abs(properties_df['area'] - median_area) / median_area
        
        # Normalize other metrics to [0, 1] range
        for col in ['distance_from_line', 'com_distance_normalized', 'area_deviation']:
            if col in properties_df.columns:
                max_val = properties_df[col].max()
                if max_val > 0:
                    properties_df[f'{col}_norm'] = properties_df[col] / max_val
                else:
                    properties_df[f'{col}_norm'] = 0
        
        # Normalize red_intensity_per_area (higher is better, so invert)
        max_red = properties_df['red_intensity_per_area'].max()
        if max_red > 0:
            properties_df['red_intensity_norm'] = 1 - (properties_df['red_intensity_per_area'] / max_red)
        else:
            properties_df['red_intensity_norm'] = 0
        
        # Circularity (higher is better, so invert)
        max_circ = properties_df['circularity'].max()
        if max_circ > 0:
            properties_df['circularity_norm'] = 1 - (properties_df['circularity'] / max_circ)
        else:
            properties_df['circularity_norm'] = 0
        
        # Calculate composite quality score (lower is better)
        properties_df['quality_score'] = (
            properties_df['distance_from_line_norm'] +
            properties_df['com_distance_normalized_norm'] +
            properties_df['area_deviation_norm'] +
            properties_df['red_intensity_norm'] +
            properties_df['circularity_norm']
        ) / 5
        
        return properties_df.sort_values('quality_score')

    def find_boundary_sharing_candidates(self, properties_df):
        """Find regions that share significant boundary length"""
        merge_candidates = []
        
        # Skip background region (label 1 is typically the large background region)
        actual_stains = properties_df[properties_df['label_id'] != 1].copy()
        
        for i, row1 in actual_stains.iterrows():
            for j, row2 in actual_stains.iterrows():
                if row1['label_id'] >= row2['label_id']:  # Avoid duplicates
                    continue
                    
                # Get boundaries of both regions
                mask1 = self.label_img == row1['label_id']
                mask2 = self.label_img == row2['label_id']
                
                boundary1 = find_boundaries(mask1, mode='inner')
                boundary2 = find_boundaries(mask2, mode='inner')
                
                # Dilate boundaries slightly to check for proximity
                dilated1 = binary_dilation(boundary1, structure=np.ones((3,3)))
                dilated2 = binary_dilation(boundary2, structure=np.ones((3,3)))
                
                # Check overlap
                overlap = np.sum(dilated1 & dilated2)
                boundary1_length = np.sum(boundary1)
                boundary2_length = np.sum(boundary2)
                
                if boundary1_length > 0 and boundary2_length > 0:
                    shared_ratio1 = overlap / boundary1_length
                    shared_ratio2 = overlap / boundary2_length
                    avg_shared_ratio = (shared_ratio1 + shared_ratio2) / 2
                    
                    # Quality difference
                    quality_diff = abs(row1['quality_score'] - row2['quality_score'])
                    
                    merge_candidates.append({
                        'label1': row1['label_id'],
                        'label2': row2['label_id'],
                        'shared_boundary_ratio': avg_shared_ratio,
                        'quality_diff': quality_diff,
                        'should_merge': avg_shared_ratio > 0.3 and quality_diff < 0.15
                    })
        
        return pd.DataFrame(merge_candidates)

    def apply_merges(self, merge_candidates):
        """Apply the boundary sharing merges"""
        merged_labels = self.label_img.copy()
        merge_history = []
        
        # Get actual stains (exclude background)
        actual_stains = self.properties_df[self.properties_df['label_id'] != 1].copy()
        
        # Apply boundary sharing merges
        merges_to_apply = merge_candidates[merge_candidates['should_merge'] == True].copy()
        
        if self.debug:
            print(f"Applying {len(merges_to_apply)} boundary-sharing merges:")
        
        for _, merge_info in merges_to_apply.iterrows():
            label1 = merge_info['label1']
            label2 = merge_info['label2']
            
            # Check if both labels still exist (haven't been merged already)
            if label1 not in np.unique(merged_labels) or label2 not in np.unique(merged_labels):
                continue
                
            # Determine merge direction - merge into the higher quality region
            quality1 = actual_stains[actual_stains['label_id'] == label1]['quality_score'].iloc[0]
            quality2 = actual_stains[actual_stains['label_id'] == label2]['quality_score'].iloc[0]
            
            if quality1 <= quality2:  # Lower score = better quality
                target, source = label1, label2
            else:
                target, source = label2, label1
                
            if self.debug:
                print(f"  Merging {source} -> {target} (shared boundary: {merge_info['shared_boundary_ratio']:.2f})")
            
            # Perform the merge
            merged_labels[merged_labels == source] = target
            merge_history.append({
                'source': source,
                'target': target,
                'shared_boundary_ratio': merge_info['shared_boundary_ratio'],
                'quality_diff': merge_info['quality_diff']
            })
        
        return merged_labels, merge_history

    def merge(self):
        """
        Main merging function that applies the quality-based boundary sharing strategy.
        
        Returns:
            Merged label array
        """
        if self.debug:
            print("Starting quality-based boundary sharing merge...")
        
        # Step 1: Calculate region properties
        self.properties_df = self.calculate_region_properties()
        self.properties_df = self.calculate_circularity_from_line_fit(self.properties_df)
        self.properties_df = self.quality_score_regions(self.properties_df)
        
        if self.debug:
            actual_stains = self.properties_df[self.properties_df['label_id'] != 1]
            print(f"Found {len(actual_stains)} actual stain regions")
            print(f"Top 5 quality regions: {actual_stains.head(5)['label_id'].tolist()}")
        
        # Step 2: Find boundary sharing candidates
        merge_candidates = self.find_boundary_sharing_candidates(self.properties_df)
        
        # Step 3: Apply merges
        self.merged_label_array, self.merge_history = self.apply_merges(merge_candidates)
        
        if self.debug:
            original_count = len(np.unique(self.label_img)) - 1
            merged_count = len(np.unique(self.merged_label_array)) - 1
            print(f"Merging complete: {original_count} -> {merged_count} regions ({len(self.merge_history)} merges)")
        
        return self.merged_label_array

    def get_debug_info(self):
        """Return debug information about the merging process"""
        return {
            'properties_df': self.properties_df,
            'merge_history': self.merge_history,
            'original_regions': len(np.unique(self.label_img)) - 1,
            'merged_regions': len(np.unique(self.merged_label_array)) - 1 if self.merged_label_array is not None else 0
        }
