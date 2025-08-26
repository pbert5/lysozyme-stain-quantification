"""
Scoring-based region selector that replaces the complex merge pipeline.
Instead of merging, we score all detections based on quality metrics and select the best ones.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.linear_model import LinearRegression
from skimage.measure import regionprops, perimeter
import matplotlib.pyplot as plt


class ScoringSelector:
    """Selector for choosing the best regions based on scoring criteria instead of merging."""
    
    def __init__(self, label_img, raw_img=None, debug=False, max_regions=5, weights=None):
        """
        Initialize the scoring selector.
        
        Args:
            label_img: Labeled image array with detected regions
            raw_img: Raw red channel image for quality assessment
            debug: Whether to enable debug output
            max_regions: Maximum number of regions to select (default: 5)
            weights: Dict of scoring weights (default weights used if None)
        """
        self.label_img = label_img.copy()
        self.raw_img = raw_img
        self.debug = debug
        self.max_regions = max_regions
        
        # Default weights if none provided
        self.weights = weights if weights is not None else {
            'circularity': 0.35,    # Most important - want circular regions
            'area': 0.25,           # Second - want consistent sizes
            'line_fit': 0.15,       # Moderate - want aligned regions
            'red_intensity': 0.15,  # Moderate - want bright regions
            'com_consistency': 0.10 # Least - center consistency
        }
        
        # Results storage
        self.properties_df = None
        self.selected_labels = None
        self.scoring_history = []

    def calculate_region_properties(self):
        """Calculate comprehensive properties for each detected region"""
        if self.debug:
            unique_labels = np.unique(self.label_img)
            print(f"[SCORING DEBUG] Input label array has {len(unique_labels)} unique labels: {unique_labels}")
        
        properties = []
        
        for label_id in np.unique(self.label_img):
            if label_id == 0:  # Skip background
                continue
                
            # Get mask for this region
            mask = self.label_img == label_id
            
            # Basic properties
            area = np.sum(mask)
            if area == 0:
                if self.debug:
                    print(f"[SCORING WARNING] Label {label_id} has zero area, skipping")
                continue
            
            if self.debug:
                print(f"[SCORING DEBUG] Processing label {label_id}: area = {area} pixels")
                
            # Physical center of mass
            physical_com = center_of_mass(mask)
            
            # Red intensity calculation (if raw image provided)
            if self.raw_img is not None:
                red_values = self.raw_img[mask]
                # Red intensity per area - this is what we want for scoring
                total_red_intensity = np.sum(red_values)
                red_intensity_per_area = total_red_intensity / area if area > 0 else 0
            else:
                total_red_intensity = 0
                red_intensity_per_area = 0
            
            # Circularity measure
            region_perimeter = perimeter(mask.astype(int))
            circularity = 4 * np.pi * area / (region_perimeter**2) if region_perimeter > 0 else 0
            
            properties.append({
                'label_id': label_id,
                'area': area,
                'physical_com': physical_com,
                'red_intensity_per_area': red_intensity_per_area,
                'total_red_intensity': total_red_intensity,
                'circularity': circularity,
                'perimeter': region_perimeter
            })
        
        return pd.DataFrame(properties)

    def calculate_line_fit_deviation(self, properties_df):
        """Calculate how far each detection center is from line of best fit through all centers,
        normalized by region size (radius approximation)"""
        if len(properties_df) < 2:
            properties_df['distance_from_line'] = 0
            properties_df['normalized_line_distance'] = 0
            return properties_df
        
        # Get physical centers
        centers = np.array([prop for prop in properties_df['physical_com']])
        
        # Fit line through centers
        X = centers[:, 1].reshape(-1, 1)  # x coordinates
        y = centers[:, 0]  # y coordinates
        
        reg = LinearRegression().fit(X, y)
        
        # Calculate distance from line for each point
        distances = []
        normalized_distances = []
        
        for idx, center in enumerate(centers):
            x, y = center[1], center[0]
            # Distance from point to line ax + by + c = 0
            # Line: y = mx + b -> mx - y + b = 0
            m = reg.coef_[0]
            b = reg.intercept_
            distance = abs(m * x - y + b) / np.sqrt(m**2 + 1)
            distances.append(distance)
            
            # Normalize by approximate radius (sqrt(area/2))
            area = properties_df.iloc[idx]['area']
            radius_approx = np.sqrt(area / 2)
            normalized_distance = distance / radius_approx if radius_approx > 0 else 0
            normalized_distances.append(normalized_distance)
        
        properties_df['distance_from_line'] = distances
        properties_df['normalized_line_distance'] = normalized_distances
        return properties_df

    def score_regions(self, properties_df):
        """
        Calculate quality scores for all regions.
        Lower scores are better (easier to rank by ascending order).
        """
        if len(properties_df) == 0:
            return properties_df
        
        # Score components (normalized to 0-1, where 0 is best)
        
        # 1. Circularity score (higher circularity is better, so invert)
        max_circularity = properties_df['circularity'].max()
        if max_circularity > 0:
            properties_df['circularity_score'] = 1 - (properties_df['circularity'] / max_circularity)
        else:
            properties_df['circularity_score'] = 1.0
        
        # 2. Area score (larger areas are better, so invert)
        max_area = properties_df['area'].max()
        if max_area > 0:
            properties_df['area_score'] = 1 - (properties_df['area'] / max_area)
        else:
            properties_df['area_score'] = 0.0
        
        # 3. Line fit score (closer to line is better, using normalized distance)
        max_line_dist = properties_df['normalized_line_distance'].max()
        if max_line_dist > 0:
            properties_df['line_fit_score'] = properties_df['normalized_line_distance'] / max_line_dist
        else:
            properties_df['line_fit_score'] = 0.0
        
        # 4. Red intensity score (higher intensity is better, so invert)
        # This is already calculated as red_intensity_per_area = total_red_intensity / area
        max_red_intensity = properties_df['red_intensity_per_area'].max()
        if max_red_intensity > 0:
            properties_df['red_intensity_score'] = 1 - (properties_df['red_intensity_per_area'] / max_red_intensity)
        else:
            properties_df['red_intensity_score'] = 1.0
        
        # Composite quality score (weighted combination, lower is better)
        # Use weights passed during initialization - drop COM consistency
        properties_df['quality_score'] = (
            self.weights.get('circularity', 0.4) * properties_df['circularity_score'] +
            self.weights.get('area', 0.3) * properties_df['area_score'] +
            self.weights.get('line_fit', 0.2) * properties_df['line_fit_score'] +
            self.weights.get('red_intensity', 0.1) * properties_df['red_intensity_score']
        )
        
        # Store scoring history for debugging
        if self.debug:
            self.scoring_history.append({
                'max_circularity': max_circularity,
                'max_area': max_area,
                'max_line_dist': max_line_dist,
                'max_red_intensity': max_red_intensity,
                'weights': self.weights
            })
        
        return properties_df.sort_values('quality_score')

    def select_best_regions(self, properties_df):
        """Select the best N regions based on quality scores"""
        
        # Sort by quality score (ascending - lower is better)
        sorted_df = properties_df.sort_values('quality_score')
        
        # Select top N regions
        n_to_select = min(self.max_regions, len(sorted_df))
        best_regions = sorted_df.head(n_to_select)
        
        if self.debug:
            print(f"[SCORING DEBUG] Selected top {n_to_select} regions out of {len(sorted_df)} total")
            print(f"[SCORING DEBUG] Selected labels: {best_regions['label_id'].tolist()}")
            print(f"[SCORING DEBUG] Quality scores: {best_regions['quality_score'].tolist()}")
            
            # Print detailed scoring breakdown for top regions
            for idx, row in best_regions.iterrows():
                print(f"[SCORING DEBUG] Label {row['label_id']}: "
                      f"area={row['area']:.1f}, circ={row['circularity']:.3f}, "
                      f"red_int={row['red_intensity_per_area']:.2f}, "
                      f"line_dist={row['normalized_line_distance']:.2f}, "
                      f"final_score={row['quality_score']:.3f}")
        
        return best_regions['label_id'].tolist()

    def create_filtered_labels(self, selected_label_ids):
        """Create new label array with only selected regions"""
        
        if self.debug:
            print(f"[SCORING DEBUG] Creating filtered labels from {len(selected_label_ids)} selected regions")
            print(f"[SCORING DEBUG] Selected IDs: {selected_label_ids}")
        
        # Create new label array with only selected regions
        filtered_labels = np.zeros_like(self.label_img)
        
        # Relabel selected regions with sequential IDs starting from 1
        for new_id, old_id in enumerate(selected_label_ids, start=1):
            mask = self.label_img == old_id
            filtered_labels[mask] = new_id
            
            if self.debug:
                pixels_count = np.sum(mask)
                print(f"[SCORING DEBUG] Relabeled {old_id} -> {new_id}, {pixels_count} pixels")
        
        if self.debug:
            print(f"[SCORING DEBUG] Filtered label array unique labels: {np.unique(filtered_labels)}")
        
        return filtered_labels

    def select(self):
        """
        Main selection function that scores all regions and selects the best ones.
        
        Returns:
            Filtered label array containing only the best regions
        """
        if self.debug:
            print("Starting quality-based region selection...")
        
        # Step 1: Calculate region properties
        self.properties_df = self.calculate_region_properties()
        
        if len(self.properties_df) == 0:
            if self.debug:
                print("[SCORING DEBUG] No regions found to score")
            return np.zeros_like(self.label_img)
        
        # Step 2: Calculate line fit deviations
        self.properties_df = self.calculate_line_fit_deviation(self.properties_df)
        
        # Step 3: Score all regions
        self.properties_df = self.score_regions(self.properties_df)
        
        if self.debug:
            print(f"[SCORING DEBUG] Scored {len(self.properties_df)} regions")
        
        # Step 4: Select best regions
        self.selected_labels = self.select_best_regions(self.properties_df)
        
        # Step 5: Create filtered label array
        filtered_labels = self.create_filtered_labels(self.selected_labels)
        
        if self.debug:
            original_count = len(np.unique(self.label_img)) - 1
            selected_count = len(np.unique(filtered_labels)) - 1
            print(f"[SCORING DEBUG] Selection complete: {original_count} -> {selected_count} regions")
        
        return filtered_labels

    def get_debug_info(self):
        """Return debug information about the selection process"""
        return {
            'properties_df': self.properties_df,
            'selected_labels': self.selected_labels,
            'scoring_history': self.scoring_history,
            'original_regions': len(np.unique(self.label_img)) - 1,
            'selected_regions': len(self.selected_labels) if self.selected_labels else 0
        }

    def plot_scoring_results(self, save_path=None):
        """Plot scoring results for visualization"""
        if self.properties_df is None:
            print("No scoring results to plot. Run select() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot individual score components - removed com_consistency_score
        score_columns = ['circularity_score', 'area_score', 'line_fit_score', 
                        'red_intensity_score', 'quality_score']
        
        # Add a basic info plot for the 6th subplot
        score_columns.append('area')  # Show actual area values
        
        for i, col in enumerate(score_columns):
            if col in self.properties_df.columns:
                self.properties_df.plot.scatter(x='label_id', y=col, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col.replace("_", " ").title()}')
                axes[i].set_xlabel('Label ID')
                axes[i].set_ylabel('Score' if 'score' in col else 'Value')
                
                # Highlight selected regions
                if self.selected_labels:
                    selected_df = self.properties_df[self.properties_df['label_id'].isin(self.selected_labels)]
                    axes[i].scatter(selected_df['label_id'], selected_df[col], 
                                  c='red', s=100, alpha=0.8, marker='x', linewidth=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
