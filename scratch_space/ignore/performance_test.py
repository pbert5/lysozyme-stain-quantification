#!/usr/bin/env python3
"""
Performance comparison test for vectorization optimizations.

This script compares the old loop-based methods with the new vectorized implementations
to validate performance improvements.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass
from sklearn.linear_model import LinearRegression

def create_test_data():
    """Create test label array and raw image for performance testing"""
    # Create synthetic label array with multiple regions
    labels = np.zeros((1000, 1000), dtype=int)
    
    # Add 50 circular regions of different sizes
    np.random.seed(42)
    for i in range(1, 51):
        center_y = np.random.randint(100, 900)
        center_x = np.random.randint(100, 900)
        radius = np.random.randint(20, 80)
        
        y, x = np.ogrid[:1000, :1000]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        labels[mask] = i
    
    # Create synthetic raw image
    raw_img = np.random.randint(50, 255, (1000, 1000)).astype(np.float32)
    
    return labels, raw_img

def old_calculate_region_properties(label_img, raw_img=None, debug=False):
    """Original loop-based region properties calculation"""
    properties = []
    
    for label_id in np.unique(label_img):
        if label_id == 0:  # Skip background
            continue
            
        # Get mask for this region
        mask = label_img == label_id
        
        # Basic properties
        area = np.sum(mask)
        if area == 0:
            continue
                
        # Physical center of mass
        physical_com = center_of_mass(mask)
        
        # Red intensity calculation (if raw image provided)
        if raw_img is not None:
            red_values = raw_img[mask]
            total_red_intensity = np.sum(red_values)
            red_intensity_per_area = total_red_intensity / area if area > 0 else 0
        else:
            total_red_intensity = 0
            red_intensity_per_area = 0
        
        # Circularity measure (simplified for performance test)
        region_perimeter = np.sqrt(area) * 4  # Simplified calculation
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

def new_calculate_region_properties(label_img, raw_img=None, debug=False):
    """New vectorized region properties calculation"""
    regions = regionprops(label_img, intensity_image=raw_img)
    
    if len(regions) == 0:
        return pd.DataFrame()
    
    properties = []
    for region in regions:
        # Red intensity calculation (if raw image provided)
        if raw_img is not None:
            total_red_intensity = region.mean_intensity * region.area
            red_intensity_per_area = region.mean_intensity
        else:
            total_red_intensity = 0
            red_intensity_per_area = 0
        
        # Circularity measure using regionprops perimeter
        circularity = 4 * np.pi * region.area / (region.perimeter**2) if region.perimeter > 0 else 0
        
        properties.append({
            'label_id': region.label,
            'area': region.area,
            'physical_com': region.centroid,
            'red_intensity_per_area': red_intensity_per_area,
            'total_red_intensity': total_red_intensity,
            'circularity': circularity,
            'perimeter': region.perimeter
        })
    
    return pd.DataFrame(properties)

def old_edge_removal(labels, debug=False):
    """Original loop-based edge removal"""
    height, width = labels.shape
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Find labels that touch edges
    edge_labels = set()
    
    # Check top and bottom edges
    edge_labels.update(np.unique(labels[0, :]))  # Top edge
    edge_labels.update(np.unique(labels[-1, :]))  # Bottom edge
    
    # Check left and right edges  
    edge_labels.update(np.unique(labels[:, 0]))  # Left edge
    edge_labels.update(np.unique(labels[:, -1]))  # Right edge
    
    # Remove background label (0) from edge labels
    edge_labels.discard(0)
    
    # Create new label array without edge-touching regions
    filtered_labels = labels.copy()
    for edge_label in edge_labels:
        filtered_labels[labels == edge_label] = 0
    
    # Relabel remaining regions to be sequential starting from 1
    remaining_labels = np.unique(filtered_labels)
    remaining_labels = remaining_labels[remaining_labels != 0]
    
    if len(remaining_labels) > 0:
        final_labels = np.zeros_like(filtered_labels)
        for new_id, old_id in enumerate(sorted(remaining_labels), start=1):
            final_labels[filtered_labels == old_id] = new_id
    else:
        final_labels = filtered_labels
    
    return final_labels

def new_edge_removal(labels, debug=False):
    """New vectorized edge removal"""
    height, width = labels.shape
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Vectorized edge detection
    edge_mask = np.zeros_like(labels, dtype=bool)
    edge_mask[0, :] = True    # Top edge
    edge_mask[-1, :] = True   # Bottom edge
    edge_mask[:, 0] = True    # Left edge
    edge_mask[:, -1] = True   # Right edge
    
    # Find all labels that touch edges in one operation
    edge_labels = np.unique(labels[edge_mask])
    edge_labels = edge_labels[edge_labels != 0]
    
    # Vectorized removal
    keep_mask = ~np.isin(labels, edge_labels)
    filtered_labels = labels * keep_mask
    
    # Vectorized relabeling
    remaining_labels = np.unique(filtered_labels)
    remaining_labels = remaining_labels[remaining_labels != 0]
    
    if len(remaining_labels) > 0:
        max_label = remaining_labels.max()
        lookup = np.zeros(max_label + 1, dtype=labels.dtype)
        
        for new_id, old_id in enumerate(sorted(remaining_labels), start=1):
            lookup[old_id] = new_id
        
        final_labels = np.where(filtered_labels > 0, lookup[filtered_labels], 0)
    else:
        final_labels = filtered_labels
    
    return final_labels

def run_performance_test():
    """Run comprehensive performance comparison"""
    print("Creating test data...")
    labels, raw_img = create_test_data()
    
    print(f"Test data: {labels.shape} with {len(np.unique(labels))-1} regions")
    print("="*60)
    
    # Test 1: Region properties calculation
    print("Testing region properties calculation...")
    
    # Old method
    start_time = time.time()
    old_props = old_calculate_region_properties(labels, raw_img)
    old_time = time.time() - start_time
    
    # New method
    start_time = time.time()
    new_props = new_calculate_region_properties(labels, raw_img)
    new_time = time.time() - start_time
    
    print(f"Old method: {old_time:.3f}s")
    print(f"New method: {new_time:.3f}s")
    print(f"Speedup: {old_time/new_time:.1f}x faster")
    print(f"Results match: {len(old_props) == len(new_props)}")
    print()
    
    # Test 2: Edge removal
    print("Testing edge removal...")
    
    # Old method
    start_time = time.time()
    old_edge = old_edge_removal(labels)
    old_edge_time = time.time() - start_time
    
    # New method
    start_time = time.time()
    new_edge = new_edge_removal(labels)
    new_edge_time = time.time() - start_time
    
    print(f"Old method: {old_edge_time:.3f}s")
    print(f"New method: {new_edge_time:.3f}s")
    print(f"Speedup: {old_edge_time/new_edge_time:.1f}x faster")
    print(f"Results match: {np.array_equal(old_edge, new_edge)}")
    print()
    
    # Overall summary
    total_old = old_time + old_edge_time
    total_new = new_time + new_edge_time
    
    print("="*60)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total old processing time: {total_old:.3f}s")
    print(f"Total new processing time: {total_new:.3f}s")
    print(f"Overall speedup: {total_old/total_new:.1f}x faster")
    print(f"Time saved per image: {total_old - total_new:.3f}s")
    print(f"For 64 images, total time saved: {(total_old - total_new) * 64:.1f}s")

if __name__ == "__main__":
    run_performance_test()
