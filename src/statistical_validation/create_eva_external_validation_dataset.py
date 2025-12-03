#!/usr/bin/env python3
"""
Create clean dataset for external validation
Filter to Eva set (both sources), keep manual reviewers separate
Focus on interpretability with clear column naming
"""

import pandas as pd
import os

# Read the manual data with metrics (has all the fluorescence data we need)
manual_data = pd.read_csv("outputs/final/manual_uncollapsed_with_metrics.csv")

# Read the mapping file to get subject keys
mapping_data = pd.read_csv("outputs/final/manual_to_auto_mapping.csv")

# Extract just what we need from mapping: subject_key relationship
subject_lookup = mapping_data[['manual_slice_source', 'mouse_id', 'manual_slice_name', 'subject_key']].drop_duplicates()

# Join to get subject keys
combined_data = manual_data.merge(
    subject_lookup,
    left_on=['manual_source', 'mouse_id', 'slice'],
    right_on=['manual_slice_source', 'mouse_id', 'manual_slice_name'],
    how='left'
)

# Filter to only Eva sources (both Adam and Eva's manual ratings)
# Eva sources are the two CSV files
eva_data = combined_data[combined_data['manual_source'].isin([
    'image_j_quantification_eva_04222025(Adam).csv',
    'image_j_quantification_eva_04222025(Eva).csv'
])]

# Create clean column names and select relevant data
eva_data['manual_reviewer'] = eva_data['manual_source'].map({
    'image_j_quantification_eva_04222025(Adam).csv': 'Adam',
    'image_j_quantification_eva_04222025(Eva).csv': 'Eva'
})

clean_data = eva_data[[
    'manual_reviewer',
    'mouse_id',
    'slice',
    'subject_key',
    'count',
    'total_area_px',
    'avg_size_px',
    'mean_intensity_px',
    'manual_integrated_px',
    'um_per_px',
    'manual_integrated_um2',
    'percent_area'
]].copy()

# Rename columns for clarity
clean_data.columns = [
    'manual_reviewer',
    'mouse_id',
    'slice_name',
    'subject_key',
    'crypt_count',
    'total_area_px',
    'avg_crypt_size_px',
    'mean_fluorescence_px',
    'integrated_fluorescence_px',
    'um_per_px',
    'integrated_fluorescence_um2',
    'percent_tissue_area'
]

# Remove rows with no data (NaN crypt counts)
final_data = clean_data.dropna(subset=['crypt_count'])

# Reorder columns for readability
final_data = final_data[[
    'manual_reviewer',
    'mouse_id',
    'slice_name',
    'subject_key',
    'crypt_count',
    # Pixel-based measurements
    'total_area_px',
    'avg_crypt_size_px',
    'mean_fluorescence_px',
    'integrated_fluorescence_px',
    # Micrometers-based measurements
    'um_per_px',
    'integrated_fluorescence_um2',
    # Other
    'percent_tissue_area'
]]

# Sort for easy reading
final_data = final_data.sort_values(['manual_reviewer', 'mouse_id', 'slice_name'])

# Write output
output_path = "outputs/final/eva_set_external_validation.csv"
final_data.to_csv(output_path, index=False)

# Print summary
print("\n=== Dataset Summary ===")
print(f"Total rows: {len(final_data)}")
print("\nRows per reviewer:")
print(final_data['manual_reviewer'].value_counts())
print(f"\nUnique mice: {final_data['mouse_id'].nunique()}")
print(f"\nColumn names:")
print(list(final_data.columns))
print("\nFirst few rows:")
print(final_data.head(10).to_string())
print(f"\nOutput saved to: {output_path}")
print(f"\nData description:")
print("- manual_reviewer: Adam or Eva (who manually rated the slice)")
print("- mouse_id: Mouse identifier")
print("- slice_name: Slice/image name")
print("- subject_key: Unique key for matching with automated analysis")
print("- crypt_count: Number of crypts manually identified")
print("- *_px: Measurements in pixels")
print("- *_um2: Measurements in square micrometers (um²)")
print("- integrated_fluorescence: Total fluorescence (sum of intensities)")
