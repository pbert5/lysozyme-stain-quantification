#!/usr/bin/env python3
"""
Main entry point for the lysozyme stain quantification pipeline.
All configuration is now hard-coded below for easier usage (no CLI args).
"""

import sys
from pathlib import Path

from pipeline.bulk_processor import BulkProcessor
from utils.file_utils import validate_directories, find_image_pairs

# ------------------------------------------------------------------
# User configuration (edit these)
# ------------------------------------------------------------------
IMG_DIR = Path(r"/home/phillip/documents/lysozyme/lysozyme images")        # Directory containing TIFF images
RESULTS_DIR = Path(r"/home/phillip/documents/lysozyme/results/All")  # Output directory
RED_CHANNEL = "RFP"        # Identifier substring for red channel files
BLUE_CHANNEL = "DAPI"      # Identifier substring for blue channel files
DEBUG = False              # Set True for verbose errors / debug info
IMAGE_LIMIT = 40         # Limit number of images for testing (set to None for all images)

# Pixel dimensions in micrometers (choose one of the following options):

# 1. Use default dict (multiple objectives)
PIXEL_DIMS = {
    "default": 0.4476,
    "40x-4": 0.2253
}

# 2. Or force a single pixel size (uncomment below)
# PIXEL_DIMS = {"default": 0.4476}

# 3. Or set to None and let downstream code handle absence
# PIXEL_DIMS = None

# Scoring weights for region selection (lower is better)
SCORING_WEIGHTS = {
    'circularity': 0.1,     # Most important - want circular regions
    'area': 0.4,            # Second - want larger areas
    'line_fit': 0.2,        # Moderate - want aligned regions
    'red_intensity': 0.3,   # Least - want bright regions
}

# Maximum number of regions to select
MAX_REGIONS = 5
# ------------------------------------------------------------------


def main():
    # Validate directories
    if not validate_directories(IMG_DIR, RESULTS_DIR):
        sys.exit(1)

    # Find image pairs
    print(f"Searching for image pairs in {IMG_DIR}...")
    image_pairs = find_image_pairs(IMG_DIR, RED_CHANNEL, BLUE_CHANNEL)

    if not image_pairs:
        print(f"No image pairs found with channels {RED_CHANNEL}/{BLUE_CHANNEL}")
        sys.exit(1)

    print(f"Found {len(image_pairs)} image pairs")
    
    # Debug: Check for duplicates
    if DEBUG:
        pair_names = [pair[0].name for pair in image_pairs]
        unique_names = set(pair_names)
        if len(pair_names) != len(unique_names):
            print(f"[DEBUG] WARNING: Duplicate image pairs detected!")
            for name in unique_names:
                count = pair_names.count(name)
                if count > 1:
                    print(f"[DEBUG] {name} appears {count} times")
        
        print(f"[DEBUG] First 5 pairs:")
        for i, (red, blue) in enumerate(image_pairs[:5]):
            print(f"[DEBUG] {i+1}: {red.name} + {blue.name}")

    # Limit images for testing if specified
    if IMAGE_LIMIT is not None:
        original_count = len(image_pairs)
        image_pairs = image_pairs[:IMAGE_LIMIT]
        print(f"Limited to first {len(image_pairs)} image pairs for testing (out of {original_count} total)")

    # Initialize and run bulk processor
    processor = BulkProcessor(
        output_dir=RESULTS_DIR,
        pixel_dims=PIXEL_DIMS,
        scoring_weights=SCORING_WEIGHTS,
        max_regions=MAX_REGIONS,
        debug=DEBUG
    )

    try:
        results = processor.process_pairs(image_pairs)
        print(f"Processing complete. Results saved to {RESULTS_DIR}")
        print(f"Processed {len(results)} image pairs successfully")
        
        # Ask if user wants to run visual inspector
        print("\nWould you like to run the visual inspector to review results? (y/n): ", end="")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            print("Launching visual inspector...")
            from visual_inspector import VisualInspector
            inspector = VisualInspector(RESULTS_DIR)
            inspector.run()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
