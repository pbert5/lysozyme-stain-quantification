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
IMG_DIR = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\lysozyme images\Jej LYZ")        # Directory containing TIFF images
RESULTS_DIR = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\Jej LYZ")  # Output directory
RED_CHANNEL = "RFP"        # Identifier substring for red channel files
BLUE_CHANNEL = "DAPI"      # Identifier substring for blue channel files
DEBUG = False              # Set True for verbose errors / debug info
IMAGE_LIMIT = 4            # Limit number of images for testing (set to None for all images)

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

    # Limit images for testing if specified
    if IMAGE_LIMIT is not None:
        original_count = len(image_pairs)
        image_pairs = image_pairs[:IMAGE_LIMIT]
        print(f"Limited to first {len(image_pairs)} image pairs for testing (out of {original_count} total)")

    # Initialize and run bulk processor
    processor = BulkProcessor(
        output_dir=RESULTS_DIR,
        pixel_dims=PIXEL_DIMS,
        debug=DEBUG
    )

    try:
        results = processor.process_pairs(image_pairs)
        print(f"Processing complete. Results saved to {RESULTS_DIR}")
        print(f"Processed {len(results)} image pairs successfully")
    except Exception as e:
        print(f"Error during processing: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
