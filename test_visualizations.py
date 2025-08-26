#!/usr/bin/env python3
"""
Test script for the visualization feature.
"""

import sys
from pathlib import Path

from pipeline.bulk_processor import BulkProcessor
from utils.file_utils import validate_directories, find_image_pairs

# Test configuration - just one folder for quick test
IMG_DIR = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\lysozyme images\Jej LYZ\G2")
RESULTS_DIR = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\visualization_test")
RED_CHANNEL = "RFP"
BLUE_CHANNEL = "DAPI"
DEBUG = False  # Test with debug off to ensure visualizations still work

PIXEL_DIMS = {
    "default": 0.4476,
    "40x-4": 0.2253
}

def main():
    print("=== TESTING VISUALIZATION FEATURE ===")
    print("Testing that visualizations are saved even when DEBUG=False")
    
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
    
    # Test with just the first 2 pairs
    test_pairs = image_pairs[:2]
    print(f"Testing with first {len(test_pairs)} pairs...")

    # Initialize and run bulk processor
    processor = BulkProcessor(
        output_dir=RESULTS_DIR,
        pixel_dims=PIXEL_DIMS,
        debug=DEBUG
    )

    try:
        results = processor.process_pairs(test_pairs)
        print(f"Processing complete. Results saved to {RESULTS_DIR}")
        print(f"Processed {len(results)} image pairs successfully")
        
        # Check if visualizations were created
        viz_dir = RESULTS_DIR / 'visualizations'
        if viz_dir.exists():
            viz_files = list(viz_dir.glob('*.png'))
            print(f"✅ SUCCESS: Found {len(viz_files)} visualization files in {viz_dir}")
            for viz_file in viz_files:
                print(f"  - {viz_file.name}")
        else:
            print(f"❌ FAILURE: Visualizations directory not found at {viz_dir}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
