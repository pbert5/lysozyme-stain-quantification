"""
Test script to run watershed processing on actual RFP/DAPI images
without showing plots (for automated testing).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from blob_det import BlobDetector, get_rfp_dapi_pairs
from primary import RFPDAPIProcessor
import numpy as np

def test_watershed_processing():
    """Test watershed processing on real images."""
    print("Testing watershed processing on real RFP/DAPI images...")
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root.absolute()}")
    
    # Find image pairs
    images_root = project_root / 'lysozyme images'
    pairs = get_rfp_dapi_pairs(images_root, max_pairs=2)
    
    if not pairs:
        print("No RFP/DAPI pairs found!")
        return False
    
    print(f"Found {len(pairs)} pairs to test")
    
    # Test individual pair processing
    detector = BlobDetector(debug=False)
    
    success_count = 0
    for i, (rfp_path, dapi_path) in enumerate(pairs):
        print(f"\nProcessing pair {i+1}: {rfp_path.name}")
        
        try:
            # Process without visualization
            ws_labels = detector.process_rfp_dapi_pair(rfp_path, dapi_path)
            
            if ws_labels.size > 0:
                print(f"  ✓ Success! Generated labels array shape: {ws_labels.shape}")
                print(f"  ✓ Number of unique labels: {len(np.unique(ws_labels))}")
                print(f"  ✓ Max label value: {np.max(ws_labels)}")
                success_count += 1
            else:
                print(f"  ✗ Failed - empty result")
                
        except Exception as e:
            print(f"  ✗ Error processing pair: {e}")
    
    print(f"\nResults: {success_count}/{len(pairs)} pairs processed successfully")
    
    # Test RFPDAPIProcessor
    print("\nTesting RFPDAPIProcessor...")
    try:
        processor = RFPDAPIProcessor(project_root, debug=False)
        found_pairs = processor.find_image_pairs(max_pairs=2)
        print(f"  ✓ RFPDAPIProcessor found {len(found_pairs)} pairs")
        
        if found_pairs:
            rfp_path, dapi_path = found_pairs[0]
            ws_labels = processor.process_single_pair(
                rfp_path, dapi_path, show_visualization=False
            )
            if ws_labels.size > 0:
                print(f"  ✓ Single pair processing successful: {ws_labels.shape}")
            else:
                print("  ✗ Single pair processing failed")
        
    except Exception as e:
        print(f"  ✗ RFPDAPIProcessor error: {e}")
    
    return success_count > 0

if __name__ == "__main__":
    success = test_watershed_processing()
    if success:
        print("\n🎉 Watershed processing tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Watershed processing tests FAILED!")
        sys.exit(1)
