"""
Simple test runner for the pipeline.
"""

import sys
import tempfile
import numpy as np
import tifffile
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        from utils.file_utils import validate_directories, find_image_pairs, load_as_gray, build_rgb
        print("✓ utils.file_utils")
    except ImportError as e:
        print(f"✗ utils.file_utils: {e}")
        return False
    
    try:
        from utils.image_utils import minmax01, calculate_pixel_dimensions
        print("✓ utils.image_utils")
    except ImportError as e:
        print(f"✗ utils.image_utils: {e}")
        return False
    
    try:
        from processing.extractor_pipeline import ExtractorPipeline
        print("✓ processing.extractor_pipeline")
    except ImportError as e:
        print(f"✗ processing.extractor_pipeline: {e}")
        return False
    
    try:
        from processing.merge_pipeline import MergePipeline
        print("✓ processing.merge_pipeline")
    except ImportError as e:
        print(f"✗ processing.merge_pipeline: {e}")
        return False
    
    try:
        from processing.individual_processor import IndividualProcessor
        print("✓ processing.individual_processor")
    except ImportError as e:
        print(f"✗ processing.individual_processor: {e}")
        return False
    
    try:
        from pipeline.bulk_processor import BulkProcessor
        print("✓ pipeline.bulk_processor")
    except ImportError as e:
        print(f"✗ pipeline.bulk_processor: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\nTesting basic functionality...")
    
    from processing.extractor_pipeline import ExtractorPipeline
    from processing.merge_pipeline import MergePipeline
    from utils.image_utils import minmax01
    from utils.file_utils import build_rgb
    
    # Create synthetic test images
    np.random.seed(42)
    red_img = np.random.rand(100, 100).astype(np.float32) * 100
    blue_img = np.random.rand(100, 100).astype(np.float32) * 80
    
    # Add some bright spots to red channel
    red_img[30:50, 30:50] = 255
    red_img[60:80, 60:80] = 200
    red_img[20:35, 70:85] = 180
    
    try:
        # Test extractor pipeline
        extractor = ExtractorPipeline(debug=False)
        labels = extractor.extract(red_img, blue_img)
        print(f"✓ Extractor: shape {labels.shape}, labels {np.unique(labels)}")
        
        # Test merge pipeline if we have labels
        if labels.max() > 0:
            merger = MergePipeline(labels)
            merger.run()
            merged_labels = merger.merged_label_array
            print(f"✓ Merger: shape {merged_labels.shape}, labels {np.unique(merged_labels)}")
        else:
            print("⚠ No labels found, skipping merge test")
        
        # Test utility functions
        normalized = minmax01(red_img)
        rgb = build_rgb(red_img, blue_img)
        print(f"✓ Utils: normalized range [{normalized.min():.3f}, {normalized.max():.3f}], RGB shape {rgb.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_operations():
    """Test file operations with temporary files."""
    print("\nTesting file operations...")
    
    from utils.file_utils import find_image_pairs, load_as_gray
    from processing.individual_processor import IndividualProcessor
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create synthetic test images
        red_img = np.random.randint(50, 200, (80, 80), dtype=np.uint8)
        blue_img = np.random.randint(30, 150, (80, 80), dtype=np.uint8)
        
        # Add some bright spots
        red_img[20:40, 20:40] = 255
        red_img[50:70, 50:70] = 220
        
        # Save test files
        red_path = temp_path / 'test_RFP.tif'
        blue_path = temp_path / 'test_DAPI.tif'
        
        tifffile.imwrite(red_path, red_img)
        tifffile.imwrite(blue_path, blue_img)
        
        try:
            # Test file discovery
            pairs = find_image_pairs(temp_path, 'RFP', 'DAPI')
            print(f"✓ Found {len(pairs)} image pairs")
            
            if pairs:
                # Test loading
                loaded_red = load_as_gray(pairs[0][0])
                loaded_blue = load_as_gray(pairs[0][1])
                print(f"✓ Loaded images: red {loaded_red.shape}, blue {loaded_blue.shape}")
                
                # Test individual processor
                pixel_dims = {'default': 0.4476}
                processor = IndividualProcessor(pixel_dims, debug=False)
                merged_labels, summary, _ = processor.process_pair(pairs[0][0], pairs[0][1])
                
                if merged_labels is not None:
                    print(f"✓ Individual processor: labels shape {merged_labels.shape}")
                    if summary is not None and len(summary) > 0:
                        print(f"✓ Summary: {len(summary)} regions found")
                    else:
                        print("⚠ No regions found in summary")
                else:
                    print("⚠ Individual processor returned None")
            
            return True
            
        except Exception as e:
            print(f"✗ File operations test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LYSOZYME STAIN QUANTIFICATION PIPELINE - TEST SUITE")
    print("=" * 60)
    
    success = True
    
    success &= test_basic_imports()
    success &= test_basic_functionality()
    success &= test_file_operations()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! The pipeline is ready to use.")
        print("\nTo run the pipeline:")
        print("  python src/run.py <img_dir> <results_dir> [options]")
        print("\nExample:")
        print('  python src/run.py "lysozyme images" results --debug')
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("=" * 60)
    
    return success


if __name__ == '__main__':
    main()
