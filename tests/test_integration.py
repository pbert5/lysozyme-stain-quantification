"""
Integration test for the complete pipeline.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import tifffile
import sys

# Import test configuration
from . import *

# Test the main run script functionality
def test_main_imports():
    """Test that main components can be imported successfully."""
    try:
        from pipeline.bulk_processor import BulkProcessor
        from processing.individual_processor import IndividualProcessor
        from utils.file_utils import find_image_pairs, validate_directories
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directory structure
        self.img_dir = self.temp_path / 'images'
        self.results_dir = self.temp_path / 'results'
        self.img_dir.mkdir()
        
        # Create test image pairs
        self._create_test_image_pairs()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image_pairs(self):
        """Create synthetic test image pairs."""
        for i in range(3):
            # Create synthetic images with some structure
            red_img = self._create_synthetic_image_with_blobs(seed=i*2)
            blue_img = self._create_synthetic_image_with_blobs(seed=i*2+1, intensity=100)
            
            # Save as TIFF files
            red_path = self.img_dir / f'test_{i:02d}_RFP.tif'
            blue_path = self.img_dir / f'test_{i:02d}_DAPI.tif'
            
            tifffile.imwrite(red_path, red_img)
            tifffile.imwrite(blue_path, blue_img)
    
    def _create_synthetic_image_with_blobs(self, size=(150, 150), seed=42, intensity=200):
        """Create a synthetic image with blob-like structures."""
        np.random.seed(seed)
        
        # Start with noise
        img = np.random.normal(30, 10, size).astype(np.float32)
        
        # Add some blob-like structures
        centers = [(40, 40), (80, 100), (120, 60), (60, 120)]
        for center_x, center_y in centers:
            y, x = np.ogrid[:size[0], :size[1]]
            mask = ((x - center_x)**2 + (y - center_y)**2) < (15 + np.random.randint(5))**2
            img[mask] = intensity + np.random.normal(0, 20)
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def test_import_functionality(self):
        """Test that all main components can be imported."""
        success = test_main_imports()
        self.assertTrue(success, "Failed to import main pipeline components")
    
    def test_file_discovery(self):
        """Test that image pairs are discovered correctly."""
        try:
            from utils.file_utils import find_image_pairs
            
            pairs = find_image_pairs(self.img_dir, 'RFP', 'DAPI')
            self.assertEqual(len(pairs), 3)
            
            # Check that all pairs have both red and blue files
            for red_path, blue_path in pairs:
                self.assertTrue(red_path.exists())
                self.assertTrue(blue_path.exists())
                self.assertIn('RFP', red_path.name)
                self.assertIn('DAPI', blue_path.name)
        except ImportError:
            self.skipTest("Cannot import file_utils")
    
    def test_individual_processing(self):
        """Test processing of individual image pairs."""
        try:
            from processing.individual_processor import IndividualProcessor
            from utils.file_utils import find_image_pairs
            
            pairs = find_image_pairs(self.img_dir, 'RFP', 'DAPI')
            if not pairs:
                self.skipTest("No image pairs found")
            
            pixel_dims = {'default': 0.4476}
            processor = IndividualProcessor(pixel_dims, debug=False)
            
            red_path, blue_path = pairs[0]
            merged_labels, summary, _ = processor.process_pair(red_path, blue_path)
            
            # Check that processing succeeded
            self.assertIsNotNone(merged_labels)
            self.assertIsInstance(summary, np.ndarray)
            
        except ImportError:
            self.skipTest("Cannot import processing components")
    
    def test_bulk_processing_basic(self):
        """Test basic bulk processing functionality."""
        try:
            from pipeline.bulk_processor import BulkProcessor
            from utils.file_utils import find_image_pairs
            
            pairs = find_image_pairs(self.img_dir, 'RFP', 'DAPI')
            if not pairs:
                self.skipTest("No image pairs found")
            
            pixel_dims = {'default': 0.4476}
            
            # Test with just one pair to keep it fast
            processor = BulkProcessor(
                output_dir=self.results_dir,
                pixel_dims=pixel_dims,
                debug=False
            )
            
            results = processor.process_pairs(pairs[:1])
            
            # Check that processing succeeded
            self.assertGreater(len(results), 0)
            
            # Check that output directories were created
            self.assertTrue(self.results_dir.exists())
            
        except ImportError:
            self.skipTest("Cannot import bulk processor")


if __name__ == '__main__':
    # First check if imports work
    if not test_main_imports():
        print("ERROR: Cannot import main pipeline components.")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        sys.exit(1)
    
    unittest.main()
