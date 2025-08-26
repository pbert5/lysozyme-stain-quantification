"""
Test utility functions.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import tifffile

# Import test configuration
from . import *

from utils.file_utils import (
    validate_directories, find_image_pairs, load_as_gray, 
    build_rgb, _find_matching_blue
)
from utils.image_utils import minmax01, calculate_pixel_dimensions


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_directories_existing(self):
        """Test validation of existing directories."""
        # Create a temporary results directory
        results_dir = self.temp_path / 'results'
        
        # Should pass: existing img_dir, creatable results_dir
        self.assertTrue(validate_directories(self.temp_path, results_dir))
        self.assertTrue(results_dir.exists())
    
    def test_validate_directories_nonexistent_img(self):
        """Test validation with non-existent image directory."""
        fake_img_dir = self.temp_path / 'nonexistent'
        results_dir = self.temp_path / 'results'
        
        self.assertFalse(validate_directories(fake_img_dir, results_dir))
    
    def test_find_matching_blue(self):
        """Test finding matching blue channel files."""
        # Create test files
        red_file = self.temp_path / 'test_RFP.tif'
        blue_file = self.temp_path / 'test_DAPI.tif'
        
        red_file.touch()
        blue_file.touch()
        
        # Test matching
        result = _find_matching_blue(red_file, 'RFP', 'DAPI')
        self.assertEqual(result, blue_file)
        
        # Test no match
        result = _find_matching_blue(red_file, 'RFP', 'NONEXISTENT')
        self.assertIsNone(result)
    
    def test_load_as_gray_synthetic(self):
        """Test loading synthetic grayscale images."""
        # Create a synthetic test image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_file = self.temp_path / 'test.tif'
        tifffile.imwrite(test_file, test_image)
        
        # Load and test
        loaded = load_as_gray(test_file)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, (100, 100))
    
    def test_build_rgb(self):
        """Test RGB image construction."""
        red = np.random.rand(50, 50).astype(np.float32) * 255
        blue = np.random.rand(50, 50).astype(np.float32) * 255
        
        rgb = build_rgb(red, blue)
        
        self.assertEqual(rgb.shape, (50, 50, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        
        # Check that red and blue channels are preserved
        self.assertTrue(np.all(rgb[..., 1] == 0))  # Green should be zero
    
    def test_find_image_pairs(self):
        """Test finding image pairs."""
        # Create test files
        (self.temp_path / 'image1_RFP.tif').touch()
        (self.temp_path / 'image1_DAPI.tif').touch()
        (self.temp_path / 'image2_RFP.tif').touch()
        (self.temp_path / 'image2_DAPI.tif').touch()
        (self.temp_path / 'lonely_RFP.tif').touch()  # No matching DAPI
        
        pairs = find_image_pairs(self.temp_path, 'RFP', 'DAPI')
        
        self.assertEqual(len(pairs), 2)
        pair_names = {p[0].stem.split('_')[0] for p in pairs}
        self.assertIn('image1', pair_names)
        self.assertIn('image2', pair_names)


class TestImageUtils(unittest.TestCase):
    """Test image utility functions."""
    
    def test_minmax01(self):
        """Test minmax normalization."""
        # Test normal case
        x = np.array([1, 2, 3, 4, 5])
        normalized = minmax01(x)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # Test constant array
        x_const = np.array([5, 5, 5, 5])
        normalized_const = minmax01(x_const)
        self.assertTrue(np.allclose(normalized_const, 0.0))
    
    def test_calculate_pixel_dimensions(self):
        """Test pixel dimension calculation."""
        pixel_dims = {
            'default': 0.4476,
            '40x-4': 0.2253,
            'high-res': 0.1
        }
        
        # Test default
        self.assertEqual(calculate_pixel_dimensions('random_file.tif', pixel_dims), 0.4476)
        
        # Test pattern match
        self.assertEqual(calculate_pixel_dimensions('test_40x-4_image.tif', pixel_dims), 0.2253)
        
        # Test another pattern
        self.assertEqual(calculate_pixel_dimensions('high-res_sample.tif', pixel_dims), 0.1)


if __name__ == '__main__':
    unittest.main()
