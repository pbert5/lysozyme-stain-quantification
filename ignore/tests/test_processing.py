"""
Test processing pipeline components.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import tifffile

# Import test configuration
from . import *

from processing.extractor_pipeline import ExtractorPipeline
from processing.merge_pipeline import MergePipeline
from processing.individual_processor import IndividualProcessor


class TestExtractorPipeline(unittest.TestCase):
    """Test the extractor pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic test images
        self.red_img = self._create_synthetic_red_image()
        self.blue_img = self._create_synthetic_blue_image()
    
    def _create_synthetic_red_image(self):
        """Create a synthetic red channel image with bright spots."""
        img = np.random.normal(50, 10, (200, 200)).astype(np.float32)
        
        # Add some bright spots
        img[50:70, 50:70] = 200
        img[120:140, 120:140] = 180
        img[80:100, 150:170] = 190
        
        return np.clip(img, 0, 255)
    
    def _create_synthetic_blue_image(self):
        """Create a synthetic blue channel image."""
        img = np.random.normal(30, 8, (200, 200)).astype(np.float32)
        
        # Add some dim regions corresponding to nuclei
        img[45:75, 45:75] = 80
        img[115:145, 115:145] = 70
        img[75:105, 145:175] = 75
        
        return np.clip(img, 0, 255)
    
    def test_extract_basic(self):
        """Test basic extraction functionality."""
        extractor = ExtractorPipeline(debug=False)
        labels = extractor.extract(self.red_img, self.blue_img)
        
        # Check output properties
        self.assertEqual(labels.shape, self.red_img.shape)
        self.assertTrue(labels.dtype in [np.int32, np.int64])
        self.assertGreaterEqual(labels.max(), 0)  # Should have at least background
    
    def test_extract_debug_mode(self):
        """Test extraction with debug mode enabled."""
        extractor = ExtractorPipeline(debug=True)
        labels = extractor.extract(self.red_img, self.blue_img)
        
        debug_info = extractor.get_debug_info()
        
        # Check that debug information is captured
        expected_keys = ['red_original', 'blue_original', 'diff_r', 'combined_labels']
        for key in expected_keys:
            self.assertIn(key, debug_info)
        
        # Check debug image shapes
        self.assertEqual(debug_info['red_original'].shape, self.red_img.shape)
        self.assertEqual(debug_info['blue_original'].shape, self.blue_img.shape)
    
    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises appropriate error."""
        extractor = ExtractorPipeline()
        
        red_wrong_shape = np.random.rand(100, 100).astype(np.float32)
        blue_wrong_shape = np.random.rand(150, 150).astype(np.float32)
        
        with self.assertRaises(ValueError):
            extractor.extract(red_wrong_shape, blue_wrong_shape)


class TestMergePipeline(unittest.TestCase):
    """Test the merge pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple labeled image with adjacent regions
        self.labels = self._create_test_labels()
    
    def _create_test_labels(self):
        """Create test labeled image with known structure."""
        labels = np.zeros((100, 100), dtype=int)
        
        # Create adjacent regions
        labels[20:40, 20:40] = 1
        labels[20:40, 40:60] = 2  # Adjacent to 1
        labels[40:60, 20:40] = 3  # Adjacent to 1
        labels[60:80, 60:80] = 4  # Isolated
        
        return labels
    
    def test_merge_basic(self):
        """Test basic merge functionality."""
        merger = MergePipeline(self.labels)
        merger.run()
        
        # Check that we get a valid merged result
        self.assertIsNotNone(merger.merged_label_array)
        self.assertEqual(merger.merged_label_array.shape, self.labels.shape)
    
    def test_compute_stats(self):
        """Test statistics computation."""
        merger = MergePipeline(self.labels)
        merger.compute_stats()
        
        # Check that properties are computed
        self.assertGreater(len(merger.props), 0)
        self.assertGreater(len(merger.areas), 0)
        self.assertGreater(len(merger.perims), 0)
        
        # Check that we have the expected number of regions
        expected_labels = set(np.unique(self.labels)) - {0}
        self.assertEqual(set(merger.props.keys()), expected_labels)
    
    def test_find_triangles(self):
        """Test triangle relationship detection."""
        merger = MergePipeline(self.labels)
        merger.compute_stats().find_triangles()
        
        # With our test setup, we should find some adjacency relationships
        self.assertIsInstance(merger.triangles, list)
        # The exact number depends on the implementation, but should be reasonable
        self.assertLessEqual(len(merger.triangles), 10)


class TestIndividualProcessor(unittest.TestCase):
    """Test the individual processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create synthetic test files
        self.red_path, self.blue_path = self._create_test_files()
        
        self.pixel_dims = {'default': 0.4476, '40x-4': 0.2253}
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_files(self):
        """Create synthetic test TIFF files."""
        # Create synthetic images
        red_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blue_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Add some bright spots to red
        red_img[30:50, 30:50] = 255
        red_img[60:80, 60:80] = 200
        
        # Save to temporary files
        red_path = self.temp_path / 'test_RFP.tif'
        blue_path = self.temp_path / 'test_DAPI.tif'
        
        tifffile.imwrite(red_path, red_img)
        tifffile.imwrite(blue_path, blue_img)
        
        return red_path, blue_path
    
    def test_process_pair_basic(self):
        """Test basic pair processing."""
        processor = IndividualProcessor(self.pixel_dims, debug=False)
        merged_labels, summary, debug_visuals = processor.process_pair(
            self.red_path, self.blue_path
        )
        
        # Check outputs
        self.assertIsNotNone(merged_labels)
        self.assertIsInstance(summary, np.ndarray)
        self.assertEqual(len(debug_visuals), 0)  # No debug visuals in non-debug mode
    
    def test_process_pair_debug(self):
        """Test pair processing with debug mode."""
        processor = IndividualProcessor(self.pixel_dims, debug=True)
        merged_labels, summary, debug_visuals = processor.process_pair(
            self.red_path, self.blue_path
        )
        
        # Check that debug visuals are generated
        self.assertGreater(len(debug_visuals), 0)
        expected_visuals = ['initial_overlay', 'merged_overlay', 'combined_overlay']
        for visual in expected_visuals:
            self.assertIn(visual, debug_visuals)
    
    def test_label_summary_format(self):
        """Test that label summary has correct format."""
        processor = IndividualProcessor(self.pixel_dims, debug=False)
        merged_labels, summary, _ = processor.process_pair(
            self.red_path, self.blue_path
        )
        
        if summary is not None and len(summary) > 0:
            # Check summary format: [id, pos_x, pos_y, area, red_sum, red_intensity]
            self.assertEqual(summary.shape[1], 6)
            
            # Check that all values are finite
            self.assertTrue(np.all(np.isfinite(summary)))
            
            # Check that areas are positive
            self.assertTrue(np.all(summary[:, 3] > 0))


if __name__ == '__main__':
    unittest.main()
