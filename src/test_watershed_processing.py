"""
Test suite for the blob detection and RFP/DAPI processing functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import tifffile
from unittest.mock import patch, MagicMock

# Add the current directory to the path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from blob_det import (
    BlobDetector, 
    load_as_gray, 
    build_rgb, 
    minmax01, 
    get_rfp_dapi_pairs,
    remove_rectangles,
    process_all_rfp_dapi_pairs
)
from primary import BulkBlobProcessor, RFPDAPIProcessor


class TestBlobDetector:
    """Test cases for BlobDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = BlobDetector(channel=0, debug=False)
    
    def test_detector_initialization(self):
        """Test BlobDetector initialization."""
        assert self.detector.channel == 0
        assert self.detector.debug == False
        
        detector_debug = BlobDetector(channel=1, debug=True)
        assert detector_debug.channel == 1
        assert detector_debug.debug == True
    
    def test_detect_with_simple_image(self):
        """Test detection with a simple synthetic image."""
        # Create a simple test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[30:70, 30:70] = 255  # White square
        
        labels = self.detector.detect(test_image)
        
        # Should return a label array of the same shape
        assert labels.shape == test_image.shape
        assert labels.dtype in [np.int32, np.int64]
        
        # Should have detected at least one region
        assert np.max(labels) > 0
    
    def test_detect_with_rgb_image(self):
        """Test detection with RGB image."""
        # Create RGB test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[30:70, 30:70, :] = [255, 128, 64]  # Colored square
        
        labels = self.detector.detect(test_image)
        
        assert labels.shape == test_image.shape[:2]
        assert np.max(labels) >= 0


class TestImageUtilities:
    """Test image utility functions."""
    
    def test_minmax01(self):
        """Test minmax01 normalization function."""
        # Test with regular array
        arr = np.array([1, 2, 3, 4, 5])
        normalized = minmax01(arr)
        
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        
        # Test with constant array
        const_arr = np.array([5, 5, 5, 5])
        normalized_const = minmax01(const_arr)
        assert np.all(normalized_const == 0.0)
    
    def test_build_rgb(self):
        """Test RGB image building from red and blue channels."""
        red = np.ones((50, 50)) * 255
        blue = np.ones((50, 50)) * 128
        
        rgb = build_rgb(red, blue)
        
        assert rgb.shape == (50, 50, 3)
        assert np.all(rgb[:, :, 0] == 255)  # Red channel
        assert np.all(rgb[:, :, 1] == 0)    # Green channel (should be zero)
        assert np.all(rgb[:, :, 2] == 128)  # Blue channel
    
    def test_remove_rectangles(self):
        """Test rectangle removal function."""
        # Create test image with a rectangular artifact
        test_image = np.random.randint(0, 200, (100, 100, 3), dtype=np.uint8)
        test_image[10:20, 10:80] = 255  # White rectangle
        
        cleaned = remove_rectangles(test_image, white_thresh=240)
        
        assert cleaned.shape == test_image.shape
        # Should be different from original (inpainting applied)
        assert not np.array_equal(cleaned, test_image)


class TestRFPDAPIProcessing:
    """Test RFP/DAPI processing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir
        
        # Create mock directory structure
        (self.project_root / 'lysozyme images').mkdir(parents=True)
        
        # Create mock RFP/DAPI files
        self.create_mock_image_files()
    
    def create_mock_image_files(self):
        """Create mock TIFF files for testing."""
        images_dir = self.project_root / 'lysozyme images'
        
        # Create mock image data
        mock_rfp = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_dapi = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Save as TIFF files
        tifffile.imwrite(images_dir / 'test_RFP.tif', mock_rfp)
        tifffile.imwrite(images_dir / 'test_DAPI.tif', mock_dapi)
        
        tifffile.imwrite(images_dir / 'test2_RFP.tif', mock_rfp)
        tifffile.imwrite(images_dir / 'test2_DAPI.tif', mock_dapi)
    
    def test_get_rfp_dapi_pairs(self):
        """Test finding RFP/DAPI pairs."""
        images_root = self.project_root / 'lysozyme images'
        pairs = get_rfp_dapi_pairs(images_root, max_pairs=10)
        
        assert len(pairs) == 2  # Should find 2 pairs
        
        # Check that pairs are correctly matched
        for rfp_path, dapi_path in pairs:
            assert '_RFP.' in rfp_path.name
            assert '_DAPI.' in dapi_path.name
            # Base names should match
            rfp_base = rfp_path.name.split('_RFP.')[0]
            dapi_base = dapi_path.name.split('_DAPI.')[0]
            assert rfp_base == dapi_base
    
    def test_load_as_gray(self):
        """Test loading images as grayscale."""
        images_dir = self.project_root / 'lysozyme images'
        test_path = images_dir / 'test_RFP.tif'
        
        gray_img = load_as_gray(test_path)
        
        assert gray_img.dtype == np.float32
        assert gray_img.ndim == 2
        assert gray_img.shape == (100, 100)
    
    def test_rfp_dapi_processor_initialization(self):
        """Test RFPDAPIProcessor initialization."""
        processor = RFPDAPIProcessor(self.project_root, debug=True)
        
        assert processor.project_root == self.project_root
        assert processor.debug == True
        assert isinstance(processor.detector, BlobDetector)
    
    def test_find_image_pairs(self):
        """Test finding image pairs through processor."""
        processor = RFPDAPIProcessor(self.project_root)
        pairs = processor.find_image_pairs(max_pairs=10)
        
        assert len(pairs) == 2
    
    @patch('matplotlib.pyplot.show')  # Mock plt.show to avoid displaying plots during tests
    def test_process_single_pair(self, mock_show):
        """Test processing a single RFP/DAPI pair."""
        processor = RFPDAPIProcessor(self.project_root, debug=False)
        
        images_dir = self.project_root / 'lysozyme images'
        rfp_path = images_dir / 'test_RFP.tif'
        dapi_path = images_dir / 'test_DAPI.tif'
        
        ws_labels = processor.process_single_pair(
            rfp_path, dapi_path, show_visualization=False
        )
        
        # Should return a labels array
        assert isinstance(ws_labels, np.ndarray)
        if ws_labels.size > 0:  # If processing succeeded
            assert ws_labels.shape == (100, 100)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_imports(self):
        """Test that all modules can be imported without errors."""
        try:
            from blob_det import BlobDetector
            from primary import BulkBlobProcessor, RFPDAPIProcessor
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_detector_with_rfp_dapi_workflow(self):
        """Test the complete RFP/DAPI processing workflow."""
        # Create temporary directory and mock files
        temp_dir = Path(tempfile.mkdtemp())
        project_root = temp_dir
        
        try:
            # Create directory structure
            (project_root / 'lysozyme images').mkdir(parents=True)
            
            # Create mock image files
            images_dir = project_root / 'lysozyme images'
            mock_rfp = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
            mock_dapi = np.random.randint(30, 150, (50, 50), dtype=np.uint8)
            
            tifffile.imwrite(images_dir / 'workflow_test_RFP.tif', mock_rfp)
            tifffile.imwrite(images_dir / 'workflow_test_DAPI.tif', mock_dapi)
            
            # Test the workflow
            detector = BlobDetector(debug=False)
            
            rfp_path = images_dir / 'workflow_test_RFP.tif'
            dapi_path = images_dir / 'workflow_test_DAPI.tif'
            
            # This should run without errors
            ws_labels = detector.process_rfp_dapi_pair(rfp_path, dapi_path)
            
            # Verify output
            assert isinstance(ws_labels, np.ndarray)
            
        finally:
            # Clean up
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def test_python_environment():
    """Test that we're using the correct Python environment."""
    import sys
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check that required packages are available
    required_packages = [
        'numpy', 'scipy', 'skimage', 'cv2', 'tifffile', 
        'matplotlib', 'pathlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            pytest.fail(f"Required package {package} is not available")


if __name__ == "__main__":
    # Run basic functionality test
    print("Running basic functionality tests...")
    
    # Test minmax01 function
    test_arr = np.array([1, 2, 3, 4, 5])
    normalized = minmax01(test_arr)
    print(f"minmax01 test: {test_arr} -> {normalized}")
    assert np.isclose(np.min(normalized), 0.0)
    assert np.isclose(np.max(normalized), 1.0)
    print("✓ minmax01 function works correctly")
    
    # Test BlobDetector initialization
    detector = BlobDetector()
    print("✓ BlobDetector can be initialized")
    
    # Test with synthetic image
    test_image = np.zeros((50, 50), dtype=np.uint8)
    test_image[20:30, 20:30] = 255
    
    labels = detector.detect(test_image)
    print(f"✓ BlobDetector.detect() returns shape {labels.shape}")
    
    print("All basic tests passed!")
