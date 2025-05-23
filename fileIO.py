"""
TIF IO Loader Module
====================

This module provides two main classes to assist with locating and loading .tif/.tiff images:

Classes:
--------
1. TifImage(path: str)
   - Loads a .tif image from disk.
   - Attributes:
       - path: file path to the image
       - metadata: dictionary of parsed metadata and tags
       - image_data: image content as a NumPy array
   - Internal Methods:
       - _load_image(): handles parsing TIFF structure and populating metadata

2. ImageFinder(path: str)
   - Takes either a folder or single image path.
   - If folder: recursively searches for .tif/.tiff images.
   - If file: loads only that image.
   - Methods:
       - _find_tif_files(directory: str) -> List[str]: helper to find tif paths
       - retrieve_images() -> List[TifImage]: returns list of TifImage instances

Usage Example:
--------------
    finder = ImageFinder("/path/to/folder_or_file")
    images = finder.retrieve_images()

    for img in images:
        print(img.path)
        print(img.metadata)
        print(img.image_data.shape)

"""

import os
from typing import List, Union
import numpy as np
from tifffile import TiffFile

class TifImage:
    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.image_data = None
        self._load_image()

    def _load_image(self):
        with TiffFile(self.path) as tif:
            page = tif.pages[0]  # assuming we only want the first page for now
            self.image_data = page.asarray()

            # Gather useful metadata
            self.metadata['shape'] = page.shape
            self.metadata['dtype'] = str(page.dtype)
            self.metadata['axes'] = page.axes
            self.metadata['description'] = page.description
            self.metadata['tags'] = {tag.name: tag.value for tag in page.tags.values()}
            self.metadata['is_bigtiff'] = tif.is_bigtiff
            self.metadata['is_ome'] = tif.is_ome
            self.metadata['is_tiled'] = page.is_tiled

class ImageFinder:
    def __init__(self, path: str):
        self.path = path

    def _find_tif_files(self, directory: str) -> List[str]:
        tif_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                    tif_files.append(os.path.join(root, file))
        return tif_files

    def retrieve_images(self) -> List[TifImage]:
        tif_paths = []
        if os.path.isdir(self.path):
            tif_paths = self._find_tif_files(self.path)
        elif os.path.isfile(self.path) and (self.path.lower().endswith('.tif') or self.path.lower().endswith('.tiff')):
            tif_paths = [self.path]

        return [TifImage(p) for p in tif_paths]

# --- Test snippet ---
if __name__ == "__main__":
    # Replace this with your test directory or file
    test_path = f"DemoData/G2-ABX"
    finder = ImageFinder(test_path)
    images = finder.retrieve_images()

    for i, img in enumerate(images):
        print(f"Image {i+1}: {img.path}")
        print(f"  Shape: {img.metadata['shape']}")
        print(f"  Dtype: {img.metadata['dtype']}")
        print(f"  Axes: {img.metadata['axes']}")
        print(f"  Description: {img.metadata['description'][:100]}...")
        print(f"  Tags: {list(img.metadata['tags'].keys())}\n")
