# my_pkg/__init__.py

from __future__ import annotations

# stdlib
import os
import gc
import json
from pathlib import Path
from typing import List, Dict, Any

# third-party
import tifffile
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
import json
from shapely import geometry
from skimage import (
    filters,
    segmentation,
    color,
    measure,
    img_as_ubyte,
)
from skimage.segmentation import flood, watershed
from skimage.morphology   import dilation, footprint_rectangle
from skimage.color        import label2rgb

# expose your submodules and classes
from .detection_methods import DetectionMethods
from .ImagePrep          import image_prep
from .expand             import CompetitiveFlooding
from .blobdetector       import BlobDetector


__all__ = [
    "DetectionMethods",
    "image_prep",
    "CompetitiveFlooding",
    "BlobDetector",
    # if you want to export deps too:
    "os", "gc", "json", "Path",
    "np", "cv2", "tifffile", "img_as_ubyte",
    "color", "measure", "flood", "watershed",
    "dilation", "footprint_rectangle", "label2rgb",
    "scipy", "ndi",
    "filters", "segmentation",
    "np", "plt", "filters", "segmentation", "Dict", "Any", "List", "label2rgb", "geometry", "json"
]
