from __future__ import annotations
import numpy as np
from .img.img_handeler import ImgHandler
from .np_labels.label_handeler import LabelHandeler as BlobHandeler

class BlobDetector:
    def __init__(self, channel: int = 0): # put the parameters that will be needed for blob detection across all the images #should probobly have some code in here that handles deciding what sourt of detection should be used depending on the image staining
        self.channel = channel # Default channel for blob detection
    
    
    def detect(self, image: np.ndarray) -> "BlobDetector":
        """
        Detect blobs in the provided image.
        
        Parameters:
            image (np.ndarray): The input image in which to detect blobs.
        
        Returns:
            BlobDetector: The instance with detected blobs.
        """
        # Placeholder for blob detection logic
        # This should be replaced with actual blob detection code
        image = ImgHandler.inconvenient_object_remover(image).remove_scale_bar(intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0)
        positive_mask = ImgHandler.threshold.chromaticity(image, channel=self.channel, threshold=0.5)
        labels = BlobHandeler.merge_labels(
            BlobHandeler.flood_fill(
                labels = ImgHandler.segmentation.region_based_segmentation.water_shed_segmentation(
                    ImgHandler.gray_scale.single_channel(image, channel=self.channel),
                    ...
                ),
                positive_mask=positive_mask
        )
        )
        return labels
        self.blobs = [] 