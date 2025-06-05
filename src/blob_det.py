from __future__ import annotations
import numpy as np
from .img.img_handeler import ImgHandler

class Blob_Detector:
    def __init__(self, channel: int = 0): # put the parameters that will be needed for blob detection across all the images #should probobly have some code in here that handles deciding what sourt of detection should be used depending on the image staining
        self.channel = channel # Default channel for blob detection
    
    
    def detect(self, image: np.ndarray) -> "Blob_Detector":
        """
        Detect blobs in the provided image.
        
        Parameters:
            image (np.ndarray): The input image in which to detect blobs.
        
        Returns:
            Blob_Detector: The instance with detected blobs.
        """
        # Placeholder for blob detection logic
        # This should be replaced with actual blob detection code
        image = ImgHandler.inconvenient_object_remover(image).remove_scale_bar(intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0)
        positive_mask = ImgHandler.threshold.chromaticity(image, self.channel=0, threshold=0.5)
        blobs = BlobHandeler.merge_blobs(
            BlobHandeler.flood_fill(
                blobs = Blob_handeler.blob_detector.watershed_segmentation(
                    ImgHandler.gradient.single_channel(image, channel=self.channel, threshold=0.5),
                    ...
                ),
                positive_mask=positive_mask
        )
        )
        return blobs
        self.blobs = [] 