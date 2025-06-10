from __future__ import annotations
import numpy as np
from img.img_handeler import ImgHandler
from np_labels.label_handeler import LabelHandeler as BlobHandeler

class BlobDetector:
    def __init__(self, channel: int = 0, debug = False): # put the parameters that will be needed for blob detection across all the images #should probobly have some code in here that handles deciding what sourt of detection should be used depending on the image staining
        self.channel = channel # Default channel for blob detection
        self.debug = debug  # Debug mode to save intermediate results
    
    def detect(self, image: np.ndarray, low_thresh: int=10, high_thresh:int=150) -> "BlobDetector":
        """
        Detect blobs in the provided image.
        
        Parameters:
            image (np.ndarray): The input image in which to detect blobs.
        
        Returns:
            BlobDetector: The instance with detected blobs.
        """
        # Placeholder for blob detection logic
        # This should be replaced with actual blob detection code
        image = ImgHandler.InconvenientObjectRemover(image).RemoveScaleBar(intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0)
        positive_mask = ImgHandler.transform.threshold.chromaticity(image, channel=self.channel, threshold=0.5)
        blob_labels = BlobHandeler(
            labels = ImgHandler.segmentation.region_based_segmentation.water_shed_segmentation(
                    ImgHandler.masker(
                        ImgHandler.EnhanceContrast.CLAHE(ImgHandler.transform.gray_scale.single_channel(image, channel=self.channel))).otsu().morph_cleanup().cleaned_mask,
                    low_thresh=low_thresh, high_thresh=high_thresh
                ),
                positive_mask=positive_mask
            ).flood_fill()#.merge_labels(size_factor=20)
        if self.debug is True:
            blob_labels.save_expanded_labels("component development/mergeLogic/unmerged_labels", save_csv=True)
        return blob_labels.merged_labels
        
if __name__ == "__main__":
    import tifffile
    import matplotlib.pyplot as plt

    # Example usage
    image = tifffile.imread('/home/user/documents/PiereLab/lysozyme/DemoData/ClearedForProcessing/G2ER-RFP2 40x-4.tif')
    detector = BlobDetector(channel=0, debug=True)
    blobs = detector.detect(image, low_thresh=30, high_thresh=150)

    # Display the detected blobs
    #plt.imshow(blobs, cmap='gray')
    plt.title('Detected Blobs')
    plt.show()