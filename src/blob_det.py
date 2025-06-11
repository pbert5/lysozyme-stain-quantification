from __future__ import annotations
from pathlib import Path
import numpy as np
from img.img_handeler import ImgHandler
import tifffile
from np_labels.label_handeler import LabelHandeler as BlobHandeler
from skimage.color import label2rgb
import matplotlib.pyplot as plt

class BlobDetector:
    def __init__(self, channel: int = 0, debug = False, red_image = None): # put the parameters that will be needed for blob detection across all the images #should probobly have some code in here that handles deciding what sourt of detection should be used depending on the image staining
        self.channel = channel # Default channel for blob detection
        self.debug = debug  # Debug mode to save intermediate results
        if red_image is not None:
            if isinstance(red_image, str):
                self.red_image = tifffile.load_image(red_image)
            elif isinstance(red_image, np.ndarray):
                self.red_image = red_image
            else:
                raise ValueError("red_image must be a file path or a numpy array.")
    
    def detect(self, image: np.ndarray, low_thresh: int=10, high_thresh:int=150, path: str = None    ) -> "BlobDetector":
        """
        Detect blobs in the provided image.
        
        Parameters:
            image (np.ndarray): The input image in which to detect blobs.
        
        Returns:
            BlobDetector: The instance with detected blobs.
        """
        if path is not None:
            self.path = Path(path)
        else:
            self.path = Path("blob_detection_output.tif")
        # Placeholder for blob detection logic
        # This should be replaced with actual blob detection code
        image = ImgHandler.InconvenientObjectRemover(image).RemoveScaleBar(intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0)
        positive_mask = ImgHandler.transform.threshold.chromaticity(image, channel=self.channel, threshold=0.5)
        self.expanded_labels = BlobHandeler(
                    labels = ImgHandler.segmentation.region_based_segmentation.water_shed_segmentation(
                            ImgHandler.masker(
                                ImgHandler.EnhanceContrast.CLAHE(ImgHandler.transform.gray_scale.single_channel(image, channel=self.channel))).otsu().morph_cleanup().cleaned_mask,
                            low_thresh=low_thresh, high_thresh=high_thresh
                        ),
                        positive_mask=positive_mask
                    ).flood_fill().expanded_labels
        self.swallowed_labels = BlobHandeler.MergePipeline(
                label_img = self.expanded_labels
        ).run().merged_label_array
        # if self.debug is True:
        #     self.swallowed_labels.save_expanded_labels("lysozyme-stain-quantification/component development/mergeLogic/unmerged_labels", save_csv=True)
        self.blobs = self.swallowed_labels # for a simple easy access to the labels
        return self.swallowed_labels
    def save_outputs(self, out_dir: str | Path, simple = False) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        #plt.imsave(out_dir / f"{self.path.stem}_red.png", self.red_image, cmap="gray")
        if simple:
            comp = self.swallowed_labels
        else:
            comp = np.concatenate([
                self.red_image if self.red_image.ndim == 3 else np.stack([self.red_image]*3, -1) if self.red_image is not None else None,
                self.expanded_labels ,
                self.swallowed_labels], axis=1)

        plt.imsave(out_dir / f"{self.path.stem}_cf.png", comp.astype(np.uint8))
        
if __name__ == "__main__":
    import tifffile
    import matplotlib.pyplot as plt

    # Example usage
    image = tifffile.imread('/home/user/documents/PiereLab/lysozyme/DemoData/ClearedForProcessing/G2EL-RFP2 40x-4.tif')
    detector = BlobDetector(channel=0, debug=False)
    blobs = detector.detect(image, low_thresh=30, high_thresh=150)
    detector.save_outputs(out_dir='/home/user/documents/PiereLab/lysozyme/DemoData/Results', simple=False)

 