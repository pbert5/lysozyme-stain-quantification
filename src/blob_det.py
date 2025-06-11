from __future__ import annotations
from pathlib import Path
import numpy as np
from img.img_handeler import ImgHandler
import tifffile
from np_labels.label_handeler import LabelHandeler as BlobHandeler
from skimage.color import label2rgb
import matplotlib.pyplot as plt

class BlobDetector:
    """
    BlobDetector

    A utility class for detecting and processing “blobs” (connected regions of interest)
    in microscopy or histology images. Supports a configurable detection pipeline,
    two-stage merging of segmented regions, and easy saving of results.

    Usage:
        # 1. Initialize with an optional reference image (e.g., raw red channel)
        detector = BlobDetector(channel=0, debug=True, red_image="path/to/image.tif")

        # 2. Run detection on a new image array (or file path) with threshold parameters
        merged_labels = detector.detect(
            image_array,
            low_thresh=10,
            high_thresh=150,
            path="output_basename"
        )

        # 3. Save outputs to disk
        #    - simple=True saves only the final merged labels
        #    - simple=False concatenates raw/red, expanded, and merged label overlays
        detector.save_outputs(out_dir="results_folder", simple=False)

    Class Initialization:
        __init__(channel: int = 0,
                debug: bool = False,
                red_image: Union[str, np.ndarray, None] = None)
            - channel: which color channel to use for threshold-based mask creation.
            - debug:   if True, intermediate artifacts (e.g., CSV of raw labels) can be saved.
            - red_image: either a file path or ndarray to use as the background/reference.

    Key Methods:
        detect(image: np.ndarray,
            low_thresh: int = 10,
            high_thresh: int = 150,
            path: Optional[str] = None) -> np.ndarray
            - image: raw image in which to detect blobs.
            - low_thresh / high_thresh: parameters controlling the watershed segmentation.
            - path: base name for saving output files.
            - Returns the 2nd-stage merged label array.

        save_outputs(out_dir: Union[str, Path], simple: bool = False) -> None
            - out_dir: directory to write results.
            - simple: if True, only final merged labels are saved; otherwise, a side-by-side
                    montage of raw/red, expanded_labels, and swallowed_labels is written.

    Internal Pipeline:
        1. Remove scale bar or other artifacts.
        2. Threshold by chromaticity → positive_mask.
        3. Watershed on CLAHE-enhanced grayscale → expanded_labels.
        4. Two-stage merge using MergePipeline:
            a. Stage 1: choose best neighborhood group per label.
            b. Stage 2: refine by compactness and distance metrics.
        → swallowed_labels.
        5. Expose `blobs` attribute for downstream analysis.

    Attributes After detect():
        - self.expanded_labels:  label image after initial segmentation.
        - self.swallowed_labels: label image after two-stage merging.
        - self.blobs:           alias for swallowed_labels.
        - self.path:            Path object for naming outputs.
    """

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
        else:
            self.red_image = None
    
    def detect(self, image: np.ndarray, segmentation_low_thresh: int=10, segmentation_high_thresh:int=150, path: str = None, scale_bar_intensity_threshold=240, scale_bar_min_area=500, scale_bar_aspect_ratio_thresh=4.0, positive_mask_threshold=0.5, singleton_penalty=10) -> "BlobDetector":
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
        image = ImgHandler.InconvenientObjectRemover(image).RemoveScaleBar(intensity_threshold=scale_bar_intensity_threshold, min_area=scale_bar_min_area, aspect_ratio_thresh=scale_bar_aspect_ratio_thresh)
        positive_mask = ImgHandler.transform.threshold.chromaticity(image, channel=self.channel, threshold=positive_mask_threshold)
        self.expanded_labels = BlobHandeler(
                    labels = ImgHandler.segmentation.region_based_segmentation.water_shed_segmentation(
                            ImgHandler.masker(
                                ImgHandler.EnhanceContrast.CLAHE(ImgHandler.transform.gray_scale.single_channel(image, channel=self.channel))).otsu().morph_cleanup().cleaned_mask,
                            low_thresh=segmentation_low_thresh, high_thresh=segmentation_high_thresh
                        ),
                        positive_mask=positive_mask
                    ).flood_fill().expanded_labels
        self.swallowed_labels = BlobHandeler.MergePipeline(
                label_img = self.expanded_labels,
                singleton_penalty=singleton_penalty
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
                
                self.expanded_labels,
                self.swallowed_labels], axis=1)

        plt.imsave(out_dir / f"{self.path.stem}_cf.png", label2rgb(comp.astype(np.uint8)))
        
if __name__ == "__main__":
    import tifffile
    import matplotlib.pyplot as plt

    # Example usage
    image = tifffile.imread('/home/user/documents/PiereLab/lysozyme/DemoData/ClearedForProcessing/G2EL-RFP2 40x-4.tif')
    detector = BlobDetector(channel=0, debug=False)
    blobs = detector.detect(image, low_thresh=30, high_thresh=150)
    detector.save_outputs(out_dir='/home/user/documents/PiereLab/lysozyme/DemoData/Results', simple=False)

 