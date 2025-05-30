# import packages
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
import tifffile
import matplotlib.pyplot as plt
from tools.ImagePrep import image_prep # Importing the image_prep class from tools.ImagePrep
from tools.detection_methods import DetectionMethods  # Importing the DetectionMethods class from tools.detection_methods

class BlobDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = tifffile.imread(image_path)
        
        self.bar_mask = None
        self.binary_mask = None
        self.cleaned_mask = None
        self.blob_props = None
    def save (self, img=None, title = None):
        if img is None:
            img = self.current_image
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        if title is not None:
            plt.title(title)
            plt.savefig(f"lysozyme-stain-quantification/results/{title}_result.png")
        else:
            plt.savefig(f"lysozyme-stain-quantification/results/result.png")
        
        
    def display(self, img=None):
        if img is None:
            img = self.current_image
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    def detectTheBlobs(self):
        self.current_image = self.raw_image.copy()
        self.current_image = image_prep.inconvenient_object_remover(self.current_image).remove_scale_bar()
        
        self.red_chr_image = image_prep.select_image_channels.red_chromaticity(self.current_image)
        self.red_image = image_prep.select_image_channels.red(self.current_image)
        self.red_chr_image_enhanced = image_prep.enhance_contrast.enhance_nonblack(self.red_chr_image)
        self.red_image_enhanced = image_prep.enhance_contrast.CLAHE(self.red_image)
        water_detector = DetectionMethods.region_based_segmentation(self.red_image, low_thresh=30, high_thresh=150)
        water_detector.detect_blobs()
        
if __name__ == "__main__":
    # Example usage
    image_path = "lysozyme-stain-quantification/DevNotebooks/G2EB-RFP 40x-4.tif"
    detector = BlobDetector(image_path)
    detector.detectTheBlobs()
    detector.save(detector.red_image_enhanced)
 