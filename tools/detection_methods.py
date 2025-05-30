import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, segmentation, color, measure
from scipy import ndimage as ndi

class DetectionMethods:
    class region_based_segmentation:
        def __init__(self, image, low_thresh=10, high_thresh=150):
            self.image = image
            self.low_thresh = low_thresh
            self.high_thresh = high_thresh

            self.elevation_map = None
            self.markers = None
            self.segmentation = None
            self.labeled = None
            self.label_overlay = None

        def detect_blobs(self):
            # 1️⃣ Compute elevation map
            self.elevation_map = filters.sobel(self.image)

            # 2️⃣ Generate markers
            self.markers = np.zeros_like(self.image, dtype=np.uint8)
            self.markers[self.image < self.low_thresh] = 1
            self.markers[self.image > self.high_thresh] = 2

            # 3️⃣ Watershed segmentation
            self.segmentation = segmentation.watershed(self.elevation_map, self.markers)
            self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

            # 4️⃣ Label blobs
            self.labeled, _ = ndi.label(self.segmentation)

            # 5️⃣ Create label overlay
            self.label_overlay = color.label2rgb(self.labeled, image=self.image, bg_label=0)
            return self
        def save_results(self, save_path='segmentation_results.png'):
            if self.elevation_map is None:
                raise RuntimeError("Run detect_blobs() first!")

            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            
            # Original image
            axes[0, 0].imshow(self.image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Elevation map
            axes[0, 1].imshow(self.elevation_map, cmap='gray')
            axes[0, 1].set_title('Elevation Map')
            axes[0, 1].axis('off')

            # Markers
            axes[0, 2].imshow(self.markers, cmap='nipy_spectral')
            axes[0, 2].set_title('Markers')
            axes[0, 2].axis('off')

            # Segmentation
            axes[1, 0].imshow(self.segmentation, cmap='gray')
            axes[1, 0].set_title('Segmentation Mask')
            axes[1, 0].axis('off')

            # Labeled blobs
            axes[1, 1].imshow(self.label_overlay)
            axes[1, 1].set_title('Label Overlay')
            axes[1, 1].axis('off')

            # Segmentation overlay on original
            axes[1, 2].imshow(self.image, cmap='gray')
            axes[1, 2].contour(self.segmentation, [0.5], linewidths=1.2, colors='y')
            axes[1, 2].set_title('Segmentation Overlay')
            axes[1, 2].axis('off')

            fig.tight_layout()
            plt.savefig(save_path)
            

        