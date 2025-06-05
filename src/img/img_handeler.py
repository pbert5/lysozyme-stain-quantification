from __future__ import annotations
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, segmentation, color, measure, img_as_ubyte
from scipy import ndimage as ndi
class ImgHandler:
    
    class InconvenientObjectRemover:
        def __init__(self, raw_image):
            self.raw_image = raw_image
        def remove_scale_bar(self, intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0):
            """
            Removes the scale bar from an image based on intensity threshold, minimum area, and aspect ratio.

            Args:
                intensity_threshold (int): Pixel intensity threshold for detecting the scale bar. Default is 240.
                min_area (int): Minimum area of a contour to be considered as a scale bar. Default is 500.
            # Ensure the input image is grayscale
            if gray.ndim != 2:
                raise ValueError("Input image must be a grayscale image for thresholding.")
            _, binary = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)

            Returns:
                np.ndarray: The image with the scale bar removed.
            """
            img = self.raw_image
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()
    
            _, binary = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            mask = np.ones_like(gray, dtype=np.uint8)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                aspect_ratio = max(w / h, h / w)
                if area > min_area and aspect_ratio > aspect_ratio_thresh:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
    
            self.bar_mask = mask
            if img.ndim == 3:
                self.current_image = cv2.merge([
                    cv2.bitwise_and(img[:, :, c], img[:, :, c], mask=mask)
                    for c in range(img.shape[2])
                ])
            else:
                self.current_image = cv2.bitwise_and(img, img, mask=mask)
            return self.current_image
    class transform:
        class threshold:
            @staticmethod
            
            def chromaticity(img, channel: int =0, threshold: float = 0.5):
                """
                Computes the chromaticity of a specified channel in an RGB image.
            
                Args:
                    img (np.ndarray): Input RGB image of shape (H, W, 3).
                    channel (int): Channel index to compute chromaticity for (0=R, 1=G, 2=B). Default is 0 (Red).
                    threshold (float): Threshold value for chromaticity computation (not used in this implementation). Default is 0.5.
            
                Returns:
                    np.ndarray: Grayscale image of the specified channel's chromaticity, scaled to 0-255.
                """
                if img.ndim != 3:
                    raise ValueError("Red chromaticity requires an RGB image.")
                R = img[:, :, 0].astype(float)
                G = img[:, :, 1].astype(float)
                B = img[:, :, 2].astype(float)
        
                epsilon = 1e-8  # Avoid division by zero
                sum_rgb = R + G + B + epsilon
                chroma = img[:, :, channel].astype(float) / sum_rgb
        
                return (chroma * 255).astype(np.uint8)  # Scale back to 0-255 for image-like display
        class gray_scale:
            @staticmethod
            def single_channel(img, channel=0):
                """
                Extracts a single channel from an RGB image.

                Args:
                    img (np.ndarray): RGB image of shape (H, W, 3)
                    channel (int): Channel index to extract (0=R, 1=G, 2=B)

                Returns:
                    np.ndarray: Grayscale image of selected channel (H, W)
                """
                if img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError("Expected an RGB image with 3 channels.")
                if channel not in (0, 1, 2):
                    raise ValueError("Channel must be 0 (R), 1 (G), or 2 (B).")
                
                return img[:, :, channel]
    class EnhanceContrast:
        @staticmethod
        def CLAHE(img, clip_limit=2.0, tile_grid_size=(8,8)):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            img_8bit = img_as_ubyte(img / img.max())  # Normalize to 8-bit
            return clahe.apply(img_8bit)
        @staticmethod
        def EnhanceNonblack(img, value=255):
            """
            Set all non-zero pixels in a grayscale image to a fixed value.
            Black (0) pixels remain black.

            Parameters:
            - img: Grayscale image as numpy array.
            - value: Brightness value to set (default 255).

            Returns:
            - Enhanced image.
            """
            # Ensure image is a numpy array
            img = np.asarray(img)

            # Create output array
            enhanced = np.zeros_like(img, dtype=np.uint8)

            # Set non-zero pixels to 'value'
            enhanced[img > 0] = value

            return enhanced
    class masker:            
        def __init__(self, image):
            self.current_image = image
        def otsu(self):
            _, binary = cv2.threshold(self.current_image, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.binary_mask = binary
            return self
        def morph_cleanup(self, kernel_size=5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            cleaned = cv2.morphologyEx(self.binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            self.cleaned_mask = cleaned
            return self
    class segmentation:
        
        class region_based_segmentation:

            @staticmethod
            def water_shed_segmentation(image, low_thresh: int=10, high_thresh:int=150):
                # 1️⃣ Compute elevation map
                elevation_map = filters.sobel(image)

                # 2️⃣ Generate markers
                markers = np.zeros_like(image, dtype=np.uint8)
                markers[image < low_thresh] = 1
                markers[image > high_thresh] = 2

                # 3️⃣ Watershed segmentation
                segmentation = segmentation.watershed(elevation_map, markers)
                segmentation = ndi.binary_fill_holes(segmentation - 1)

                # 4️⃣ Label blobs
                labeled, _ = ndi.label(segmentation)

                ## 5️⃣ Create label overlay
                #label_overlay = color.label2rgb(labeled, image=image, bg_label=0)
                return labeled
   