import cv2
from skimage import img_as_ubyte
import numpy as np

class image_prep:
    class inconvenient_object_remover:
        def __init__(self, raw_image):
            self.raw_image = raw_image
        def remove_scale_bar(self, intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0):
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
    class select_image_channels:
        @staticmethod
        def red(img):
            
            if img.ndim == 3:
                current_image = img[:, :, 0]
            else:
                current_image = img.copy()
            return current_image
        @staticmethod
        def red_chromaticity(img):
            if img.ndim != 3:
                raise ValueError("Red chromaticity requires an RGB image.")
            R = img[:, :, 0].astype(float)
            G = img[:, :, 1].astype(float)
            B = img[:, :, 2].astype(float)
    
            epsilon = 1e-8  # Avoid division by zero
            sum_rgb = R + G + B + epsilon
            red_chroma = R / sum_rgb
    
            return (red_chroma * 255).astype(np.uint8)  # Scale back to 0-255 for image-like display
    class enhance_contrast:
        @staticmethod
        def CLAHE(img, clip_limit=2.0, tile_grid_size=(8,8)):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            img_8bit = img_as_ubyte(img / img.max())  # Normalize to 8-bit
            return clahe.apply(img_8bit)
        @staticmethod
        def enhance_nonblack(img, value=255):
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
        
            
    