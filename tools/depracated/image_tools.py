import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
import tifffile

class BlobDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = self.load_image()
        self.processed_image = None
        self.binary_mask = None
        self.cleaned_mask = None
        self.blob_props = None

    # 1️⃣ Load image
    def load_image(self):
        return tifffile.imread(self.image_path)

    # 2️⃣ Get red channel (if RGB)
    def get_red_channel(self):
        img = self.bar_removed if hasattr(self, 'bar_removed') else self.raw_image
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, 0]
        else:
            return img.copy()


    # 3️⃣ Apply CLAHE
    def enhance_contrast(self, img, clip_limit=2.0, tile_grid_size=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_8bit = img_as_ubyte(img / img.max())  # Normalize to 8-bit
        return clahe.apply(img_8bit)

    # 4️⃣ Threshold (Otsu)
    def otsu_threshold(self, img):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.binary_mask = binary
        return binary

    # Remove scale bar
    def remove_scale_bar(self, intensity_threshold=240, min_area=500, aspect_ratio_thresh=4.0):
        img = self.raw_image
        if img.ndim == 3 and img.shape[2] >= 3:
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

        # Store the masked image
        self.bar_mask = mask
        if img.ndim == 3:
            # Apply mask to each channel
            self.bar_removed = cv2.merge([
                cv2.bitwise_and(img[:, :, c], img[:, :, c], mask=mask)
                for c in range(img.shape[2])
            ])
        else:
            self.bar_removed = cv2.bitwise_and(img, img, mask=mask)

        return mask

    # 6️⃣ Morphological cleanup
    def morph_cleanup(self, binary_img, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        self.cleaned_mask = cleaned
        return cleaned

    # 7️⃣ Get blob properties
    def get_blobs(self, mask=None, top_n=5):
        if mask is None:
            mask = self.cleaned_mask
        label_img = label(mask)
        props = regionprops(label_img)
        self.blob_props = sorted(props, key=lambda x: x.area, reverse=True)[:top_n]
        return self.blob_props

    # 8️⃣ Draw blobs on image
    def draw_blobs(self, base_img=None, blobs=None):
        if base_img is None:
            base_img = self.raw_image
        if blobs is None:
            blobs = self.blob_props

        # If grayscale, convert to BGR
        output = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if base_img.ndim == 2 else base_img.copy()

        for i, blob in enumerate(blobs):
            minr, minc, maxr, maxc = blob.bbox
            cv2.rectangle(output, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cy, cx = blob.centroid
            cv2.putText(output, f"{i+1}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return output
