import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage import img_as_ubyte
import tifffile

# 1️⃣ Load image
def load_image(path):
    img = tifffile.imread(path)
    return img

# 2️⃣ Extract red channel (or passthrough grayscale)
def get_red_channel(img):
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, 0]
    else:
        return img.copy()

# 3️⃣ Enhance contrast using CLAHE
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_8bit = img_as_ubyte(img / img.max())  # Ensure 8-bit for CLAHE
    return clahe.apply(img_8bit)

# 4️⃣ Threshold using Otsu
def otsu_threshold(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def remove_scale_bar(img, intensity_threshold=200, min_area=500, aspect_ratio_thresh=5.0):
    """
    Find and mask out scale bar-like objects from an image.
    
    Parameters:
        img: 2D numpy array (grayscale image)
        intensity_threshold: pixel value threshold for bar detection (default=200 for white bars)
        min_area: minimum area of a contour to consider (default=500)
        aspect_ratio_thresh: minimum aspect ratio (w/h or h/w) to consider a scale bar (default=5.0)
        
    Returns:
        mask: binary mask (1 = keep, 0 = remove)
    """
    # Threshold for bright objects
    _, binary = cv2.threshold(img, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask
    mask = np.ones_like(img, dtype=np.uint8)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = max(w / h, h / w)
        
        if area > min_area and aspect_ratio > aspect_ratio_thresh:
            # Remove the bar by setting mask to 0 in this region
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
    
    return mask

# 5️⃣ Morphological cleanup
def morph_cleanup(binary_img, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

# 6️⃣ Label and extract blobs
def get_blob_props(binary_img, top_n=5):
    label_img = label(binary_img)
    props = regionprops(label_img)
    sorted_blobs = sorted(props, key=lambda x: x.area, reverse=True)[:top_n]
    return label_img, sorted_blobs

# 7️⃣ Draw blobs on an image
def draw_blobs(base_img, blobs):
    output = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if base_img.ndim == 2 else base_img.copy()
    for i, blob in enumerate(blobs):
        minr, minc, maxr, maxc = blob.bbox
        cv2.rectangle(output, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        cy, cx = blob.centroid
        cv2.putText(output, f"{i+1}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return output
