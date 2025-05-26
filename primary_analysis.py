from image_tools import BlobDetector
import matplotlib.pyplot as plt
import cv2

# Load and process
img = "DemoData/G2-ABX/G2-ABX/G2EL-RFP2 40x-4.tif"
detector = BlobDetector(img)

# Remove bars first
detector.remove_scale_bar()

# Use the bar-removed image for further processing
red = detector.get_red_channel()  # Will use detector.raw_image
enhanced = detector.enhance_contrast(red)
masked = cv2.bitwise_and(enhanced, enhanced, mask=detector.bar_mask)

binary = detector.otsu_threshold(masked)
cleaned = detector.morph_cleanup(binary)
blobs = detector.get_blobs(top_n=5)
overlay = detector.draw_blobs(base_img=enhanced)

# Save or display overlay
cv2.imwrite("output_overlay.png", overlay)
