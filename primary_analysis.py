from image_tools import load_image, get_red_channel, apply_clahe, otsu_threshold, morph_cleanup, get_blob_props, draw_blobs, remove_scale_bar
import matplotlib.pyplot as plt
import cv2

# Load and process
img = load_image("DemoData/G2-ABX/G2-ABX/G2EL-RFP2 40x-4.tif")
red = get_red_channel(img)
clahe = apply_clahe(red)
# Remove scale bar
mask = remove_scale_bar(clahe, intensity_threshold=200)
clahe_masked = cv2.bitwise_and(clahe, clahe, mask=mask)
# continue
binary = otsu_threshold(clahe_masked)
cleaned = morph_cleanup(binary)
label_img, blobs = get_blob_props(cleaned)
overlay = draw_blobs(clahe, blobs)

# Show results
plt.imshow(overlay)
plt.show()
