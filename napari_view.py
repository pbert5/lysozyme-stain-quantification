import napari
from image_tools import BlobDetector
import numpy as np

# Initialize BlobDetector
detector = BlobDetector("DemoData/G2-ABX/G2-ABX/G2EL-RFP2 40x-4.tif")
detector.remove_scale_bar()
detector.select_red_channel()
detector.enhance_contrast()
detector.threshold()
detector.morph_cleanup()
blobs = detector.extract_blobs()
blob_overlay = detector.draw_blobs()

# Start Napari viewer
viewer = napari.Viewer()

# Add raw image
viewer.add_image(detector.raw_image, name="Raw Image", colormap="gray")

# Add scale-bar-masked image
viewer.add_image(detector.current_image, name="Masked & Processed", colormap="gray")

# Add binary mask
viewer.add_labels(detector.binary_mask, name="Binary Mask")

# Add cleaned mask
viewer.add_labels(detector.cleaned_mask, name="Cleaned Mask")

# Add overlay as an image layer
viewer.add_image(blob_overlay, name="Blob Overlay")

napari.run()
