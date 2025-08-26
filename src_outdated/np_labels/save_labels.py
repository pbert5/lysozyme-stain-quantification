import numpy as np
import cv2
import json
from shapely import geometry
def LabelsToGeoJSON(label_array, output_path, pixel_size=1.0, origin=(0, 0), label_prefix="ROI", expand_by=0.0):
    """
    Converts a labeled numpy array into a GeoJSON file compatible with QuPath,
    with optional expansion of ROI borders.

    Args:
        label_array (np.ndarray): Labeled array where each integer >0 is a separate ROI.
        output_path (str): Path to save the GeoJSON file.
        pixel_size (float): Pixel scaling factor (default=1.0 for 1:1 mapping).
        origin (tuple): (x0, y0) origin offset in real-world units.
        label_prefix (str): Prefix for label names.
        expand_by (float): Amount to expand each ROI outward (in same units as pixel_size).

    Returns:
        None
    """
    features = []
    for label in np.unique(label_array):
        if label == 0:
            continue  # Skip background

        mask = (label_array == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze().astype(float)
            if contour.ndim != 2:
                continue  # Skip invalid contours

            # Scale and translate coordinates
            contour[:, 0] = contour[:, 0] * pixel_size + origin[0]
            contour[:, 1] = contour[:, 1] * pixel_size + origin[1]

            poly = geometry.Polygon(contour)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersections

            if expand_by > 0.0:
                poly = poly.buffer(expand_by)

            features.append({
                "type": "Feature",
                "properties": {"name": f"{label_prefix}_{label}"},
                "geometry": geometry.mapping(poly)
            })

    geojson = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"[QuPath Exporter] Saved {len(features)} ROIs to {output_path}")
    
