from __future__ import annotations
from tools import *

def labels_to_geojson(label_array, output_path, pixel_size=1.0, origin=(0, 0), label_prefix="ROI", expand_by=0.0):
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




class BlobDetector:
    def __init__(self, image_path: str | Path, debug: bool = False):
        self.path = Path(image_path)
        self.debug = debug
        self.raw_image: np.ndarray = tifffile.imread(str(self.path))
        # populated in detect()
        self.red_image: np.ndarray | None = None
        self.flood: CompetitiveFlooding | None = None
        self.water: Any = None
        
    def detect(self) -> "BlobDetector":
        cleaned = image_prep.inconvenient_object_remover(self.raw_image.copy()).remove_scale_bar()
        self.red_image = image_prep.select_image_channels.red(cleaned)
        red_chr = image_prep.select_image_channels.red_chromaticity(cleaned)
        red_chr_enh = image_prep.enhance_contrast.enhance_nonblack(red_chr)
        bin_mask = image_prep.masker(red_chr_enh).otsu().morph_cleanup().cleaned_mask

        # tight blob segmentation
        water = DetectionMethods.region_based_segmentation(self.red_image, low_thresh=30, high_thresh=150)
        water.detect_blobs()

        # competitive flooding
        self.flood = CompetitiveFlooding(water.labeled, bin_mask, self.red_image, debug=self.debug).run()
        return self
    def diagnostic(self):
        cleaned = image_prep.inconvenient_object_remover(self.raw_image.copy()).remove_scale_bar()
        self.red_image = image_prep.select_image_channels.red(cleaned)
        red_chr = image_prep.select_image_channels.red_chromaticity(cleaned)
        red_chr_enh = image_prep.enhance_contrast.enhance_nonblack(red_chr)
        bin_mask = image_prep.masker(red_chr_enh).otsu().morph_cleanup().cleaned_mask

        # tight blob segmentation
        self.water = DetectionMethods.region_based_segmentation(image=self.red_image, low_thresh=30, high_thresh=150)
        self.water.detect_blobs()#.visualize()

        # competitive flooding
        self.flood = CompetitiveFlooding(self.water.labeled, bin_mask, self.red_image, debug=self.debug).run()
        return self
        #return {"bin_mask": bin_mask, "water": water.labeled, "flood": self.flood}
    # ───────────────────────────── summary ──────────────────────────────
    def summary(self) -> Dict[str, Any]:
        # top_props now returns a dict with both the full expanded_labels
        # array and the list of top‐N blob dicts under 'props'
        result = self.flood.top_props(5)
        props = result["props"]
        # Optionally include JSON string for easy debugging
        props_json = json.dumps(props, indent=2)
        return {
            "image_path": str(self.path.resolve()),
            "expanded_labels": result["expanded_labels"],
            "top_blobs": result["props"],
            "top_blobs_json": props_json
        }

    # ───────────────────────────── saving ───────────────────────────────
    def save_outputs(self, out_dir: str | Path, simple = False) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        #plt.imsave(out_dir / f"{self.path.stem}_red.png", self.red_image, cmap="gray")
        if simple:
            comp = self.flood.swallowed_labels
        else:
            comp = np.concatenate([
                self.red_image if self.red_image.ndim == 3 else np.stack([self.red_image]*3, -1),
                self.flood._overlay(self.flood.expanded_labels),
                self.flood._overlay(self.flood.swallowed_labels)], axis=1)

        plt.imsave(out_dir / f"{self.path.stem}_cf.png", comp.astype(np.uint8))
        
    def export_rois_to_qupath(self, output_geojson=None, expand_by=0.0):
        results_dir = Path("lysozyme-stain-quantification/results/ROIs")
        results_dir.mkdir(parents=True, exist_ok=True)
        if output_geojson is None:
            # Create a new output filename based on the original image name
            output_filename = Path(self.path).stem + "_rois.geojson"
            output_geojson = results_dir / output_filename

        labels_to_geojson(
            self.flood.swallowed_labels
,
            output_path=output_geojson,
            pixel_size=1.0,  # Adjust if your image has a physical pixel size
            origin=(0, 0),
            expand_by=expand_by
        )

    # ─────────────────────────── memory free ────────────────────────────
    def dispose(self):
        del self.raw_image, self.red_image, self.flood
        if self.water is not None:
            del self.water
        gc.collect()