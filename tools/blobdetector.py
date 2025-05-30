from __future__ import annotations
from tools import *
class BlobDetector:
    def __init__(self, image_path: str | Path, debug: bool = False):
        self.path = Path(image_path)
        self.debug = debug
        self.raw_image: np.ndarray = tifffile.imread(str(self.path))
        # populated in detect()
        self.red_image: np.ndarray | None = None
        self.flood: CompetitiveFlooding | None = None
        
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
    def save_outputs(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_dir / f"{self.path.stem}_red.png", self.red_image, cmap="gray")

        comp = np.concatenate([
            self.red_image if self.red_image.ndim == 3 else np.stack([self.red_image]*3, -1),
            self.flood._overlay(self.flood.expanded_labels),
            self.flood._overlay(self.flood.swallowed_labels)], axis=1)

        plt.imsave(out_dir / f"{self.path.stem}_cf.png", comp.astype(np.uint8))

    # ─────────────────────────── memory free ────────────────────────────
    def dispose(self):
        del self.raw_image, self.red_image, self.flood
        gc.collect()