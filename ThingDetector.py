"""
Competitive Flooding + Blob Detection bulk-processing toolkit.

This file now contains:
• CompetitiveFlooding  - fast, fluent pipeline (unchanged speed)
• BlobDetector         - wraps prep → segmentation → flooding
• BulkBlobProcessor    - iterate over many images, save outputs and
                          return summary dicts while freeing memory.
"""
from __future__ import annotations

# ─────────────────────────────────── stdlib ────────────────────────────────────
import os, gc, json
from pathlib import Path
from typing import List, Dict, Any

# ─────────────────────────────── third‑party libs ─────────────────────────────
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from skimage import measure, img_as_ubyte
from skimage.segmentation import watershed, flood
from skimage.morphology import dilation, square
from skimage.color import label2rgb

# ───────────────────────────────── user modules ───────────────────────────────
from tools.ImagePrep import image_prep              # noqa: F401
from tools.detection_methods import DetectionMethods # noqa: F401
from tools.expand import CompetitiveFlooding  # noqa: F401

# ═══════════════════════════════════ BlobDetector ═════════════════════════════
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

# ═══════════════════════════════ BulkBlobProcessor ════════════════════════════
class BulkBlobProcessor:
    """Process a list of images, save outputs, and write a master summary JSON."""

    def __init__(self, img_paths: List[str | Path], out_root: str | Path = "results", debug: bool = False):
        self.paths = [Path(p) for p in img_paths]
        self.out_root = Path(out_root)
        self.debug = debug
        self.summaries: List[Dict[str, Any]] = []
        self.out_root.mkdir(parents=True, exist_ok=True)

    def process_all(self) -> List[Dict[str, Any]]:
        """
        Iterate over images, saving outputs, collecting full blob data for next step,
        and writing a debug summary JSON containing only the top blob props for each image.
        Returns a list of dicts: each with 'image_path', 'expanded_labels' array, and 'props' list.
        """
        full_results: List[Dict[str, Any]] = []
        debug_props_list: List[List[Dict[str, Any]]] = []

        for p in self.paths:
            # 1) Run detection
            detector = BlobDetector(p, debug=self.debug).detect()
            detector.save_outputs(self.out_root)

            # 2) Extract top props & expanded_labels
            top_dict = detector.flood.top_props(5)
            props = top_dict["props"]
            labels = top_dict["expanded_labels"]

            # 3) Store full result for downstream processing
            full_results.append({
                "image_path": str(p.resolve()),
                "expanded_labels": labels,
                "props": props
            })

            # 4) Collect props-only for debug JSON
            debug_props_list.append(props)

            # 5) Clean up memory
            detector.dispose()
            if self.debug:
                print(f"Processed {p.name}: {len(props)} top blobs")

        # Write debug summary JSON (props only)
        with open(self.out_root / "summary.json", "w") as fh:
            json.dump(debug_props_list, fh, indent=2)

        return full_results
    
if __name__ == "__main__":
    #from tools.expand import BulkBlobProcessor
    import glob

    imgs = glob.glob("*/DevNotebooks/*.tif")

    summaries = BulkBlobProcessor(
        img_paths=imgs,
        out_root="lysozyme-stain-quantification/results",
        debug=True           # prints progress
    ).process_all()

    print("Master summary JSON:", summaries[0].keys())
    
    

 
