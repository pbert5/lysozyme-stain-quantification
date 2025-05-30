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
        top = [{
            "label": p.label,
            "area": int(p.area),
            "centroid": tuple(map(float, p.centroid)),
            "bbox": tuple(map(int, p.bbox))
        } for p in self.flood.top_props(5)]
        return {
            "image_path": str(self.path.resolve()),
            "top_blobs": top
        }

    # ───────────────────────────── saving ───────────────────────────────
    def save_outputs(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{self.path.stem}_red.png"), img_as_ubyte(self.red_image))
        self.flood.save_results(out_dir, f"{self.path.stem}_cf.png")

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
        for p in self.paths:
            detector = BlobDetector(p, debug=self.debug).detect()
            detector.save_outputs(self.out_root)
            summary = detector.summary()
            # add file paths to summary
            summary["red_channel_file"] = str((self.out_root / f"{p.stem}_red.png").resolve())
            summary["competitive_flooding_file"] = str((self.out_root / f"{p.stem}_cf.png").resolve())
            self.summaries.append(summary)
            detector.dispose()
            if self.debug:
                print(f"Processed {p.name} ({len(summary['top_blobs'])} blobs)")
        # master JSON
        with open(self.out_root / "summary.json", "w") as fh:
            json.dump(self.summaries, fh, indent=2)
        return self.summaries
    
if __name__ == "__main__":
    #from tools.expand import BulkBlobProcessor
    import glob

    imgs = glob.glob("lysozyme-stain-quantification/DevNotebooks/*.tif")

    summaries = BulkBlobProcessor(
        img_paths=imgs,
        out_root="lysozyme-stain-quantification/results",
        debug=True           # prints progress
    ).process_all()

    print("Master summary JSON:", summaries[0].keys())
    
    

 
