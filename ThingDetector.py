"""
Competitive Flooding + Blob Detection bulk-processing toolkit.

This file now contains:
• CompetitiveFlooding  - fast, fluent pipeline (unchanged speed)
• BlobDetector         - wraps prep → segmentation → flooding
• BulkBlobProcessor    - iterate over many images, save outputs and
                          return summary dicts while freeing memory.
"""
from __future__ import annotations

# # ─────────────────────────────────── stdlib ────────────────────────────────────
# import os, gc, json
from pathlib import Path
from typing import List, Dict, Any

# # ─────────────────────────────── third‑party libs ─────────────────────────────
# import numpy as np
# import cv2
# import tifffile
# import matplotlib.pyplot as plt
# from skimage import measure, img_as_ubyte
# from skimage.segmentation import watershed, flood
# from skimage.morphology import dilation, square
# from skimage.color import label2rgb

# ───────────────────────────────── user modules ───────────────────────────────
# from tools.ImagePrep import image_prep              # noqa: F401
# from tools.detection_methods import DetectionMethods # noqa: F401
# from tools.expand import CompetitiveFlooding  # noqa: F401
from tools import *

# ═══════════════════════════════════ BlobDetector ═════════════════════════════


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

    imgs = glob.glob("DemoData/ClearedForProcessing/*.tif")

    summaries = BulkBlobProcessor(
        img_paths=imgs,
        out_root="lysozyme-stain-quantification/results",
        debug=True           # prints progress
    ).process_all()

    print("Master summary JSON:", summaries[0].keys())
    
    

 
