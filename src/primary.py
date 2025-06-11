from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from blob_det import BlobDetector
from np_labels.labels_to_geojson import LabelsToGeoJSON
import tifffile
from skimage.color import label2rgb
import matplotlib.pyplot as plt


class BulkBlobProcessor:
    """Process a list of images, save outputs, and write a master summary JSON."""

    def __init__(self, img_paths: List[str | Path], out_root: str | Path = "results", results_dir: str | Path = None, ROI_expand_by: int=0.0, debug: bool = False):
        self.paths = [Path(p) for p in img_paths]
        self.out_root = Path(out_root)
        self.debug = debug
        self.results_dir = Path(results_dir) if results_dir else None
        self.ROI_expand_by = ROI_expand_by
    def load_images(self) -> List[Dict[str, Any]]:
        """Read each file in self.paths into memory."""
        self.images = []
        for p in self.paths:
            arr = tifffile.imread(p)
            self.images.append({
                "id": p.stem,
                "array": arr,
                "path": p,
            })
        return self.images
    def process_all(self, low_thresh=30, high_thresh=150, singleton_penalty=10) -> List[Dict[str, Any]]:
        """
        Iterate over images, saving outputs, collecting full blob data for next step,
        and writing a debug summary JSON containing only the top blob props for each image.
        Returns a list of dicts: each with 'image_path', and the blob labels.
        """
        self.full_results = []
        
        for entry in self.images:
            # run detection on the ndarray, get back a label array
            labels = BlobDetector(channel=0, debug=self.debug).detect(
                image=entry["array"],
                segmentation_low_thresh=low_thresh,
                segmentation_high_thresh=high_thresh,
                scale_bar_intensity_threshold=240,
                scale_bar_min_area=500,
                scale_bar_aspect_ratio_thresh=4.0,
                positive_mask_threshold=0.5,
                singleton_penalty=singleton_penalty
            )

            self.full_results.append({
                "id": entry["id"],
                "image_path": entry["path"],
                "labels": labels,
            })
            # if you really need to free memory:
            # del entry["array"]

        return self.full_results

        return self
    def save_results(self) -> None:
        for labels in self.full_results:
            # Save each label set to a file
            if self.out_root:
                self.out_root.mkdir(parents=True, exist_ok=True)
            output_filename = Path(labels["image_path"]).stem + "_rois.geojson"
            output_geojson = self.out_root / output_filename

            LabelsToGeoJSON(
                labels["labels"],
                output_path=output_geojson,
                pixel_size=1.0,  # Adjust if your image has a physical pixel size
                origin=(0, 0),
                expand_by=self.ROI_expand_by
            )
            out_path = self.out_root / f"{labels['id']}_labels.npy"
            np.save(out_path, labels['labels'])
            print(f"Saved labels for {labels['id']} to {out_path}")
        # Save master summary JSON
    def save_visuals(self, out_dir: str | Path, alpha: float = 0.5) -> None:
        """
        Quick-look renderer: overlays labels on original image.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for r in self.full_results:
            # load original image
            original = tifffile.imread(r["image_path"])
            # overlay colored labels on top
            overlay = label2rgb(
                r["labels"].astype(np.uint8),
                image=original,
                alpha=alpha,
                bg_label=0,
            )
            plt.imsave(out / f"{r['id']}_overlay.png", overlay)
        
if __name__ == "__main__":
    print("Running BulkBlobProcessor...")
    from pathlib import Path

    # 1) Define your inputs:
    img_dir    = Path("/home/user/documents/PiereLab/lysozyme/DemoData/ClearedForProcessing")       # ← change this
    img_glob   = "*.tif"                            # ← or "*.png", etc.
    img_paths  = sorted(img_dir.glob(img_glob))

    # 2) Define output + options:
    out_root   = Path("results")                    # where to save
    results_dir = None                              # not used for now
    expand_by  = 1.0                                # how much to expand each ROI
    debug      = True                               # dump debug artifacts?
    singleton_penalty = 4                                # proportion of how much more perimeter needs to be in contact then not for merge to happen

    # 3) Instantiate & run:
    processor = BulkBlobProcessor(
        img_paths=img_paths,
        out_root=out_root,
        results_dir=results_dir,
        ROI_expand_by=expand_by,
        debug=debug,
    )
    print(f"Processing {len(processor.paths)} images...")
    processor.load_images()     # read all images into memory
    print("Loaded images, starting processing...")
    processor.process_all(low_thresh=30, high_thresh=150, singleton_penalty=singleton_penalty)     # run your BlobDetector on each
    print("Processing complete, saving results...")
    processor.save_results()    # emit .geojson, .npy, and prints
    print("Results saved, generating visuals...")
    processor.save_visuals(out_root / "quick_check")
    print("Visuals saved, all done!")


"""
python blob_bulk.py /path/to/images/*.tif --out_root=results --expand_by=2.0 --debug
"""