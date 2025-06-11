from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from blob_det import BlobDetector
from np_labels.labels_to_geojson import LabelsToGeoJSON
import tifffile


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
    def process_all(self) -> List[Dict[str, Any]]:
        """
        Iterate over images, saving outputs, collecting full blob data for next step,
        and writing a debug summary JSON containing only the top blob props for each image.
        Returns a list of dicts: each with 'image_path', and the blob labels.
        """
        self.full_results = []
        
        for entry in self.images:
            # run detection on the ndarray, get back a label array
            labels = BlobDetector(channel=0, debug=self.debug).detect(
                image=entry["array"]
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