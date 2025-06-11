from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from blob_det import BlobDetector
from np_labels.labels_to_geojson import LabelsToGeoJSON



class BulkBlobProcessor:
    """Process a list of images, save outputs, and write a master summary JSON."""

    def __init__(self, img_paths: List[str | Path], out_root: str | Path = "results", results_dir: str | Path = None, ROI_expand_by: int=0.0, debug: bool = False):
        self.paths = [Path(p) for p in img_paths]
        self.out_root = Path(out_root)
        self.debug = debug
        self.results_dir = Path(results_dir) if results_dir else None
        self.ROI_expand_by = ROI_expand_by
    def load_images(self) -> List[Path]:
        """
        Load images from the provided paths.
        Returns a list the images.
        """
        self.images: List[Dict[str, np.ndarray]] = []
        ... # Placeholder for actual image loading logic

    def process_all(self) -> List[Dict[str, Any]]:
        """
        Iterate over images, saving outputs, collecting full blob data for next step,
        and writing a debug summary JSON containing only the top blob props for each image.
        Returns a list of dicts: each with 'image_path', and the blob labels.
        """
        self.full_results: [{image_path: string, labels: np.ndarray }] = [] # this would be a list of all the labels from each image

        for p in self.images:
            # 1) Run detection
            labels = BlobDetector(channel =0).detect(image = p)
            
            # 2) Collect full results for next step
            self.full_results.append({
                "id": p.name,
                "image_path": str | Path(p),
                "labels": labels
                })


            #  Clean up memory
            labels.dispose()

        return self
    def save_results(self) -> None:
        for labels in self.full_results:
            # Save each label set to a file
            if self.out_root:
                self.out_root.mkdir(parents=True, exist_ok=True)
            output_filename = Path(labels["image_path"]).stem + "_rois.geojson"
            output_geojson = self.out_root / output_filename

            LabelsToGeoJSON(
                labels,
                output_path=output_geojson,
                pixel_size=1.0,  # Adjust if your image has a physical pixel size
                origin=(0, 0),
                expand_by=self.expand_by
            )
            out_path = self.out_root / f"{labels['id']}_labels.npy"
            np.save(out_path, labels['labels'])
            print(f"Saved labels for {labels['id']} to {out_path}")
        # Save master summary JSON