from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional
import numpy as np
from blob_det import BlobDetector, get_rfp_dapi_pairs, process_all_rfp_dapi_pairs
from np_labels.save_labels import LabelsToGeoJSON
import tifffile
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
from roifile import ImagejRoi
import zipfile
import os


class BulkBlobProcessor:
    """Process a list of images, save outputs, and write a master summary JSON."""

    def __init__(self, img_paths: List[Path], out_root: str | Path = "results", results_dir: Optional[Path] = None, ROI_expand_by: int=0, debug: bool = False):
        self.paths = img_paths
        self.out_root = Path(out_root)
        self.debug = debug
        self.results_dir = Path(results_dir) if results_dir is not None else None
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
            labels_array = np.array(labels)  # Convert labels to a NumPy array
            print(f"Processed {entry['id']}: found {np.unique(labels_array).size - 1} ROIs (excluding background)") if self.debug else None

        return self.full_results

        return self
    def save_results(self, output_format="geojson") -> None:
        common_prefix = os.path.commonpath([str(p) for p in self.paths])
        relative_paths = [p.relative_to(common_prefix) for p in self.paths]

        for labels, rel_path in zip(self.full_results, relative_paths):
            img_path = Path(labels["image_path"])
            base_name = img_path.stem

            # Create subdirectory based on the relative path
            output_subdir = self.out_root / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            if output_format == "geojson":
                # ——— Save original GeoJSON and NPY as before ———
                geojson_dir = output_subdir / "geojson"
                geojson_dir.mkdir(parents=True, exist_ok=True)
                LabelsToGeoJSON(
                    labels["labels"],
                    output_path=geojson_dir / f"{base_name}_rois.geojson",
                    pixel_size=1.0,
                    origin=(0, 0),
                    expand_by=self.ROI_expand_by
                )

                npy_dir = output_subdir / "npy"
                npy_dir.mkdir(parents=True, exist_ok=True)
                np.save(npy_dir / f"{labels['id']}_labels.npy", labels["labels"])
            elif output_format == "tifffile":
                # ——— Prepare ImageJ-ready TIFF + ROI ZIP ———
                out_dir = self.out_root / "ImageJ_ready"
                out_dir.mkdir(parents=True, exist_ok=True)

                # 1) copy the TIFF
                img = tifffile.imread(img_path)
                tifffile.imwrite(out_dir / f"{base_name}.tif", img)

                # 2) generate one .roi per contour
                mask = labels["labels"].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                roi_files: list[Path] = []
                for i, cnt in enumerate(contours):
                    if cnt.shape[0] < 3:
                        continue
                    coords = cnt[:, 0, :].astype(float)  # Nx2 float array
                    roi = ImagejRoi.frompoints(coords, name=f"{base_name}_{i}")
                    roi_path = out_dir / f"{base_name}_{i}.roi"
                    roi.tofile(str(roi_path))  # write the .roi file :contentReference[oaicite:0]{index=0}
                    roi_files.append(roi_path)

                # 3) bundle them into a ZIP named exactly like the TIFF
                zip_path = out_dir / f"{base_name}.zip"
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for fn in roi_files:
                        zf.write(fn, arcname=fn.name)
            else:
                raise ValueError("Unsupported output format. Choose 'geojson' or 'tifffile'.")

        print(f"Saved {output_format} results for {base_name}")
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


class RFPDAPIProcessor:
    """Process RFP/DAPI image pairs using watershed refinement."""
    
    def __init__(self, project_root: Path, debug: bool = False):
        self.project_root = Path(project_root)
        self.debug = debug
        self.detector = BlobDetector(debug=debug)
    
    def find_image_pairs(self, max_pairs: int = 30) -> List[tuple]:
        """Find RFP/DAPI image pairs in the project."""
        images_root = self.project_root / 'lysozyme images'
        return get_rfp_dapi_pairs(images_root, max_pairs=max_pairs)
    
    def process_all_pairs(self, max_pairs: int = 30, show_plots: bool = True) -> List[np.ndarray]:
        """Process all RFP/DAPI pairs and return watershed results."""
        return process_all_rfp_dapi_pairs(
            self.project_root, 
            max_show=max_pairs, 
            show_plots=show_plots
        )
    
    def process_single_pair(self, rfp_path: Path, dapi_path: Path, 
                           show_visualization: bool = True) -> np.ndarray:
        """Process a single RFP/DAPI pair."""
        ws_labels = self.detector.process_rfp_dapi_pair(rfp_path, dapi_path)
        
        if ws_labels.size > 0 and show_visualization:
            self.detector.visualize_watershed_results(
                rfp_path, dapi_path, ws_labels, self.project_root
            )
        
        return ws_labels


        
if __name__ == "__main__":
    print("Running BulkBlobProcessor...")
    from pathlib import Path

    # 1) Define your inputs:
    img_dir    = Path("/home/user/nfs/analysisdata/Stt4 Lysozyme stain quantification/Lysozome images")       # ← change this
    img_glob   = "**/*.tif"                            # ← or "*.png", etc.
    img_paths  = sorted(img_dir.glob(img_glob))

    # 2) Define output + options:
    out_root   = Path("/home/user/nfs/analysisdata/Stt4 Lysozyme stain quantification/results")                    # where to save
    results_dir = None                              # not used for now
    expand_by  = 1.0                                # how much to expand each ROI
    debug      = True                               # dump debug artifacts?
    singleton_penalty = 4                                # proportion of how much more perimeter needs to be in contact then not for merge to happen

    # 3) Instantiate & run:
    # Ensure img_paths is a list of strings
    max_images = 10  # Set the maximum number of images to process for testing
    img_paths = [Path(p) for p in img_paths[:max_images]]

    # Ensure results_dir is a string if it's not None
    results_dir = Path(results_dir) if results_dir else None

    # Convert expand_by to an integer
    ROI_expand_by = int(expand_by)

    processor = BulkBlobProcessor(
        img_paths=img_paths,
        out_root=out_root,
        results_dir=results_dir,
        ROI_expand_by=ROI_expand_by,
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

