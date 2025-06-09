from __future__ import annotations
import numpy as np
from skimage import measure
from skimage.segmentation import flood, watershed
from skimage.morphology import dilation#, footprint_rectangle
from skimage.color import label2rgb

# ─────────────────────────────────── stdlib ────────────────────────────────────
import os, gc, json
from pathlib import Path
from typing import List, Dict, Any
from matplotlib import pyplot as plt

class LabelHandeler:
    def __init__(self, labels: np.ndarray = None, terrain: np.ndarray | None = None, positive_mask: np.ndarray = None):
        
        self.labels = labels
        self.positive_mask = positive_mask if positive_mask is not None else np.ones_like(labels, dtype=bool)
        # Pre‑compute original blob areas
        self.original_areas: Dict[int, int] = {reg.label: reg.area for reg in measure.regionprops(labels)}
        # Runtime products
        self.expanded_labels: np.ndarray | None = None
        self.merged_labels: np.ndarray | None = None
        self.adjacency: Dict[int, List[int]] = {}
        self.terrain = terrain if terrain is not None else np.ones_like(positive_mask, dtype=np.uint8)
        
    # ──────────────────────────── helpers ────────────────────────────────
    @staticmethod
    def _border_labels(label_img: np.ndarray) -> set[int]:
        mask = np.zeros_like(label_img, bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return set(np.unique(label_img[mask])) - {0}
    def _overlay(self, og_image, labels: np.ndarray   ) -> np.ndarray:
        return (label2rgb(labels, image=og_image, bg_label=0, bg_color=None) * 255).astype(np.uint8)
    # ───────────────────────── pipeline methods ────────────────────────────
    def flood_fill(self) -> LabelHandeler: #aka run_watershed(self) -> "CompetitiveFlooding"
        """
        Perform flood fill on the labels.

        Returns:
            LabelHandeler: The instance with flood-filled labels.
        """
        self.expanded_labels = watershed(self.terrain, markers=self.labels, mask=self.positive_mask)
        # self.swallowed_labels = self.expanded_labels.copy()
        self.border_labels = self._border_labels(self.expanded_labels)
        return self
        
        
        return self
    
    def save_expanded_labels(self, save_path: str | Path,  save_csv: bool = False ) -> None:
        """
        Save the expanded label array to disk for experimentation.

        Parameters
        ----------
        save_path : str or Path
            Base path (no extension) where the expanded labels will be saved.
            For example: 'out/labels_test' will produce:
                - 'labels_test.npy'
                - 'labels_test_overlay.png' (if save_overlay)
                - 'labels_test_props.csv' (if save_csv)
        save_overlay : bool
            Whether to save a PNG image of the overlay.
        save_csv : bool
            Whether to export region properties to a CSV.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as .npy
        np.save(f"{save_path}.npy", self.expanded_labels)

        

        if save_csv:
            props = measure.regionprops(self.expanded_labels)
            with open(f"{save_path}_props.csv", "w") as f:
                f.write("label,area,centroid_y,centroid_x,bbox\n")
                for r in props:
                    line = f"{r.label},{r.area},{r.centroid[0]:.2f},{r.centroid[1]:.2f},{list(r.bbox)}\n"
                    f.write(line)

    def merge_labels(self, size_factor: int = 20) -> "LabelHandeler":
        ...

        return self
    
    
    
    
            

    
    