from __future__ import annotations
import numpy as np
from skimage import measure
from skimage.segmentation import flood, watershed
from skimage.morphology import dilation, footprint_rectangle

# ─────────────────────────────────── stdlib ────────────────────────────────────
import os, gc, json
from pathlib import Path
from typing import List, Dict, Any

class LabelHandeler:
    def __init__(self, labels: np.ndarray = None, terrain: np.ndarray | None = None, positive_mask: np.ndarray = None):
        
        self.labels = labels
        # Pre‑compute original blob areas
        self.original_areas: Dict[int, int] = {reg.label: reg.area for reg in measure.regionprops(labels)}
        # Runtime products
        self.expanded_labels: np.ndarray | None = None
        self.merged_labels: np.ndarray | None = None
        self.terrain = terrain if terrain is not None else np.ones_like(positive_mask, dtype=np.uint8)
        
    # ──────────────────────────── helpers ────────────────────────────────
    @staticmethod
    def _border_labels(label_img: np.ndarray) -> set[int]:
        mask = np.zeros_like(label_img, bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return set(np.unique(label_img[mask])) - {0}
    # ───────────────────────── pipeline methods ────────────────────────────
    def flood_fill(self, labels) -> LabelHandeler: #aka run_watershed(self) -> "CompetitiveFlooding"
        """
        Perform flood fill on the labels.

        Returns:
            LabelHandeler: The instance with flood-filled labels.
        """
        self.expanded_labels = watershed(self.terrain, markers=self.tight_labels.copy(), mask=self.loose_mask)
        self.swallowed_labels = self.expanded_labels.copy()
        self.border_labels = self._border_labels(self.expanded_labels)
        return self
        
        
        return self
    
    def merge_labels(self) -> LabelHandeler:
        ... # Placeholder for the actual implementation of merging labels
        return self # This method should contain the logic to merge labels based on specific criteria., should be able to chain together
    
    
    
    
            

    
    