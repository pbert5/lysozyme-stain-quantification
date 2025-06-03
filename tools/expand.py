from __future__ import annotations
from skimage import measure
from skimage.segmentation import flood, watershed
from skimage.morphology import dilation, footprint_rectangle
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────── stdlib ────────────────────────────────────
import os, gc, json
from pathlib import Path
from typing import List, Dict, Any

# ═════════════════════════════════ CompetitiveFlooding ═════════════════════════
class CompetitiveFlooding:
    """Competitive flooding pipeline with fluent-interface support."""

    def __init__(self, tight_labels: np.ndarray, loose_mask: np.ndarray, red_image: np.ndarray,
                 terrain: np.ndarray | None = None, debug: bool = False):
        self.debug = debug
        self.tight_labels = tight_labels
        self.loose_mask  = loose_mask.astype(bool)
        self.red_image   = red_image
        self.terrain     = terrain if terrain is not None else np.ones_like(loose_mask, dtype=np.uint8)

        # Pre‑compute original blob areas
        self.original_areas: Dict[int, int] = {reg.label: reg.area for reg in measure.regionprops(tight_labels)}

        # Runtime products
        self.expanded_labels: np.ndarray | None = None
        self.swallowed_labels: np.ndarray | None = None
        self.border_labels: set[int] = set()
        self.adjacency: Dict[int, List[int]] = {}
        self.border_lengths: Dict[tuple[int, int], int] = {}
        self.swallow_plan: Dict[int, int] = {}

    # ──────────────────────────── helpers ────────────────────────────────
    @staticmethod
    def _border_labels(label_img: np.ndarray) -> set[int]:
        mask = np.zeros_like(label_img, bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return set(np.unique(label_img[mask])) - {0}

    def _overlay(self, labels: np.ndarray) -> np.ndarray:
        return (label2rgb(labels, image=self.red_image, bg_label=0, bg_color=None) * 255).astype(np.uint8)

    # ───────────────────────── pipeline steps ────────────────────────────
    def run_watershed(self) -> "CompetitiveFlooding":
        self.expanded_labels = watershed(self.terrain, markers=self.tight_labels.copy(), mask=self.loose_mask)
        self.swallowed_labels = self.expanded_labels.copy()
        self.border_labels = self._border_labels(self.expanded_labels)
        return self

    def compute_adjacency(self, size_factor: int = 20) -> "CompetitiveFlooding":
        if self.expanded_labels is None:
            raise RuntimeError("run_watershed() first")
        self.adjacency.clear(); self.border_lengths.clear()
        for lbl in np.unique(self.expanded_labels):
            if lbl == 0:
                continue
            area_lbl = self.original_areas.get(lbl, 0)
            mask = self.expanded_labels == lbl
            dil = dilation(mask, footprint_rectangle(3))
            neigh = [n for n in np.unique(self.expanded_labels[dil]) if n not in (0, lbl)]
            for n in neigh:
                if self.original_areas.get(n, 0) >= area_lbl * size_factor:
                    self.adjacency.setdefault(lbl, []).append(n)
                    shared = np.logical_and(dil, self.expanded_labels == n)
                    self.border_lengths[(lbl, n)] = int(np.sum(shared))
        return self

    def plan_swallow(self) -> "CompetitiveFlooding":
        self.swallow_plan.clear()
        for lbl, neigh in self.adjacency.items():
            if lbl in self.border_labels:
                continue
            best = max(neigh, key=lambda n: (self.border_lengths.get((lbl, n), 0), self.original_areas.get(n, 0)))
            self.swallow_plan[lbl] = best
        return self

    def apply_swallow(self) -> "CompetitiveFlooding":
        if self.expanded_labels is None:
            raise RuntimeError("run_watershed() first")
        self.swallowed_labels = self.expanded_labels.copy()
        for small, big in self.swallow_plan.items():
            self.swallowed_labels[self.swallowed_labels == small] = big
        return self

    # ───────────────────────── convenience chain ────────────────────────
    def run(self, size_factor: int = 20) -> "CompetitiveFlooding":
        return (self.run_watershed()
                    .compute_adjacency(size_factor)
                    .plan_swallow()
                    .apply_swallow())

    # ───────────────────────────── output ───────────────────────────────
    def save_results(self, save_dir: str | Path, save_name: str, save_labels = False) -> None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        comp = np.concatenate([
            self.red_image if self.red_image.ndim == 3 else np.stack([self.red_image]*3, -1),
            self._overlay(self.expanded_labels),
            self._overlay(self.swallowed_labels)], axis=1)
        plt.imsave(str(save_dir / save_name), comp.astype(np.uint8))
        if save_labels is True:
            # Also save the full labeled arrays
            np.save(save_dir / "expanded_labels.npy", self.expanded_labels)
            np.save(save_dir / "swallowed_labels.npy", self.swallowed_labels)

    def top_props(self, n: int = 5) -> Dict[str, Any]:
        """
        Return top `n` blob properties plus the full expanded_labels array.

        Returns a dict:
          - 'expanded_labels': 2D label array
          - 'props': list of dicts with 'label','area','centroid','bbox'
        """
        if self.swallowed_labels is None:
            raise RuntimeError("apply_swallow() must be called first")
        regions = measure.regionprops(self.swallowed_labels)
        top = sorted(regions, key=lambda r: r.area, reverse=True)[:n]
        props = [
            {
                "label": int(r.label),
                "area": int(r.area),
                "centroid": [float(c) for c in r.centroid],
                "bbox": [int(b) for b in r.bbox]
            }
            for r in top
        ]
        return {"expanded_labels": self.expanded_labels, "props": props}
