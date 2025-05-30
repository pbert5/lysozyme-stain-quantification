from skimage import measure
from skimage.segmentation import flood, watershed
from skimage.morphology import dilation, square
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt
import os

class CompetitiveFlooding:
    def __init__(self, tight_labels, loose_mask, red_image, terrain=None, debug = False):
        
        self.tight_labels = tight_labels
        self.loose_mask = loose_mask
        self.red_image = red_image
        self.terrain = (terrain if terrain is not None
                        else np.ones_like(loose_mask, dtype=np.uint8))

        # Containers populated by the pipeline
        self.original_areas: dict[int, int] = {}
        self.loose_boundaries: dict[int, np.ndarray] = {}
        self.expanded_labels: np.ndarray | None = None
        self.swallowed_labels: np.ndarray | None = None
        self.adjacency: dict[int, list[int]] = {}
        self.border_lengths: dict[tuple[int, int], int] = {}
        self.swallow_plan: dict[int, int] = {}
        self.border_labels: set[int] = set()
        self.debug = debug

        self._compute_original_areas()
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_original_areas(self):
        for region in measure.regionprops(self.tight_labels):
            self.original_areas[region.label] = region.area

    def _get_border_labels(self, label_image):
        mask = np.zeros_like(label_image, dtype=bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        labels = np.unique(label_image[mask])
        return set(labels[labels != 0])
    # ------------------------------------------------------------------
    # AIO
    # ------------------------------------------------------------------
    def run(self, save_dir="results"):
        self.compute_loose_boundaries().run_watershed().compute_adjacency(size_factor=20).plan_swallow().apply_swallow().visualize(save_dir)
    
    # ------------------------------------------------------------------
    # Public pipeline steps (each returns self for chaining)
    # ------------------------------------------------------------------

    def compute_loose_boundaries(self):
        """Flood fill around each centroid on the loose mask."""
        for region in measure.regionprops(self.tight_labels):
            centroid = tuple(map(int, region.centroid))
            self.loose_boundaries[region.label] = flood(
                self.loose_mask, seed_point=centroid)
        if self.debug: print("Loose boundaries computed for tight labels.")
        return self

    def run_watershed(self):
        """Expand tight blobs into loose regions via watershed."""
        self.expanded_labels = watershed(
            self.terrain,
            markers=self.tight_labels.copy(),
            mask=self.loose_mask)
        self.swallowed_labels = self.expanded_labels.copy()
        self.border_labels = self._get_border_labels(self.expanded_labels)
        if self.debug:
            print(f"Watershed run complete. "
                  f"Expanded labels shape: {self.expanded_labels.shape}, "
                  f"Border labels: {self.border_labels}")
        return self

    def compute_adjacency(self, size_factor: int = 20):
        """Build adjacency of small blobs to large neighbors (≥ size_factor ×)."""
        if self.expanded_labels is None:
            raise RuntimeError("run_watershed() first")

        self.adjacency.clear()
        self.border_lengths.clear()
        for lbl in np.unique(self.expanded_labels):
            if lbl == 0:
                continue
            area_lbl = self.original_areas.get(lbl, 0)
            mask = self.expanded_labels == lbl
            dil = dilation(mask, square(3))
            neighbors = np.unique(self.expanded_labels[dil])
            neighbors = [n for n in neighbors if n not in (0, lbl)]

            eligible = []
            for n in neighbors:
                area_n = self.original_areas.get(n, 0)
                if area_n >= area_lbl * size_factor:
                    eligible.append(n)
                    shared = np.logical_and(dil, self.expanded_labels == n)
                    self.border_lengths[(lbl, n)] = int(np.sum(shared))
            if eligible:
                self.adjacency[lbl] = eligible
        if self.debug:
            print(f"Adjacency computed. "
                  f"Found {len(self.adjacency)} small blobs with neighbors.")
            for lbl, neighbors in self.adjacency.items():
                print(f"Label {lbl} has neighbors: {neighbors} "
                      f"with border lengths: {[self.border_lengths.get((lbl, n), 0) for n in neighbors]}")
        return self

    def plan_swallow(self):
        """Select best large neighbor for each small blob (largest shared border)."""
        self.swallow_plan.clear()
        for lbl, neighbors in self.adjacency.items():
            if lbl in self.border_labels:
                continue  # do not swallow border‑touching blobs
            best = max(neighbors, key=lambda n: (
                self.border_lengths.get((lbl, n), 0),
                self.original_areas.get(n, 0)))
            self.swallow_plan[lbl] = best
        if self.debug:
            print(f"Label {lbl} will be swallowed by {best} "
                    f"with border length {self.border_lengths.get((lbl, best), 0)}")
        return self

    def apply_swallow(self):
        """Relabel small blobs with their chosen large neighbor labels."""
        if self.expanded_labels is None:
            raise RuntimeError("run_watershed() first")
        self.swallowed_labels = self.expanded_labels.copy()
        for small, big in self.swallow_plan.items():
            self.swallowed_labels[self.swallowed_labels == small] = big
        if self.debug:
            print(f"Swallow plan applied. "
                  f"Swallowed labels: {list(self.swallow_plan.keys())} "
                  f"into {list(self.swallow_plan.values())}")
        return self

    def visualize(self, save_dir= None,
                  save_name = "competitive_flooding.png"):
        """Show and optionally save a three‑panel summary figure.

        Returns self so you can keep chaining if you like (e.g., to access
        attributes afterwards without breaking the chain).
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if self.swallowed_labels is None:
            raise RuntimeError("apply_swallow() first")

        expanded_rgb = label2rgb(self.expanded_labels, image=self.red_image, bg_label=0)
        swallowed_rgb = label2rgb(self.swallowed_labels, image=self.red_image, bg_label=0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(self.red_image, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(expanded_rgb)
        axes[1].set_title("Expanded Labels")
        axes[1].axis("off")

        axes[2].imshow(swallowed_rgb)
        axes[2].set_title("Swallowed Labels")
        axes[2].axis("off")

        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, save_name), bbox_inches="tight", dpi=300)
        plt.show()
        if self.debug:
            print(f"Visualization saved to {os.path.join(save_dir, save_name)}" if save_dir else "Visualization displayed.")
        return self
        
