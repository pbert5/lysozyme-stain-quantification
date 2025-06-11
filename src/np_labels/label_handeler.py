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


    ## ai CLASS refactored 
from collections import defaultdict
from itertools import combinations
import numpy as np
from skimage.measure import regionprops, perimeter
from skimage.morphology import dilation, rectangle
from skimage.color import label2rgb
import matplotlib.pyplot as plt

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
        np.save(f"{save_path}_prev.npy", self.labels)

        

        if save_csv:
            props = measure.regionprops(self.expanded_labels)
            with open(f"{save_path}_props.csv", "w") as f:
                f.write("label,area,centroid_y,centroid_x,bbox\n")
                for r in props:
                    line = f"{r.label},{r.area},{r.centroid[0]:.2f},{r.centroid[1]:.2f},{list(r.bbox)}\n"
                    f.write(line)

    class MergePipeline:
        def __init__(self, label_img, raw_img=None, singleton_penalty=10.0, sample_size=100):
            self.label_img = label_img
            self.raw_img = raw_img
            self.singleton_penalty = singleton_penalty
            self.sample_size = sample_size
            
            # Will be populated
            self.props = {}
            self.perims = {}
            self.cents = {}
            self.areas = {}
            self.shared = defaultdict(lambda: defaultdict(int))
            self.triangles = []
            self.combos = {}
            self.best_stage1 = {}
            self.second_stage_results = {}
            self.merged_label_array = None

        def compute_stats(self):
            # Compute region properties
            self.props = {r.label: r for r in regionprops(self.label_img)}
            self.perims = {lbl: perimeter(self.label_img == lbl) for lbl in self.props}
            self.cents = {lbl: self.props[lbl].centroid for lbl in self.props}
            self.areas = {lbl: self.props[lbl].area for lbl in self.props}
            # Build shared-perimeter adjacency
            for lbl in self.props:
                mask = self.label_img == lbl
                dil = dilation(mask, rectangle(3,3))
                neighs = set(np.unique(self.label_img[dil])) - {0, lbl}
                for n in neighs:
                    shared_p = int(np.logical_and(dil, self.label_img==n).sum())
                    self.shared[lbl][n] = shared_p
                    self.shared[n][lbl] = shared_p
            return self

        def find_triangles(self):
            triangles = set()
            for a in self.shared:
                neighs = list(self.shared[a])
                for b, c in combinations(neighs, 2):
                    if b in self.shared[c] and c in self.shared[b]:
                        triangles.add(tuple(sorted((a, b, c))))
            self.triangles = sorted(triangles)
            return self

        def build_candidate_groups(self):
            tri_index = defaultdict(list)
            for tri in self.triangles:
                for lbl in tri:
                    tri_index[lbl].append(tri)
            combos = defaultdict(list)
            for lbl in self.shared:
                combos[lbl].append((lbl,))
                for n in self.shared[lbl]:
                    if n != lbl:
                        combos[lbl].append((lbl, n))
                combos[lbl].extend(tri_index[lbl])
                seen = set()
                for t1 in tri_index[lbl]:
                    for t2 in tri_index[lbl]:
                        if t1 == t2:
                            continue
                        inter = set(t1).intersection(t2)
                        if len(inter) == 2 and lbl in inter:
                            merged = tuple(sorted(set(t1).union(t2)))
                            if merged not in seen:
                                combos[lbl].append(merged)
                                seen.add(merged)
            self.combos = combos
            return self

        def evaluate_stage1(self):
            best = {}
            for lbl, clist in self.combos.items():
                P = self.perims[lbl]
                best_score, best_combo = -1.0, (lbl,)
                for combo in clist:
                    if combo == (lbl,):
                        score = 1.0 / self.singleton_penalty
                    else:
                        shared_sum = sum(self.shared[lbl][n] for n in combo if n != lbl)
                        score = shared_sum / (P + 1e-8)
                    if score > best_score:
                        best_score, best_combo = score, combo
                best[lbl] = (best_combo, best_score)
                    # ←–– New block to catch “no-combo” labels ––→
            for lbl in self.props:
                if lbl not in best:
                    best[lbl] = ((lbl,), 1.0/self.singleton_penalty)
                    
            self.best_stage1 = best
            return self

        def evaluate_stage2(self):
            # assign groups and leaders
            group_map = defaultdict(set)
            for lbl, (group, _) in self.best_stage1.items():
                norm = tuple(sorted(int(g) for g in group))
                group_map[norm].add(int(lbl))
            group_leaders = {grp: max(grp, key=lambda x: self.areas[x]) for grp in group_map}
            # stage2 results
            results = {}
            for grp, members in group_map.items():
                leader = group_leaders[grp]
                expanded = set(grp)
                for l in grp:
                    expanded.update(self.shared[l].keys())
                expanded = {int(l) for l in expanded if int(l) in self.props}
                for lbl in members:
                    # compute score
                    tot_area = sum(self.areas[l] for l in expanded)
                    tot_perim = sum(self.perims[l] for l in expanded)
                    coords = np.vstack([self.props[l].coords for l in expanded])
                    com = coords.mean(axis=0)
                    lbl_coords = self.props[lbl].coords
                    if len(lbl_coords) > self.sample_size:
                        idx = np.linspace(0, len(lbl_coords)-1, self.sample_size).astype(int)
                        lbl_coords = lbl_coords[idx]
                    dists = np.linalg.norm(lbl_coords - com, axis=1)
                    avg_dist = dists.mean()
                    score2 = (tot_area / (tot_perim + 1e-8)) / (avg_dist + 1e-8)
                    results[lbl] = (leader, score2)
            # add true singletons
            all_lbls = set(self.props.keys())
            for lbl in all_lbls - set(results):
                results[int(lbl)] = (int(lbl), 0.0)
            self.second_stage_results = results
            return self

        def relabel(self):
            mapping = {lbl: tgt for lbl, (tgt, _) in self.second_stage_results.items()}
            out = np.zeros_like(self.label_img)
            for old, new in mapping.items():
                out[self.label_img == old] = new
            self.merged_label_array = out
            return self

        def run(self):
            return (self.compute_stats()
                        .find_triangles()
                        .build_candidate_groups()
                        .evaluate_stage1()
                        .evaluate_stage2()
                        .relabel())

        def plot_initial(self, title="Original Labels"):
            img = label2rgb(self.label_img, bg_label=0)
            plt.figure(figsize=(6,6))
            plt.imshow(img); plt.title(title); plt.axis('off')

        def plot_stage1(self, title="Stage 1 Groupings"):
            img = label2rgb(self.label_img, bg_label=0)
            plt.figure(figsize=(6,6))
            plt.imshow(img); plt.title(title); plt.axis('off')
            for region in regionprops(self.label_img):
                if region.label == 0: continue
                grp, _ = self.best_stage1[region.label]
                txt = ",".join(str(int(x)) for x in grp)
                y,x = region.centroid
                plt.text(x,y,txt,ha='center',va='center',color='white',
                        bbox=dict(facecolor='black',alpha=0.5,lw=0))

        def plot_final(self, title="Final Merged Labels"):
            img = label2rgb(self.merged_label_array, bg_label=0)
            plt.figure(figsize=(6,6))
            plt.imshow(img); plt.title(title); plt.axis('off')
            for region in regionprops(self.merged_label_array):
                if region.label == 0: continue
                y,x = region.centroid
                plt.text(x,y,str(region.label),ha='center',va='center',color='white',
                        bbox=dict(facecolor='black',alpha=0.5,lw=0))
    
    
    
    
            

    
    