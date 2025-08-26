"""
Merge pipeline for combining adjacent labeled regions.
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from skimage.measure import regionprops, perimeter
from skimage.morphology import dilation
from skimage.color import label2rgb
import matplotlib.pyplot as plt


class MergePipeline:
    """Pipeline for merging adjacent labeled regions based on shared perimeters."""
    
    def __init__(self, label_img, raw_img=None, singleton_penalty=10.0, sample_size=100, debug=False):
        """
        Initialize the merge pipeline.
        
        Args:
            label_img: Labeled image array
            raw_img: Optional raw image for additional analysis
            singleton_penalty: Penalty factor for singleton regions
            sample_size: Number of coordinate samples for distance calculations
            debug: Whether to enable debug output
        """
        self.label_img = label_img
        self.raw_img = raw_img
        self.singleton_penalty = singleton_penalty
        self.sample_size = sample_size
        self.debug = debug
        
        # Will be populated during processing
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
        """Compute region properties and adjacency statistics."""
        # Compute region properties
        self.props = {r.label: r for r in regionprops(self.label_img)}
        self.perims = {lbl: perimeter(self.label_img == lbl) for lbl in self.props}
        self.cents = {lbl: self.props[lbl].centroid for lbl in self.props}
        self.areas = {lbl: self.props[lbl].area for lbl in self.props}
        
        if self.debug:
            print(f"[MERGE DEBUG] Found {len(self.props)} regions with areas: {[self.areas[lbl] for lbl in sorted(self.props.keys())]}")
        
        # Build shared-perimeter adjacency
        adjacency_count = 0
        for lbl in self.props:
            mask = self.label_img == lbl
            dil = dilation(mask, np.ones((3, 3), dtype=bool))
            neighs = set(np.unique(self.label_img[dil])) - {0, lbl}
            for n in neighs:
                shared_p = int(np.logical_and(dil, self.label_img == n).sum())
                self.shared[lbl][n] = shared_p
                self.shared[n][lbl] = shared_p
                adjacency_count += 1
        
        if self.debug:
            print(f"[MERGE DEBUG] Found {adjacency_count//2} adjacency relationships")
            # Show first few adjacencies
            for i, (lbl, neighs) in enumerate(list(self.shared.items())[:3]):
                if neighs:
                    print(f"[MERGE DEBUG] Region {lbl} adjacent to: {list(neighs.keys())}")
        
        return self

    def find_triangles(self):
        """Find triangular relationships between regions."""
        triangles = set()
        for a in self.shared:
            neighs = list(self.shared[a])
            for b, c in combinations(neighs, 2):
                if b in self.shared[c] and c in self.shared[b]:
                    triangles.add(tuple(sorted((a, b, c))))
        self.triangles = sorted(triangles)
        return self

    def build_candidate_groups(self):
        """Build candidate groupings for each region."""
        tri_index = defaultdict(list)
        for tri in self.triangles:
            for lbl in tri:
                tri_index[lbl].append(tri)
        
        combos = defaultdict(list)
        for lbl in self.shared:
            # Single region
            combos[lbl].append((lbl,))
            
            # Pairs with neighbors
            for n in self.shared[lbl]:
                if n != lbl:
                    combos[lbl].append((lbl, n))
            
            # Triangles containing this label
            combos[lbl].extend(tri_index[lbl])
            
            # Merged triangles
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
        """Evaluate stage 1 groupings based on shared perimeter ratios."""
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
            
            if self.debug and lbl <= 5:  # Show debug for first few regions
                print(f"[MERGE DEBUG] Region {lbl}: best combo {best_combo} with score {best_score:.4f}")
        
        # Catch "no-combo" labels
        for lbl in self.props:
            if lbl not in best:
                best[lbl] = ((lbl,), 1.0 / self.singleton_penalty)
        
        if self.debug:
            singleton_count = sum(1 for combo, _ in best.values() if len(combo) == 1)
            group_count = len(best) - singleton_count
            print(f"[MERGE DEBUG] Stage 1: {singleton_count} singletons, {group_count} groups")
        
        self.best_stage1 = best
        return self

    def evaluate_stage2(self):
        """Evaluate stage 2 groupings with area/perimeter and distance metrics."""
        # Assign groups and leaders
        group_map = defaultdict(set)
        for lbl, (group, _) in self.best_stage1.items():
            norm = tuple(sorted(int(g) for g in group))
            group_map[norm].add(int(lbl))
        
        group_leaders = {grp: max(grp, key=lambda x: self.areas[x]) for grp in group_map}
        
        if self.debug:
            print(f"[MERGE DEBUG] Stage 2: {len(group_map)} groups identified")
            for i, (grp, members) in enumerate(list(group_map.items())[:5]):
                print(f"[MERGE DEBUG] Group {i}: {grp} -> members {members}, leader {group_leaders[grp]}")
        
        # Stage2 results
        results = {}
        for grp, members in group_map.items():
            leader = group_leaders[grp]
            expanded = set(grp)
            for l in grp:
                expanded.update(self.shared[l].keys())
            expanded = {int(l) for l in expanded if int(l) in self.props}
            
            for lbl in members:
                # Compute score
                tot_area = sum(self.areas[l] for l in expanded)
                tot_perim = sum(self.perims[l] for l in expanded)
                coords = np.vstack([self.props[l].coords for l in expanded])
                com = coords.mean(axis=0)
                
                lbl_coords = self.props[lbl].coords
                if len(lbl_coords) > self.sample_size:
                    idx = np.linspace(0, len(lbl_coords) - 1, self.sample_size).astype(int)
                    lbl_coords = lbl_coords[idx]
                
                dists = np.linalg.norm(lbl_coords - com, axis=1)
                avg_dist = dists.mean()
                score2 = (tot_area / (tot_perim + 1e-8)) / (avg_dist + 1e-8)
                results[lbl] = (leader, score2)
        
        # Add true singletons
        all_lbls = set(self.props.keys())
        for lbl in all_lbls - set(results):
            results[int(lbl)] = (int(lbl), 0.0)
        
        if self.debug:
            leader_counts = defaultdict(int)
            for leader, _ in results.values():
                leader_counts[leader] += 1
            print(f"[MERGE DEBUG] Final leaders: {dict(leader_counts)}")
        
        self.second_stage_results = results
        return self

    def relabel(self):
        """Create the final merged label array."""
        mapping = {lbl: tgt for lbl, (tgt, _) in self.second_stage_results.items()}
        out = np.zeros_like(self.label_img)
        for old, new in mapping.items():
            out[self.label_img == old] = new
        self.merged_label_array = out
        return self

    def run(self):
        """Run the complete merge pipeline."""
        return (self.compute_stats()
                .find_triangles()
                .build_candidate_groups()
                .evaluate_stage1()
                .evaluate_stage2()
                .relabel())

    def plot_initial(self, title="Original Labels"):
        """Plot the initial labeled image."""
        img = label2rgb(self.label_img, bg_label=0)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    def plot_stage1(self, title="Stage 1 Groupings"):
        """Plot stage 1 groupings with annotations."""
        img = label2rgb(self.label_img, bg_label=0)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
        for region in regionprops(self.label_img):
            if region.label == 0:
                continue
            grp, _ = self.best_stage1[region.label]
            txt = ",".join(str(int(x)) for x in grp)
            y, x = region.centroid
            plt.text(x, y, txt, ha='center', va='center', color='white',
                    bbox=dict(facecolor='black', alpha=0.5, lw=0))

    def plot_final(self, title="Final Merged Labels"):
        """Plot the final merged labels."""
        img = label2rgb(self.merged_label_array, bg_label=0)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
        for region in regionprops(self.merged_label_array):
            if region.label == 0:
                continue
            y, x = region.centroid
            plt.text(x, y, str(region.label), ha='center', va='center', color='white',
                    bbox=dict(facecolor='black', alpha=0.5, lw=0))
