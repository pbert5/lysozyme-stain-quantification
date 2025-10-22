import numpy as np
from skimage.segmentation import clear_border, relabel_sequential
import dask.array as da


def remove_edge_touching_regions_sk(labels, debug=False):
    cleared = clear_border(labels, buffer_size=0, bgval=0)  # drops any border-touching label
    out, _, _ = relabel_sequential(cleared)                # compact to 1..K
    if debug:
        print(f"[DEBUG] regions: {len(np.unique(labels))-1} -> {len(np.unique(out))-1}")
    return out
