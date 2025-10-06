
import numpy as np


from .crypts.identify_potential_crypts_mod import identify_potential_crypts
from .crypts.remove_edge_touching_regions_mod import remove_edge_touching_regions_sk
from .crypts.scoring_selector_mod import scoring_selector
# new style:
def segment_crypts(
        channels: tuple[np.ndarray, np.ndarray], # i.e. [crypt_stain_channel, tissue_counterstain_channel] i.e. [RFP, DAPI]
        blob_size_px: int = 15, # approximate size of crypts in pixels (length)
        debug: bool = False,
        scoring_weights: dict[str, float] | None = None,
        masks: list[np.ndarray] | None = None) -> np.ndarray:
    """Segment crypts in the given image.

    Args:
        channels: Tuple of two 2D numpy arrays representing the red and blue channels of the image.
        blob_size_px: Approximate size of crypts in pixels (length).
    """
    scoring_weights = scoring_weights if scoring_weights is not None else {
        'circularity': 0.35,    # Most important - want circular regions
        'area': 0.25,           # Second - want consistent sizes
        'line_fit': 0.15,       # Moderate - want aligned regions
        'red_intensity': 0.15,  # Moderate - want bright regions
        'com_consistency': 0.10 # Least - center consistency
    }
    crypt_img, tissue_image = channels
    if crypt_img.shape != tissue_image.shape:
        raise ValueError(f"Shape mismatch: red {crypt_img.shape} vs blue {tissue_image.shape}")
    potential_crypts = identify_potential_crypts(crypt_img, tissue_image, blob_size_px, debug)

    cleaned_crypts = remove_edge_touching_regions_sk(potential_crypts)

    best_crypts, crypt_scores = scoring_selector(
        cleaned_crypts,
        crypt_img,
        debug=debug,
        max_regions=debug,
        weights=scoring_weights,
        return_details=True,
    )
    
    return best_crypts
