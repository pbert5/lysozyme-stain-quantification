
import pytest
import numpy as np
from .tools.demo_crypt_image_generator import generate_test_crypts
from src.lysozyme_stain_quantification.crypts.remove_edge_touching_regions_mod import remove_edge_touching_regions_sk as remove_edge_touching_regions
from src.lysozyme_stain_quantification.crypts.identify_potential_crypts_mod import identify_potential_crypts

num_crypts=5

def test_RETR():
    red, blue, _ = generate_test_crypts(num_crypts=num_crypts, image_size=(200, 300), radius_px=10, positions_override={0: 0}, tissue_bottom_offset_r=3)
    potential_crypts = identify_potential_crypts(red, blue, debug=False)
    cleaned_up = remove_edge_touching_regions(potential_crypts)
    assert cleaned_up is not None
    assert isinstance(cleaned_up, np.ndarray)
    assert cleaned_up.shape == (200, 300)
    assert cleaned_up.max() == 4  # should have removed some regions
    assert cleaned_up.min() == 0  # background