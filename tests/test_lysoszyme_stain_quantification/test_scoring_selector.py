from src.lysozyme_stain_quantification.crypts.scoring_selector import scoring_selector
from .tools.demo_crypt_image_generator import generate_test_crypts
import pytest

from src.lysozyme_stain_quantification.crypts.identify_potential_crypts import identify_potential_crypts
from src.lysozyme_stain_quantification.crypts.remove_edge_touching_regions import remove_edge_touching_regions

num_crypts = 5

def test_scoring_selector_basic():
    red, blue, _ = generate_test_crypts(num_crypts=num_crypts, image_size=(200, 300), radius_px=10, positions_override={}, tissue_bottom_offset_r=3)
    potential_crypts = identify_potential_crypts(red, blue, debug=False)
    cleaned_crypts = remove_edge_touching_regions(potential_crypts)
    
