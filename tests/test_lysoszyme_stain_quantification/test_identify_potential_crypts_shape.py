import pytest
import numpy as np
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts
from .tools.demo_crypt_image_generator import generate_test_crypts



num_crypts=5




from src.lysozyme_stain_quantification.crypts.identify_potential_crypts import identify_potential_crypts

def test_identify_potential_crypts_shape():
    red, blue, _ = generate_test_crypts(num_crypts=num_crypts, image_size=(200, 200))

    potential_crypts = identify_potential_crypts(red, blue, blob_size_px=15, debug=False)
    assert potential_crypts is not None
    assert isinstance(potential_crypts, np.ndarray)
    assert potential_crypts.max() == num_crypts
    assert potential_crypts.shape == (200, 200)

def test_IPC_seperates_touching_crypts():
    red, blue, _ = generate_test_crypts(num_crypts=3, image_size=(100, 100), positions_override={0: 1, 1:1.5})

    potential_crypts = identify_potential_crypts(red, blue, blob_size_px=15, debug=False)
    assert potential_crypts is not None
    assert isinstance(potential_crypts, np.ndarray)
    assert potential_crypts.max() == 3
    assert potential_crypts.shape == (100, 100)


