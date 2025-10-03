
import pytest
import numpy as np
from .tools.demo_crypt_image_generator import generate_test_crypts
from src.lysozyme_stain_quantification.segment_crypts import segment_crypts




num_crypts=5

red, blue, _ = generate_test_crypts(num_crypts=num_crypts, image_size=(200, 200))





def test_segment_crypts_runs_at_all():
    segmented = segment_crypts([red, blue])
    assert segmented is not None
    assert isinstance(segmented, np.ndarray)
    assert segmented.shape == (200, 200)





