import numpy as np
from typing import Dict, Tuple, List, Optional, Sequence, Union
from .tools.demo_crypt_image_generator import generate_test_crypts

import pytest

@pytest.fixture
def simple_crypt_image() -> Dict[str, np.ndarray]:
    red, blue, _ = generate_test_crypts(num_crypts=5, image_size=(200, 200))
    return {"red": red, "blue": blue}