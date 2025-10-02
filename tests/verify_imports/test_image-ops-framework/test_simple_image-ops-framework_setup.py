import numpy as np
import pytest


import numpy as np
import xarray as xr
from xarray.testing import assert_equal

from image_ops_framework.analysis_stack_xr import AnalysisStackXR


def test_add_sources_basic(tiny_imgs):
    stk = AnalysisStackXR().add_sources(
        subject=tiny_imgs["subject"],
        sources=[tiny_imgs["dapi"], tiny_imgs["rfp"]],
        sourcenames=["dapi","rfp"]
    )

    assert "source" in stk.ds
    assert set(stk.ds.coords["src"].values) == {"dapi","rfp"}
    assert stk.ds["source"].dims[:2] == ("src","subject")
    assert stk.ds["source"].dtype == np.float32 or stk.ds["source"].dtype == np.float64

    # label/scalar blocks should be initialized (width 0 along lab/sc)
    assert "label" in stk.ds and stk.ds.sizes["lab"] == 0
    assert "scalar" in stk.ds and stk.ds.sizes["sc"] == 0
