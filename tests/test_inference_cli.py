"""Smoke tests for inference CLI entry points."""

import pytest

from wetlands_ml_geoai import test as test_mask_rcnn
from wetlands_ml_geoai import test_unet


@pytest.mark.parametrize(
    "module",
    [test_mask_rcnn, test_unet],
)
def test_inference_cli_requires_inputs(module):
    with pytest.raises(SystemExit):
        module.parse_args([])

