"""Smoke tests for training CLI entry points."""

import pytest

from wetlands_ml_geoai import train, train_unet


@pytest.mark.parametrize(
    "module",
    [train, train_unet],
)
def test_training_cli_requires_inputs(module):
    with pytest.raises(SystemExit):
        module.parse_args([])

