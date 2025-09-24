"""Strategies package for dataset generation."""

from .csuite import generate_csuite_meta_dataset  # re-export for convenience
from .dynamic import (
    iter_random_configs,
    iter_preset_variations,
    iter_gradient_configs,
    iter_mixed_configs,
)

__all__ = [
    "generate_csuite_meta_dataset",
    "iter_random_configs",
    "iter_preset_variations",
    "iter_gradient_configs",
    "iter_mixed_configs",
]


