"""Dynamic configuration strategies.

These functions yield batches of DataGeneratorConfig instances according to
different strategies (random, preset_variations, gradient, mixed).
"""

from typing import Iterable, Dict, Any, Optional

from ...config_schema import DataGeneratorConfig
from ..dynamic_config_generator import (
    generate_random_configurations,
    generate_preset_variations,
    generate_gradient_configurations,
    generate_mixed_configurations,
)


def iter_random_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    for cfg_dict in generate_random_configurations(total=total, seed=seed):
        yield DataGeneratorConfig.from_dict(cfg_dict)


def iter_preset_variations(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    for cfg_dict in generate_preset_variations(total=total, seed=seed):
        yield DataGeneratorConfig.from_dict(cfg_dict)


def iter_gradient_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    for cfg_dict in generate_gradient_configurations(total=total, seed=seed):
        yield DataGeneratorConfig.from_dict(cfg_dict)


def iter_mixed_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    for cfg_dict in generate_mixed_configurations(total=total, seed=seed):
        yield DataGeneratorConfig.from_dict(cfg_dict)


