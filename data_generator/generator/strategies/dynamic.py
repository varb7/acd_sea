"""Dynamic configuration strategies.

These functions yield batches of DataGeneratorConfig instances according to
different strategies (random, preset_variations, gradient, mixed).
"""

from typing import Iterable, Dict, Any, Optional

try:
    # Prefer absolute imports when package is available
    from data_generator.config_schema import DataGeneratorConfig
    from data_generator.generator.dynamic_config_generator import DynamicConfigGenerator
except ModuleNotFoundError:
    # Fallback for running as a script from within the data_generator directory
    # where sys.path[0] is the data_generator folder.
    from config_schema import DataGeneratorConfig
    from generator.dynamic_config_generator import DynamicConfigGenerator


def _get_base_and_ranges(seed: Optional[int]) -> tuple[DataGeneratorConfig, Dict[str, Any]]:
    """Build a default base config object and generation ranges dict."""
    base_cfg_obj = DataGeneratorConfig()
    gr = base_cfg_obj.generation_ranges
    ranges = {
        "categorical_percentage": (gr.categorical_percentage[0], gr.categorical_percentage[1]),
        "normal_percentage": (gr.normal_percentage[0], gr.normal_percentage[1]),
        "truncated_normal_percentage": (gr.truncated_normal_percentage[0], gr.truncated_normal_percentage[1]),
        "lognormal_percentage": (gr.lognormal_percentage[0], gr.lognormal_percentage[1]),
        "uniform_categorical_percentage": (gr.uniform_categorical_percentage[0], gr.uniform_categorical_percentage[1]),
        "non_uniform_categorical_percentage": (gr.non_uniform_categorical_percentage[0], gr.non_uniform_categorical_percentage[1]),
        "noise_level": (gr.noise_level[0], gr.noise_level[1]),
    }
    return base_cfg_obj, ranges


def _manufacturing_dict_from_config(cfg: DataGeneratorConfig) -> Dict[str, Any]:
    m = cfg.manufacturing
    return {
        "categorical_percentage": m.categorical_percentage,
        "continuous_percentage": m.continuous_percentage,
        "continuous_distributions": dict(m.continuous_distributions),
        "categorical_distributions": dict(m.categorical_distributions),
        "noise_level": m.noise_level,
        "noise_params": dict(m.noise_params),
    }


def _apply_manufacturing_to_config(base: DataGeneratorConfig, m_dict: Dict[str, Any]) -> DataGeneratorConfig:
    """Return a new DataGeneratorConfig with manufacturing fields overridden by m_dict."""
    cfg = DataGeneratorConfig.from_dict(base.to_dict())
    cfg.manufacturing.categorical_percentage = float(m_dict["categorical_percentage"])
    cfg.manufacturing.continuous_percentage = float(m_dict["continuous_percentage"])
    cfg.manufacturing.continuous_distributions = dict(m_dict["continuous_distributions"])
    cfg.manufacturing.categorical_distributions = dict(m_dict["categorical_distributions"])
    cfg.manufacturing.noise_level = float(m_dict["noise_level"])
    # sync std with noise_level if present
    np = dict(m_dict.get("noise_params", {}))
    np["std"] = cfg.manufacturing.noise_level
    cfg.manufacturing.noise_params = np
    return cfg


def iter_random_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    base_cfg_obj, ranges = _get_base_and_ranges(seed)
    base_m = _manufacturing_dict_from_config(base_cfg_obj)
    gen = DynamicConfigGenerator(base_m, ranges, seed=seed)
    for m_dict in gen.generate_config_batch(total):
        yield _apply_manufacturing_to_config(base_cfg_obj, m_dict)


def iter_preset_variations(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    base_cfg_obj, _ = _get_base_and_ranges(seed)
    base_m = _manufacturing_dict_from_config(base_cfg_obj)
    presets: list[Dict[str, Any]] = []
    # Variation 1: more categorical
    v1 = dict(base_m)
    v1["categorical_percentage"] = min(0.40, base_m["categorical_percentage"] * 2.0)
    v1["continuous_percentage"] = 1.0 - v1["categorical_percentage"]
    presets.append(v1)
    # Variation 2: higher noise
    v2 = dict(base_m)
    v2["noise_level"] = min(0.02, base_m["noise_level"] * 2.0)
    v2["noise_params"] = dict(base_m["noise_params"])
    v2["noise_params"]["std"] = v2["noise_level"]
    presets.append(v2)
    # Variation 3: tweak distributions
    v3 = dict(base_m)
    cd = dict(v3["continuous_distributions"])
    total_cont = cd["normal"] + cd["truncated_normal"] + cd["lognormal"]
    if total_cont > 0:
        cd["normal"] = min(0.85, cd["normal"] * 1.1)
        rest = max(1e-9, 1.0 - cd["normal"])
        ratio = (cd["truncated_normal"] + cd["lognormal"]) or 1.0
        cd["truncated_normal"] = rest * (cd["truncated_normal"] / ratio)
        cd["lognormal"] = rest * (cd["lognormal"] / ratio)
    v3["continuous_distributions"] = cd
    presets.append(v3)

    for i in range(min(total, len(presets))):
        yield _apply_manufacturing_to_config(base_cfg_obj, presets[i])


def iter_gradient_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    base_cfg_obj, ranges = _get_base_and_ranges(seed)
    gen = DynamicConfigGenerator(_manufacturing_dict_from_config(base_cfg_obj), ranges, seed=seed)
    gradients = []
    gradients.extend(gen.generate_gradient_configs("categorical_percentage", ranges["categorical_percentage"][0], ranges["categorical_percentage"][1], max(1, total // 3 or 1)))
    gradients.extend(gen.generate_gradient_configs("normal_percentage", ranges["normal_percentage"][0], ranges["normal_percentage"][1], max(1, total // 3 or 1)))
    gradients.extend(gen.generate_gradient_configs("noise_level", ranges["noise_level"][0], ranges["noise_level"][1], max(1, total // 3 or 1)))
    for m_dict in gradients[:total]:
        yield _apply_manufacturing_to_config(base_cfg_obj, m_dict)


def iter_mixed_configs(total: int, seed: Optional[int] = None) -> Iterable[DataGeneratorConfig]:
    base_cfg_obj, ranges = _get_base_and_ranges(seed)
    base_m = _manufacturing_dict_from_config(base_cfg_obj)
    gen = DynamicConfigGenerator(base_m, ranges, seed=seed)
    # Start with a couple of presets
    mixed: list[Dict[str, Any]] = []
    mixed.extend([_manufacturing_dict_from_config(cfg) for cfg in iter_preset_variations(min(3, total), seed=seed)])
    # Fill the rest with randoms
    if len(mixed) < total:
        mixed.extend(gen.generate_config_batch(total - len(mixed)))
    for m_dict in mixed[:total]:
        yield _apply_manufacturing_to_config(base_cfg_obj, m_dict)


