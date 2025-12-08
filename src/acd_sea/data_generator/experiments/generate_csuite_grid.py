"""
Generate CSuite datasets for a predefined experiment grid.

Reads data_generator/experiments/experiment_grid.yaml and creates datasets
under the configured output directory, writing an index.csv for all datasets.

Each dataset is saved with train/test splits and DCDI bundle using
save_dataset_with_splits. Metadata is augmented with generation tags
(equation_type_override, var_type_tag, root/nonroot categorical controls)
to support downstream analysis.
"""

from pathlib import Path
import argparse
import os
import sys
import pickle
from typing import Dict, List, Any

import yaml
import numpy as np
import networkx as nx

# Add parent directories to path to support imports from different working directories
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent  # Go up from experiments/ to data_generator/ to root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Now try imports - this should work from any directory
try:
    # Try absolute import first (when run from project root)
    from data_generator.generator.csuite2 import generate_csuite2_dataset
    from data_generator.generator.utils import save_dataset_with_splits
except ImportError:
    # Fallback to relative import (when run from data_generator directory)
    from generator.csuite2 import generate_csuite2_dataset
    from generator.utils import save_dataset_with_splits


def valid_patterns_for_n(n: int, patterns: List[str]) -> List[str]:
    """Filter patterns according to their minimum node requirements."""
    min_req = {
        'chain': 2,
        'collider': 3,
        'backdoor': 3,
        'weak_arrow': 3,
        'large_backdoor': 4,
        'mixed_confounding': 4,
    }
    out = []
    for p in patterns:
        if n >= min_req.get(p, 2):
            out.append(p)
    return out


def build_var_type_config(tag: str, mixed_cfg: Dict) -> Dict:
    if tag == 'mixed':
        return {
            'root_categorical': bool(mixed_cfg.get('root_categorical', True)),
            'root_num_classes': int(mixed_cfg.get('root_num_classes', 3)),
            'nonroot_categorical_pct': float(mixed_cfg.get('nonroot_categorical_pct', 0.3)),
            'nonroot_categorical_num_classes': int(mixed_cfg.get('nonroot_categorical_num_classes', 3)),
        }
    # continuous default
    return {
        'root_categorical': False,
        'root_num_classes': 3,
        'nonroot_categorical_pct': 0.0,
        'nonroot_categorical_num_classes': 3,
    }


def save_one_dataset(df, metadata, G, out_dir: Path, base: str, index_file: Path, train_ratio: float):
    # adjacency ordered by temporal order
    ordered = metadata['temporal_order']
    adj = nx.to_numpy_array(G, nodelist=ordered, dtype=int)
    ddir = out_dir / base
    # Ensure directory exists and use absolute path
    ddir.mkdir(parents=True, exist_ok=True)
    save_dataset_with_splits(df, adj, metadata, str(ddir.resolve()), base, index_file=str(index_file), train_ratio=train_ratio)


def run_phase1(cfg: Dict):
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / 'index.csv'
    train_ratio = float(cfg.get('train_ratio', 0.8))
    num_stations = int(cfg.get('num_stations', 3))

    p1 = cfg['phase1']
    sizes = p1['sizes']
    patterns = p1['patterns']
    samples_list = p1['samples']
    seeds = int(p1['seeds']) if isinstance(p1['seeds'], int) else int(p1['seeds'])
    var_types = p1.get('var_types', ['continuous'])
    override_equation = bool(p1.get('override_equation', False))

    for n in sizes:
        for pattern in valid_patterns_for_n(int(n), patterns):
            for num_samples in samples_list:
                for seed in range(seeds):
                    for vt in var_types:
                        vt_cfg = build_var_type_config(vt, cfg.get('phase2', {}).get('mixed_config', {}))
                        gen_cfg = {
                            'pattern': pattern,
                            'num_nodes': int(n),
                            'num_samples': int(num_samples),
                            'seed': int(cfg.get('seed', 42)) + seed,
                            'num_stations': num_stations,
                            **vt_cfg,
                        }
                        # Do not override equation type in Phase 1 (use pattern default)
                        if override_equation:
                            gen_cfg['equation_type'] = 'linear'

                        df, metadata, G = generate_csuite2_dataset(gen_cfg)

                        # augment metadata
                        metadata = {**metadata,
                                    'seed': gen_cfg['seed'],
                                    'equation_type_override': gen_cfg.get('equation_type', metadata.get('equation_type')),
                                    'var_type_tag': vt,
                                    'root_categorical': vt_cfg['root_categorical'],
                                    'nonroot_categorical_pct': vt_cfg['nonroot_categorical_pct']}

                        base = f"csuite_{pattern}_{int(n)}n_p1_{vt}_{int(num_samples)}_{gen_cfg['seed']:04d}"
                        save_one_dataset(df, metadata, G, out_dir, base, index_file, train_ratio)


def run_phase2(cfg: Dict):
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / 'index.csv'
    train_ratio = float(cfg.get('train_ratio', 0.8))
    num_stations = int(cfg.get('num_stations', 3))

    p2 = cfg['phase2']
    sizes = p2['sizes']
    patterns = p2['patterns']
    samples_list = p2['samples']
    seeds = int(p2['seeds']) if isinstance(p2['seeds'], int) else int(p2['seeds'])
    var_types = p2.get('var_types', ['continuous', 'mixed'])
    eq_by_pattern = p2.get('equation_types_by_pattern', {'default': ['linear', 'non_linear']})

    for n in sizes:
        for pattern in valid_patterns_for_n(int(n), patterns):
            eq_types = eq_by_pattern.get(pattern, eq_by_pattern.get('default', ['linear', 'non_linear']))
            for eq_type in eq_types:
                for vt in var_types:
                    vt_cfg = build_var_type_config(vt, p2.get('mixed_config', {}))
                    for num_samples in samples_list:
                        for seed in range(seeds):
                            gen_cfg = {
                                'pattern': pattern,
                                'num_nodes': int(n),
                                'num_samples': int(num_samples),
                                'seed': int(cfg.get('seed', 42)) + seed,
                                'num_stations': num_stations,
                                'equation_type': eq_type,
                                **vt_cfg,
                            }
                            df, metadata, G = generate_csuite2_dataset(gen_cfg)

                            # augment metadata
                            metadata = {**metadata,
                                        'seed': gen_cfg['seed'],
                                        'equation_type_override': gen_cfg.get('equation_type', metadata.get('equation_type')),
                                        'var_type_tag': vt,
                                        'root_categorical': vt_cfg['root_categorical'],
                                        'nonroot_categorical_pct': vt_cfg['nonroot_categorical_pct']}

                            base = f"csuite_{pattern}_{int(n)}n_p2_{eq_type}_{vt}_{int(num_samples)}_{gen_cfg['seed']:04d}"
                            save_one_dataset(df, metadata, G, out_dir, base, index_file, train_ratio)


def _sanitize_token(token: str) -> str:
    return (
        str(token)
        .replace('.', 'p')
        .replace('-', 'm')
        .replace(' ', '')
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def describe_root_distribution(root_dist: Dict) -> str:
    dtype = root_dist.get('type', 'unknown')
    params = root_dist.get('params', {})
    token = dtype

    if dtype == 'normal':
        std = _to_float(params.get('std', 1.0), 1.0)
        mean = _to_float(params.get('mean', 0.0), 0.0)
        var_level = 'low' if std <= 1.0 else 'high'
        mean_level = 'pos' if mean > 0.5 else 'neg' if mean < -0.5 else 'zero'
        token = f"{dtype}_{var_level}var_{mean_level}mean"
    elif dtype == 'uniform':
        low = _to_float(params.get('low', 0.0), 0.0)
        high = _to_float(params.get('high', 1.0), 1.0)
        span = abs(high - low)
        span_level = 'low' if span <= 6 else 'high'
        center = (high + low) / 2
        center_level = 'pos' if center > 0.5 else 'neg' if center < -0.5 else 'zero'
        token = f"{dtype}_{span_level}span_{center_level}center"
    elif dtype == 'exponential':
        scale = _to_float(params.get('scale', 1.0), 1.0)
        scale_level = 'low' if scale <= 1.0 else 'high'
        token = f"{dtype}_{scale_level}variance"
    elif dtype == 'beta':
        a = _to_float(params.get('a', 1.0), 1.0)
        b = _to_float(params.get('b', 1.0), 1.0)
        balance = 'balanced' if abs(a - b) <= 0.5 else ('left' if a > b else 'right')
        spread = 'low' if a + b >= 4 else 'high'
        token = f"{dtype}_{balance}_{spread}var"
    else:
        extra = "_".join(f"{k}{_sanitize_token(v)}" for k, v in sorted(params.items()))
        token = f"{dtype}_{extra}" if extra else dtype

    return _sanitize_token(token)


def describe_noise_config(noise_cfg: Dict) -> str:
    dtype = noise_cfg.get('type', 'unknown')
    params = noise_cfg.get('params', {})
    token = dtype

    if dtype == 'normal':
        std = _to_float(params.get('std', 1.0), 1.0)
        level = 'low' if std <= 1.0 else 'high'
        token = f"{dtype}_{level}noise"
    elif dtype == 'uniform':
        low = _to_float(params.get('low', -1.0), -1.0)
        high = _to_float(params.get('high', 1.0), 1.0)
        span = abs(high - low)
        level = 'low' if span <= 2 else 'high'
        token = f"{dtype}_{level}noise"
    elif dtype == 'exponential':
        scale = _to_float(params.get('scale', 1.0), 1.0)
        level = 'low' if scale <= 1.0 else 'high'
        token = f"{dtype}_{level}noise"
    elif dtype == 'beta':
        a = _to_float(params.get('a', 1.0), 1.0)
        b = _to_float(params.get('b', 1.0), 1.0)
        level = 'low' if (a + b) >= 7 else 'high'
        token = f"{dtype}_{level}noise"
    else:
        extra = "_".join(f"{k}{_sanitize_token(v)}" for k, v in sorted(params.items()))
        token = f"{dtype}_{extra}" if extra else dtype

    return _sanitize_token(token)


def run_phase3(cfg: Dict):
    """Phase 3: Variations in noise parameters and root node distributions."""
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / 'index.csv'
    train_ratio = float(cfg.get('train_ratio', 0.8))
    num_stations = int(cfg.get('num_stations', 3))

    p3 = cfg['phase3']
    sizes = p3['sizes']
    patterns = p3['patterns']
    samples_list = p3['samples']
    seeds = int(p3['seeds']) if isinstance(p3['seeds'], int) else int(p3['seeds'])
    var_types = p3.get('var_types', ['continuous'])
    equation_types = p3.get('equation_types', ['linear', 'non_linear'])
    root_distributions = p3.get('root_distributions', [])
    noise_configs = p3.get('noise_configs', [])
    edge_densities = p3.get('edge_densities', [0.5])

    for n in sizes:
        for pattern in valid_patterns_for_n(int(n), patterns):
            for density in edge_densities:
                for eq_type in equation_types:
                    for vt in var_types:
                        vt_cfg = build_var_type_config(vt, cfg.get('phase2', {}).get('mixed_config', {}))
                        for root_dist in root_distributions:
                            for noise_cfg in noise_configs:
                                for num_samples in samples_list:
                                    for seed in range(seeds):
                                        gen_cfg = {
                                            'pattern': pattern,
                                            'num_nodes': int(n),
                                            'num_samples': int(num_samples),
                                            'seed': int(cfg.get('seed', 42)) + seed,
                                            'num_stations': num_stations,
                                            'equation_type': eq_type,
                                            'root_distribution_type': root_dist['type'],
                                            'root_distribution_params': root_dist.get('params', {}),
                                            'default_noise_type': noise_cfg['type'],
                                            'default_noise_params': noise_cfg.get('params', {}),
                                            'edge_density': density,
                                            **vt_cfg,
                                        }
                                        
                                        df, metadata, G = generate_csuite2_dataset(gen_cfg)

                                        # augment metadata with Phase 3 specific information
                                        metadata = {**metadata,
                                                    'seed': gen_cfg['seed'],
                                                    'equation_type_override': eq_type,
                                                    'var_type_tag': vt,
                                                    'root_categorical': vt_cfg['root_categorical'],
                                                    'nonroot_categorical_pct': vt_cfg['nonroot_categorical_pct'],
                                                    'root_distribution_type': root_dist['type'],
                                                    'root_distribution_params': str(root_dist.get('params', {})),
                                                    'noise_type': noise_cfg['type'],
                                                    'noise_params': str(noise_cfg.get('params', {}))}

                                        # Create descriptive base name
                                        root_dist_tag = describe_root_distribution(root_dist)
                                        noise_tag = describe_noise_config(noise_cfg)
                                        density_tag = f"_d{int(density*100)}" if pattern == 'random_dag' else ""
                                        base = f"csuite_{pattern}_{int(n)}n_p3{density_tag}_{eq_type}_{vt}_{root_dist_tag}_{noise_tag}_{int(num_samples)}_{gen_cfg['seed']:04d}"
                                        save_one_dataset(df, metadata, G, out_dir, base, index_file, train_ratio)


def main():
    parser = argparse.ArgumentParser(description="Generate CSuite dataset grid from YAML config")
    parser.add_argument('-c', '--config', type=str, default=str(Path(__file__).with_name('experiment_grid.yaml')),
                        help='Path to experiment grid YAML')
    args = parser.parse_args()

    # Robust YAML load with BOM handling, error capture, and type validation
    try:
        # utf-8-sig strips BOM if present (common on Windows)
        with open(args.config, 'r', encoding='utf-8-sig') as f:
            text = f.read()
        # First try safe_load
        cfg = yaml.safe_load(text)
        # If still None, try full_load as a fallback
        if cfg is None:
            cfg = yaml.full_load(text)
        if cfg is None or not isinstance(cfg, dict):
            details = f"parsed_type={type(cfg).__name__}, length={len(text)}"
            raise ValueError(f"Invalid YAML content in {args.config}: {details}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in {args.config}: {e}")

    # Seed baseline for reproducibility chain; individual seeds derived per dataset
    if 'seed' not in cfg:
        cfg['seed'] = 42

    # Run phases conditionally based on what's in the config
    if 'phase1' in cfg:
        run_phase1(cfg)
    if 'phase2' in cfg:
        run_phase2(cfg)
    if 'phase3' in cfg:
        run_phase3(cfg)

    print("CSuite experiment grid generation complete.")


if __name__ == '__main__':
    main()

