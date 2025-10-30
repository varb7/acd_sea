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
import pickle
from typing import Dict, List

import yaml
import numpy as np
import networkx as nx

try:
    # Local imports when running from data_generator root
    from generator.csuite2 import generate_csuite2_dataset
    from generator.utils import save_dataset_with_splits
except ImportError:
    # Absolute fallback
    from data_generator.generator.csuite2 import generate_csuite2_dataset
    from data_generator.generator.utils import save_dataset_with_splits


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
    save_dataset_with_splits(df, adj, metadata, str(ddir), base, index_file=str(index_file), train_ratio=train_ratio)


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


def main():
    parser = argparse.ArgumentParser(description="Generate CSuite dataset grid from YAML config")
    parser.add_argument('-c', '--config', type=str, default=str(Path(__file__).with_name('experiment_grid.yaml')),
                        help='Path to experiment grid YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Seed baseline for reproducibility chain; individual seeds derived per dataset
    if 'seed' not in cfg:
        cfg['seed'] = 42

    run_phase1(cfg)
    run_phase2(cfg)

    print("CSuite experiment grid generation complete.")


if __name__ == '__main__':
    main()

