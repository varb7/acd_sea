"""
Run registered causal discovery algorithms on generated CSuite datasets.

Usage:
  python -m inference_pipeline.run_experiments \
    --index csuite_grid_datasets/index.csv \
    --results causal_discovery_results2/experiment_results.csv \
    --with-prior true

This script deduplicates datasets by directory from the index, loads the full
data (data.npy), graph (graph.npy) and metadata (*_meta.pkl), and runs all
registered algorithms twice: without prior and with prior knowledge derived
from metadata. Results are appended to a single CSV.
"""

from pathlib import Path
import argparse
import os
import pickle
import glob
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from utils.algorithms import AlgorithmRegistry, compute_metrics
    from utils.prior_knowledge import format_prior_knowledge_for_algorithm
except ImportError:
    from inference_pipeline.utils.algorithms import AlgorithmRegistry, compute_metrics
    from inference_pipeline.utils.prior_knowledge import format_prior_knowledge_for_algorithm


def _resolve_dataset_dir(index_fp: Path, rel_or_abs_graph_path: str) -> Path:
    """
    Resolve a dataset directory given an index file path and the fp_graph value.
    Tries multiple bases:
      1) Absolute path (if provided)
      2) Relative to index directory
      3) Relative to parent of index directory (matches save_dataset_with_splits base_root)
    Returns the first candidate that contains graph.npy and data.npy.
    """
    p = Path(rel_or_abs_graph_path)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p.parent)
    else:
        index_dir = index_fp.parent
        candidates.append((index_dir / p).parent)
        candidates.append((index_dir.parent / p).parent)

    for d in candidates:
        try:
            if (d / 'graph.npy').exists() and (d / 'data.npy').exists():
                return d
        except Exception:
            continue

    # Fallback to the first constructed candidate
    return candidates[0]


def find_dataset_dirs_from_index(index_fp: Path) -> List[Path]:
    df = pd.read_csv(index_fp)
    if 'fp_graph' not in df.columns:
        raise ValueError("index.csv must contain 'fp_graph' column")

    dirs = []
    for p in df['fp_graph'].dropna().unique():
        d = _resolve_dataset_dir(index_fp, str(p))
        dirs.append(d)

    # Deduplicate
    uniq: List[Path] = []
    seen = set()
    for d in dirs:
        rp = str(Path(d).resolve())
        if rp not in seen:
            uniq.append(Path(d))
            seen.add(rp)
    return uniq


def load_metadata(dataset_dir: Path) -> Dict:
    cands = list(dataset_dir.glob("*_meta.pkl"))
    if not cands:
        raise FileNotFoundError(f"No *_meta.pkl found in {dataset_dir}")
    with open(cands[0], 'rb') as f:
        return pickle.load(f)


def run_on_dataset(dataset_dir: Path, registry: AlgorithmRegistry, use_prior: bool) -> List[Dict]:
    data_fp = dataset_dir / 'data.npy'
    graph_fp = dataset_dir / 'graph.npy'
    meta = load_metadata(dataset_dir)

    data = np.load(data_fp)
    cols = list(meta.get('temporal_order', [f'v{i}' for i in range(data.shape[1])]))
    df = pd.DataFrame(data, columns=cols)
    true_adj = np.load(graph_fp)
    max_edges = true_adj.size

    # Log high-level dataset info once
    try:
        print(f"\n=== DATASET ===")
        print(f"Path: {dataset_dir}")
        print(f"Pattern: {meta.get('pattern')} | nodes: {meta.get('num_nodes')} | edges: {meta.get('num_edges')} | samples: {meta.get('num_samples')} | seed: {meta.get('seed')}")
        print(f"Prior knowledge: {'ENABLED' if bool(use_prior) else 'DISABLED'}")
    except Exception:
        # Keep robust even if meta is missing some fields
        print(f"\n=== DATASET ===\nPath: {dataset_dir}")

    algos = registry.list_algorithms()
    results = []
    for algo in algos:
        # Per-algorithm log line
        try:
            print(f"[RUN] Algorithm: {algo} | Dataset: {dataset_dir.name} | Prior: {'yes' if bool(use_prior) else 'no'}")
        except Exception:
            print(f"[RUN] Algorithm: {algo} | Prior: {'yes' if bool(use_prior) else 'no'}")
        prior = None
        if use_prior:
            try:
                prior = format_prior_knowledge_for_algorithm(meta, algo)
            except Exception:
                prior = None
        try:
            res = registry.run_algorithm(algo, df.values, cols,
                                         use_prior_knowledge=use_prior,
                                         prior_knowledge=prior)
        except Exception as e:
            print(f"[ERROR] Algorithm {algo} failed on {dataset_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            res = None
        if res is None:
            metrics = {'shd': None, 'normalized_shd': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'pred_edge_count': 0}
            exec_t = 0.0
            pre_t = 0.0
            post_t = 0.0
        else:
            metrics = compute_metrics(res.adjacency_matrix, true_adj, max_edges)
            exec_t = res.execution_time
            pre_t = res.preprocessing_time
            post_t = res.postprocessing_time

        row = {
            'dataset_dir': str(dataset_dir).replace('\\', '/'),
            'algorithm': algo,
            'use_prior': bool(use_prior),
            'pattern': meta.get('pattern'),
            'num_nodes': meta.get('num_nodes'),
            'num_edges': meta.get('num_edges'),
            'num_samples': meta.get('num_samples'),
            'seed': meta.get('seed'),
            'equation_type': meta.get('equation_type_override', meta.get('equation_type')),
            'var_type_tag': meta.get('var_type_tag', 'unknown'),
            **metrics,
            'execution_time': exec_t,
            'preprocessing_time': pre_t,
            'postprocessing_time': post_t,
        }
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description='Run causal discovery algorithms over generated CSuite datasets')
    parser.add_argument('--index', type=str, required=True, help='Path to dataset index.csv')
    parser.add_argument('--results', type=str, default='causal_discovery_results2/experiment_results.csv', help='Output results CSV')
    parser.add_argument('--with-prior', type=str, default='true', help='Also run with prior knowledge (true/false)')
    args = parser.parse_args()

    index_fp = Path(args.index)
    results_fp = Path(args.results)
    run_with_prior = str(args.with_prior).lower() in ('1', 'true', 'yes', 'y')

    dataset_dirs = find_dataset_dirs_from_index(index_fp)
    results_rows: List[Dict] = []

    registry = AlgorithmRegistry()

    for d in dataset_dirs:
        results_rows.extend(run_on_dataset(d, registry, use_prior=False))
        if run_with_prior:
            results_rows.extend(run_on_dataset(d, registry, use_prior=True))

    # Write/append results
    results_df = pd.DataFrame(results_rows)
    results_fp.parent.mkdir(parents=True, exist_ok=True)
    if results_fp.exists():
        prev = pd.read_csv(results_fp)
        results_df = pd.concat([prev, results_df], ignore_index=True)
    results_df.to_csv(results_fp, index=False)

    print(f"Wrote results: {results_fp}")


if __name__ == '__main__':
    main()
