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

# Add the src directory to Python path for imports
import sys
from pathlib import Path
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
sys.path.insert(0, str(src_dir))

import argparse
import os
import pickle
import glob
from typing import Dict, List

import numpy as np
import pandas as pd

from acd_sea.inference_pipeline.utils.algorithms import AlgorithmRegistry, compute_metrics
from acd_sea.inference_pipeline.utils.prior_knowledge import format_prior_knowledge_for_algorithm


def _normalize_columns(columns: List[str]) -> List[str]:
    """Ensure all column names are python strings (strip np.str_)."""
    return [str(col) for col in columns]


def _compute_root_nodes_from_adj(adj_matrix: np.ndarray, columns: List[str]) -> List[str]:
    """
    Derive root nodes (zero indegree) directly from the adjacency matrix.

    Args:
        adj_matrix: Square numpy array representing the true DAG (shape n x n).
                   Rows = source nodes, Columns = target nodes.
                   Column j has non-zero entries if node j has incoming edges.
        columns: Column names aligned with adj_matrix ordering (must match temporal_order
                 used when saving the adjacency matrix).

    Returns:
        List of node names that have no incoming edges (all zeros in their column).
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj_matrix.shape}")
    if adj_matrix.shape[0] != len(columns):
        raise ValueError(f"Adjacency matrix dimension ({adj_matrix.shape[0]}) and "
                        f"column list length ({len(columns)}) differ")

    # Check each column: if all entries are zero, that node has no incoming edges (is a root)
    # axis=0 means check down each column
    incoming = np.any(adj_matrix != 0, axis=0)
    root_indices = [i for i, has_parent in enumerate(incoming) if not has_parent]
    return [columns[i] for i in root_indices]


def _sync_root_nodes_with_graph(meta: Dict, adj_matrix: np.ndarray, columns: List[str], dataset_label: str) -> Dict:
    """
    Ensure metadata root_nodes matches what the adjacency implies.

    If metadata differs, store the original list for debugging and overwrite with the
    recomputed roots so downstream prior knowledge remains consistent.
    
    This fixes the issue where pattern templates (collider, backdoor, large_backdoor)
    incorrectly list non-root nodes as roots in the metadata.
    """
    # Normalize stored roots to plain strings (handle np.str_ objects)
    stored_roots_raw = meta.get('root_nodes', [])
    stored_roots = [str(n).strip() for n in stored_roots_raw]
    
    # Compute true roots from adjacency matrix
    computed_roots = _compute_root_nodes_from_adj(adj_matrix, columns)
    
    # Normalize computed roots for comparison
    stored_set = set(stored_roots)
    computed_set = set(computed_roots)
    
    if stored_set != computed_set:
        print(f"[WARN] Root metadata mismatch for {dataset_label}: "
              f"stored={stored_roots}, recomputed={computed_roots}")
        # Create a new dict to avoid mutating cached pickle objects
        meta = dict(meta)
        meta['_root_nodes_original'] = stored_roots_raw  # Keep original for debugging
        meta['root_nodes'] = computed_roots  # Overwrite with correct roots
        print(f"[FIX] Updated root_nodes to {computed_roots}")
    return meta


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
    cols = _normalize_columns(meta.get('temporal_order', [f'v{i}' for i in range(data.shape[1])]))
    df = pd.DataFrame(data, columns=cols)
    true_adj = np.load(graph_fp)
    
    # Fix root_nodes to match actual graph structure (critical for prior knowledge)
    meta = _sync_root_nodes_with_graph(meta, true_adj, cols, dataset_dir.name)
    
    # Also normalize temporal_order in metadata to ensure consistency (handle np.str_ objects)
    if 'temporal_order' in meta and meta['temporal_order']:
        meta['temporal_order'] = _normalize_columns(meta['temporal_order'])
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

        # Extract additional metadata fields for visualizer compatibility
        # For Phase 1 datasets, provide sensible defaults for missing fields
        root_distribution_type = meta.get('root_distribution_type', 'normal')  # Phase 1 uses Gaussian
        noise_type = meta.get('noise_type', 'normal')  # Phase 1 uses Gaussian noise
        edge_density = meta.get('edge_density', 'N/A')
        
        # Calculate variation levels and bias based on available metadata
        root_variation_level = 'low'  # Phase 1 uses standard normal (std=1.0)
        root_mean_bias = 'none'  # Phase 1 uses zero mean
        noise_intensity_level = 'medium'  # Phase 1 uses std=1.0 noise
        
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
            'root_distribution_type': root_distribution_type,
            'root_variation_level': root_variation_level,
            'root_mean_bias': root_mean_bias,
            'noise_type': noise_type,
            'noise_intensity_level': noise_intensity_level,
            'edge_density': edge_density,
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
