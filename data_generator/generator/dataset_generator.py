"""
Simple, unified dataset generator for causal discovery algorithm testing.
Uses SCDG core with configurable parameters and DCDI output format.
"""

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional
from scdg import CausalDataGenerator

from .manufacturing_distributions import ManufacturingDistributionManager
from .temporal_utils import assign_mock_stations
from .utils import save_dataset_with_splits


def max_edges_dag(num_nodes: int, num_roots: int) -> int:
    """Calculate maximum edges for a DAG with given nodes and roots."""
    N, R = num_nodes, num_roots
    if R < 0 or R >= N:
        raise ValueError("Need 0 ≤ R < N")
    return (N * (N - 1) - R * (R - 1)) // 2


def validate_dataset(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate dataset for NaN and Inf values.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for NaN values
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        return False, f"Found NaN values in columns: {nan_cols}"
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        if np.isinf(numeric_cols).any().any():
            inf_cols = numeric_cols.columns[np.isinf(numeric_cols).any()].tolist()
            return False, f"Found Inf values in columns: {inf_cols}"
    
    # Check for all-NaN columns
    if df.isnull().all().any():
        all_nan_cols = df.columns[df.isnull().all()].tolist()
        return False, f"Columns with all NaN values: {all_nan_cols}"
    
    # Check for empty dataframe
    if len(df) == 0:
        return False, "DataFrame is empty"
    
    return True, ""


def generate_single_dataset(
    dataset_idx: int,
    config: Dict[str, Any],
    seed: int,
    output_dir: str,
    index_file: Optional[str] = None
) -> None:
    """
    Generate a single dataset with specified configuration.
    
    Args:
        dataset_idx: Index of this dataset
        config: Configuration dictionary with all parameters
        seed: Random seed (will be offset by dataset_idx for uniqueness)
        output_dir: Directory to save dataset
        index_file: Path to index.csv file
    """
    # Create dataset-specific seed
    dataset_seed = seed + dataset_idx
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)
    rng = np.random.default_rng(dataset_seed)
    
    # Extract configuration parameters
    nodes_min, nodes_max = config.get('nodes_range', [10, 20])
    root_pct_min, root_pct_max = config.get('root_percentage_range', [0.10, 0.30])
    edge_density_min, edge_density_max = config.get('edge_density_range', [0.30, 0.80])
    samples_min, samples_max = config.get('samples_range', [1000, 50000])
    
    # Generate graph parameters
    num_nodes = rng.integers(nodes_min, nodes_max + 1)
    
    # Calculate root nodes
    min_roots = max(1, int(num_nodes * root_pct_min))
    max_roots = max(min_roots, int(num_nodes * root_pct_max))
    max_roots = min(max_roots, num_nodes - 1)
    num_roots = rng.integers(min_roots, max_roots + 1)
    
    # Calculate edges based on density
    min_edges = num_roots + (num_nodes - num_roots - 1)
    max_edges = max_edges_dag(num_nodes, num_roots)
    
    target_density = rng.uniform(edge_density_min, edge_density_max)
    num_edges = int(target_density * max_edges)
    num_edges = max(min_edges, min(num_edges, max_edges))
    
    # Determine relationship type
    relationship_mix = config.get('relationship_mix', {'linear': 0.6, 'nonlinear': 0.3, 'mixed': 0.1})
    rel_choice = rng.choice(list(relationship_mix.keys()), p=list(relationship_mix.values()))
    equation_type_map = {'linear': 'linear', 'nonlinear': 'non_linear', 'mixed': 'random'}
    equation_type = equation_type_map.get(rel_choice, 'linear')
    
    # Determine sample size
    num_samples = rng.integers(samples_min, samples_max + 1)
    
    # Generate DAG
    cdg = CausalDataGenerator(num_samples=num_samples, seed=dataset_seed)
    G, roots = cdg.generate_random_graph(num_nodes, num_roots, num_edges)
    
    # Assign distributions
    dist_manager = ManufacturingDistributionManager(config, seed=dataset_seed)
    manufacturing_distributions = dist_manager.assign_manufacturing_distributions(roots)
    
    # Convert to SCDG format
    scdg_distributions = {}
    for node, dist_info in manufacturing_distributions.items():
        if dist_info['dist'] in ['normal', 'truncated_normal', 'lognormal']:
            scdg_distributions[node] = dist_info
        elif dist_info['dist'] == 'categorical':
            scdg_distributions[node] = {'dist': 'categorical', 'num_classes': dist_info['num_classes']}
        elif dist_info['dist'] == 'categorical_non_uniform':
            scdg_distributions[node] = {
                'dist': 'categorical_non_uniform',
                'num_classes': dist_info['num_classes'],
                'probabilities': dist_info['probabilities']
            }
    
    # Generate data
    cdg.G = G.copy()
    cdg.root_nodes = set(roots)
    df = cdg.generate_data_pipeline(
        total_nodes=num_nodes,
        root_nodes=num_roots,
        edges=num_edges,
        equation_type=equation_type,
        root_distributions_override=scdg_distributions
    )
    
    # Validate dataset for NaN and Inf values
    is_valid, error_msg = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {error_msg}")
    
    # Extract temporal/station info if requested
    extract_stations = config.get('extract_station_info', True)
    num_stations = config.get('num_stations', 3)
    
    if extract_stations:
        topo_nodes = list(nx.topological_sort(G))
        raw_assignment = assign_mock_stations(topo_nodes, num_stations=num_stations, graph=G)
        station_map = {node: raw_assignment[node].split('_')[0] for node in raw_assignment}
        ordered_stations = sorted(set(station_map.values()), key=lambda s: int(s.replace("Station", "")))
        station_to_nodes = {s: [] for s in ordered_stations}
        for n in topo_nodes:
            station_to_nodes[station_map[n]].append(n)
        station_blocks = [station_to_nodes[s] for s in ordered_stations]
        temporal_order = [n for block in station_blocks for n in block]
    else:
        temporal_order = list(G.nodes())
        station_map = {}
        station_blocks = []
        ordered_stations = []
    
    # Build adjacency matrix
    adj_matrix = nx.to_numpy_array(G, nodelist=temporal_order, dtype=int)
    
    # Build metadata
    categorical_nodes = [
        node for node, dist_info in manufacturing_distributions.items()
        if dist_info['dist'] in ['categorical', 'categorical_non_uniform']
    ]
    
    # Convert roots set to sorted list for metadata
    root_node_names = sorted(list(roots))
    
    metadata = {
        'num_nodes': num_nodes,
        'num_roots': num_roots,
        'root_nodes': root_node_names,  # List of root node names
        'num_edges': num_edges,
        'num_samples': num_samples,
        'equation_type': equation_type,
        'temporal_order': temporal_order,
        'station_map': station_map,
        'station_blocks': station_blocks,
        'station_names': ordered_stations,
        'manufacturing_distributions': manufacturing_distributions,
        'root_percentage': (num_roots / num_nodes) * 100,
        'categorical_root_percentage': (len(categorical_nodes) / num_roots * 100) if num_roots > 0 else 0,
        'seed': dataset_seed,
        'config': config
    }
    
    # Save dataset
    base_name = f"dataset_{dataset_idx:03d}"
    dataset_dir = os.path.join(output_dir, base_name)
    
    save_dataset_with_splits(
        df, adj_matrix, metadata, dataset_dir, base_name,
        index_file or os.path.join(output_dir, 'index.csv'),
        config.get('train_ratio', 0.8)
    )


def generate_all_datasets(config: Dict[str, Any]) -> None:
    """
    Generate all datasets according to configuration.
    
    Args:
        config: Configuration dictionary with all parameters
    """
    num_datasets = config.get('num_datasets', 100)
    output_dir = config.get('output_dir', 'causal_meta_dataset')
    seed = config.get('seed', 42)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    index_file = os.path.join(output_dir, 'index.csv')
    
    print(f"Starting generation of {num_datasets} datasets")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i in range(num_datasets):
        try:
            generate_single_dataset(i, config, seed, output_dir, index_file)
            success_count += 1
            print(f"[OK] [{i+1}/{num_datasets}] Generated dataset_{i:03d}")
        except ValueError as e:
            # These are validation errors (NaN/Inf)
            skip_count += 1
            print(f"[SKIP] [{i+1}/{num_datasets}] dataset_{i:03d} - {e}")
            continue
        except Exception as e:
            # Other unexpected errors
            error_count += 1
            print(f"[ERROR] [{i+1}/{num_datasets}] dataset_{i:03d} - {e}")
            continue
    
    print("-" * 60)
    print(f"Results:")
    print(f"  Successfully generated: {success_count}/{num_datasets}")
    print(f"  Skipped (NaN/Inf): {skip_count}/{num_datasets}")
    print(f"  Errors: {error_count}/{num_datasets}")
    print(f"Index file: {index_file}")

