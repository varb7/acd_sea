import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scdg import CausalDataGenerator

from .utils import save_dataset, get_equation_type
from .structural_patterns import add_structural_variations
from .temporal_utils import generate_temporal_order_from_stations
from .manufacturing_distributions import ManufacturingDistributionManager

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif 'Tensor' in str(type(obj)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def max_edges_dag(num_nodes: int, num_roots: int) -> int:
    """
    Maximum number of edges in an acyclic digraph with
    `num_nodes` total nodes and `num_roots` roots (in‑degree 0).
    """
    N, R = num_nodes, num_roots
    if R < 0 or R >= N:
        raise ValueError("Need 0\u202f\u2264\u202fR\u202f<\u202fN")
    return (N * (N - 1) - R * (R - 1)) // 2

def generate_meta_dataset(total_datasets=10, output_dir="causal_meta_dataset", index_file=None):
    # Set default index file if not provided
    if index_file is None:
        index_file = os.path.join(output_dir, "index.xlsx")
    
    for i in range(total_datasets):
        try:
            num_nodes = random.randint(3, 4)
            min_root_percentage = 0.10
            max_root_percentage = 0.30
            min_root_nodes = max(1, int(num_nodes * min_root_percentage))
            max_root_nodes = max(min_root_nodes, int(num_nodes * max_root_percentage))
            max_root_nodes = min(max_root_nodes, num_nodes - 1)
            root_nodes = random.randint(min_root_nodes, max_root_nodes)

            # Calculate min and max edges
            min_edges = root_nodes + (num_nodes - root_nodes - 1)
            max_edges = max_edges_dag(num_nodes, root_nodes)
            edges = random.randint(min_edges, max_edges)

            equation_type = get_equation_type(random.choice(["low"]))

            # Generate the DAG structure once
            cdg_struct = CausalDataGenerator(num_samples=500, seed=42)
            G, roots = cdg_struct.generate_random_graph(num_nodes, root_nodes, edges)

            # Paired 1: Categorical root nodes
            cdg_cat = CausalDataGenerator(num_samples=500, seed=42)
            cdg_cat.G = G.copy()
            cdg_cat.root_nodes = set(roots)
            df_cat = cdg_cat.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type,
                categorical_root_nodes="all",
                max_categories=4
            )
            G_cat = cdg_cat.G
            temporal_order_cat = list(G_cat.nodes())
            adj_matrix_cat = nx.to_numpy_array(G_cat, nodelist=temporal_order_cat, dtype=int)
            root_percentage = (root_nodes / num_nodes) * 100
            categorical_root_node_names = list(cdg_cat.root_nodes) if hasattr(cdg_cat, 'root_nodes') else []
            metadata_cat = {
                "temporal_order": temporal_order_cat,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "root_percentage": root_percentage,
                "categorical_root_percentage": 100.0  # All root nodes are categorical in this case
            }
            base_name_cat = f"dataset_{i}_cat"
            dataset_dir_cat = os.path.join(output_dir, base_name_cat)
            save_dataset(df_cat, adj_matrix_cat, metadata_cat, dataset_dir_cat, base_name_cat, index_file, "train")

            # Paired 2: Continuous root nodes
            cdg_cont = CausalDataGenerator(num_samples=500, seed=42)
            cdg_cont.G = G.copy()
            cdg_cont.root_nodes = set(roots)
            df_cont = cdg_cont.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type
                # Do not pass categorical_root_nodes
            )
            G_cont = cdg_cont.G
            temporal_order_cont = list(G_cont.nodes())
            adj_matrix_cont = nx.to_numpy_array(G_cont, nodelist=temporal_order_cont, dtype=int)
            metadata_cont = {
                "temporal_order": temporal_order_cont,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "root_percentage": root_percentage,
                "categorical_root_percentage": 0.0  # No categorical root nodes in continuous dataset
            }
            base_name_cont = f"dataset_{i}_cont"
            dataset_dir_cont = os.path.join(output_dir, base_name_cont)
            save_dataset(df_cont, adj_matrix_cont, metadata_cont, dataset_dir_cont, base_name_cont, index_file, "train")

            print(f"[{i+1}/{total_datasets}] Saved paired datasets: {base_name_cat} (categorical), {base_name_cont} (continuous)")
        except Exception as e:
            print(f"Skipping dataset {i+1} due to error: {e}")

def generate_meta_dataset_with_manufacturing_distributions(
    total_datasets=10, 
    output_dir="causal_meta_dataset", 
    index_file=None,
    manufacturing_config=None
):
    """
    Generate meta dataset using manufacturing-specific distributions including
    truncated normal, lognormal, and categorical distributions.
    
    Args:
        total_datasets: Number of datasets to generate
        output_dir: Output directory for datasets
        index_file: Index file path
        manufacturing_config: Manufacturing configuration dict
    """
    
    # Use default manufacturing config if not provided
    if manufacturing_config is None:
        from config import MANUFACTURING_CONFIG
        manufacturing_config = MANUFACTURING_CONFIG
    
    # Set default index file if not provided
    if index_file is None:
        index_file = os.path.join(output_dir, "index.xlsx")
    
    for i in range(total_datasets):
        try:
            num_nodes = random.randint(3, 20)
            min_root_percentage = 0.10
            max_root_percentage = 0.30
            min_root_nodes = max(1, int(num_nodes * min_root_percentage))
            max_root_nodes = max(min_root_nodes, int(num_nodes * max_root_percentage))
            max_root_nodes = min(max_root_nodes, num_nodes - 1)
            root_nodes = random.randint(min_root_nodes, max_root_nodes)

            # Calculate min and max edges
            min_edges = root_nodes + (num_nodes - root_nodes - 1)
            max_edges = max_edges_dag(num_nodes, root_nodes)
            edges = random.randint(min_edges, max_edges)

            equation_type = get_equation_type(random.choice(["low"]))

            # Generate the DAG structure once
            cdg_struct = CausalDataGenerator(num_samples=500, seed=42)
            G, roots = cdg_struct.generate_random_graph(num_nodes, root_nodes, edges)

            # Paired 1: Manufacturing distributions (mixed continuous and categorical)
            cdg_mfg = CausalDataGenerator(num_samples=500, seed=42)
            cdg_mfg.G = G.copy()
            cdg_mfg.root_nodes = set(roots)
            
            # Use manufacturing distribution manager
            dist_manager = ManufacturingDistributionManager(manufacturing_config, seed=42)
            manufacturing_distributions = dist_manager.assign_manufacturing_distributions(roots)
            
            # Convert to SCDG format
            scdg_distributions = {}
            for node, dist_info in manufacturing_distributions.items():
                if dist_info['dist'] in ['normal', 'truncated_normal', 'lognormal']:
                    scdg_distributions[node] = dist_info
                elif dist_info['dist'] == 'categorical':
                    scdg_distributions[node] = {
                        'dist': 'categorical',
                        'num_classes': dist_info['num_classes']
                    }
                elif dist_info['dist'] == 'categorical_non_uniform':
                    scdg_distributions[node] = {
                        'dist': 'categorical_non_uniform',
                        'num_classes': dist_info['num_classes'],
                        'probabilities': dist_info['probabilities']
                    }
            
            df_mfg = cdg_mfg.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type,
                root_distributions_override=scdg_distributions
            )
            
            G_mfg = cdg_mfg.G
            temporal_order_mfg = list(G_mfg.nodes())
            adj_matrix_mfg = nx.to_numpy_array(G_mfg, nodelist=temporal_order_mfg, dtype=int)
            root_percentage = (root_nodes / num_nodes) * 100
            
            # Count categorical nodes
            categorical_nodes = [node for node, dist_info in manufacturing_distributions.items() 
                               if dist_info['dist'] in ['categorical', 'categorical_non_uniform']]
            
            metadata_mfg = {
                "temporal_order": temporal_order_mfg,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "manufacturing_distributions": manufacturing_distributions,
                "root_percentage": root_percentage,
                "categorical_root_percentage": (len(categorical_nodes) / num_nodes) * 100
            }
            base_name_mfg = f"dataset_{i}_mfg"
            dataset_dir_mfg = os.path.join(output_dir, base_name_mfg)
            save_dataset(df_mfg, adj_matrix_mfg, metadata_mfg, dataset_dir_mfg, base_name_mfg, index_file, "train")

            # Paired 2: Traditional continuous root nodes (for comparison)
            cdg_cont = CausalDataGenerator(num_samples=500, seed=42)
            cdg_cont.G = G.copy()
            cdg_cont.root_nodes = set(roots)
            df_cont = cdg_cont.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type
                # Do not pass categorical_root_nodes
            )
            G_cont = cdg_cont.G
            temporal_order_cont = list(G_cont.nodes())
            adj_matrix_cont = nx.to_numpy_array(G_cont, nodelist=temporal_order_cont, dtype=int)
            metadata_cont = {
                "temporal_order": temporal_order_cont,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "root_percentage": root_percentage,
                "categorical_root_percentage": 0.0  # No categorical root nodes in continuous dataset
            }
            base_name_cont = f"dataset_{i}_cont"
            dataset_dir_cont = os.path.join(output_dir, base_name_cont)
            save_dataset(df_cont, adj_matrix_cont, metadata_cont, dataset_dir_cont, base_name_cont, index_file, "train")

            print(f"[{i+1}/{total_datasets}] Saved paired datasets: {base_name_mfg} (manufacturing), {base_name_cont} (continuous)")
        except Exception as e:
            print(f"Skipping dataset {i+1} due to error: {e}")
