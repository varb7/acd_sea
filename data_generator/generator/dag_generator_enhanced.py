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
from .dynamic_config_generator import DynamicConfigGenerator, create_diverse_configs_from_presets, analyze_config_diversity

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
        raise ValueError("Need 0 ≤ R < N")
    return (N * (N - 1) - R * (R - 1)) // 2

def generate_meta_dataset_with_manufacturing_distributions(
    total_datasets=10, 
    output_dir="causal_meta_dataset", 
    index_file=None,
    manufacturing_config=None
):
    """
    Generate meta dataset using manufacturing-specific distributions including
    truncated normal and lognormal distributions.
    
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

            # Paired 2: Continuous root nodes with manufacturing distributions
            cdg_cont = CausalDataGenerator(num_samples=500, seed=42)
            cdg_cont.G = G.copy()
            cdg_cont.root_nodes = set(roots)
            
            # Use manufacturing distribution manager to assign distributions
            dist_manager = ManufacturingDistributionManager(manufacturing_config, seed=42)
            manufacturing_distributions = dist_manager.assign_manufacturing_distributions(roots)
            
            # Convert manufacturing distributions to SCDG format
            scdg_distributions = {}
            for node, dist_info in manufacturing_distributions.items():
                if dist_info['dist'] in ['normal', 'truncated_normal', 'lognormal']:
                    scdg_distributions[node] = dist_info
                elif dist_info['dist'] == 'categorical':
                    # Skip categorical for continuous dataset
                    continue
                elif dist_info['dist'] == 'categorical_non_uniform':
                    # Skip categorical for continuous dataset
                    continue
            
            df_cont = cdg_cont.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type,
                root_distributions_override=scdg_distributions
            )
            G_cont = cdg_cont.G
            temporal_order_cont = list(G_cont.nodes())
            adj_matrix_cont = nx.to_numpy_array(G_cont, nodelist=temporal_order_cont, dtype=int)
            metadata_cont = {
                "temporal_order": temporal_order_cont,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "manufacturing_distributions": manufacturing_distributions,
                "root_percentage": root_percentage,
                "categorical_root_percentage": 0.0  # No categorical root nodes in continuous dataset
            }
            base_name_cont = f"dataset_{i}_cont"
            dataset_dir_cont = os.path.join(output_dir, base_name_cont)
            save_dataset(df_cont, adj_matrix_cont, metadata_cont, dataset_dir_cont, base_name_cont, index_file, "train")

            print(f"[{i+1}/{total_datasets}] Saved paired datasets: {base_name_cat} (categorical), {base_name_cont} (manufacturing distributions)")
        except Exception as e:
            print(f"Skipping dataset {i+1} due to error: {e}")

def generate_meta_dataset_with_diverse_configurations(
    total_datasets=10, 
    output_dir="causal_meta_dataset", 
    index_file=None,
    config_strategy="preset_variations",
    base_config=None,
    parameter_ranges=None,
    seed=42
):
    """
    Generate meta dataset using diverse configurations for varied datasets.
    
    Args:
        total_datasets: Number of datasets to generate
        output_dir: Output directory for datasets
        index_file: Index file path
        config_strategy: Strategy for generating configurations
            - "preset_variations": Use preset configs with variations
            - "random": Generate random configs within parameter ranges
            - "gradient": Generate gradient configs for specific parameters
            - "mixed": Mix of different strategies
        base_config: Base configuration template
        parameter_ranges: Parameter ranges for random generation
        seed: Random seed
    """
    
    # Use default configurations if not provided
    if base_config is None:
        from config import MANUFACTURING_CONFIG
        base_config = MANUFACTURING_CONFIG
    
    if parameter_ranges is None:
        from config import GENERATION_CONFIG
        parameter_ranges = GENERATION_CONFIG
    
    # Set default index file if not provided
    if index_file is None:
        index_file = os.path.join(output_dir, "index.xlsx")
    
    # Generate configurations based on strategy
    configs = generate_configurations_for_strategy(
        config_strategy, total_datasets, base_config, parameter_ranges, seed
    )
    
    print(f"Generated {len(configs)} configurations using strategy: {config_strategy}")
    
    # Analyze configuration diversity
    diversity_analysis = analyze_config_diversity(configs)
    print("\n📊 Configuration Diversity Analysis:")
    for param, stats in diversity_analysis.items():
        if param.endswith("_stats"):
            param_name = param.replace("_stats", "")
            print(f"  {param_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Generate datasets with each configuration
    for i, config in enumerate(configs):
        try:
            print(f"\n[{i+1}/{len(configs)}] Generating dataset with config: {config.get('config_id', f'config_{i}')}")
            
            # Generate random graph parameters using configuration
            num_nodes_range = parameter_ranges.get("graph_structure", {}).get("num_nodes_range", [3, 19])
            num_nodes = random.randint(num_nodes_range[0], num_nodes_range[1])
            
            root_nodes_percentage_range = parameter_ranges.get("graph_structure", {}).get("root_nodes_percentage_range", [0.10, 0.30])
            min_root_percentage = root_nodes_percentage_range[0]
            max_root_percentage = root_nodes_percentage_range[1]
            min_root_nodes = max(1, int(num_nodes * min_root_percentage))
            max_root_nodes = max(min_root_nodes, int(num_nodes * max_root_percentage))
            max_root_nodes = min(max_root_nodes, num_nodes - 1)
            root_nodes = random.randint(min_root_nodes, max_root_nodes)

            # Calculate min and max edges using configuration
            min_edges = root_nodes + (num_nodes - root_nodes - 1)
            max_edges = max_edges_dag(num_nodes, root_nodes)
            
            # Apply edge density constraint from configuration
            edges_density_range = parameter_ranges.get("graph_structure", {}).get("edges_density_range", [0.3, 0.8])
            target_density = random.uniform(edges_density_range[0], edges_density_range[1])
            target_edges = int(target_density * max_edges)
            
            # Ensure edges are within valid range
            edges = max(min_edges, min(target_edges, max_edges))

            # Get equation type from configuration
            relationship_config = parameter_ranges.get("relationship_types", {})
            linear_percentage = relationship_config.get("linear", {}).get("percentage", 0.6)
            non_linear_percentage = relationship_config.get("non_linear", {}).get("percentage", 0.3)
            interaction_percentage = relationship_config.get("interaction", {}).get("percentage", 0.1)
            
            # Choose equation type based on percentages
            rand_val = random.random()
            if rand_val < linear_percentage:
                equation_type = "linear"
            elif rand_val < linear_percentage + non_linear_percentage:
                equation_type = "non_linear"
            else:
                equation_type = "random"

            # Get number of samples from configuration
            num_samples_range = parameter_ranges.get("data_generation", {}).get("num_samples_range", [100, 10000])
            default_num_samples = parameter_ranges.get("data_generation", {}).get("default_num_samples", 1000)
            num_samples = random.randint(num_samples_range[0], num_samples_range[1])
            
            # Generate the DAG structure once
            cdg_struct = CausalDataGenerator(num_samples=num_samples, seed=seed + i)
            G, roots = cdg_struct.generate_random_graph(num_nodes, root_nodes, edges)

            # Generate dataset with current configuration
            cdg = CausalDataGenerator(num_samples=num_samples, seed=seed + i)
            cdg.G = G.copy()
            cdg.root_nodes = set(roots)
            
            # Use manufacturing distribution manager with current config
            dist_manager = ManufacturingDistributionManager(config, seed=seed + i)
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
            
            df = cdg.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type,
                root_distributions_override=scdg_distributions
            )
            
            # Create metadata
            temporal_order = list(G.nodes())
            adj_matrix = nx.to_numpy_array(G, nodelist=temporal_order, dtype=int)
            root_percentage = (root_nodes / num_nodes) * 100
            
            # Count categorical nodes
            categorical_nodes = [node for node, dist_info in manufacturing_distributions.items() 
                               if dist_info['dist'] in ['categorical', 'categorical_non_uniform']]
            
            metadata = {
                "temporal_order": temporal_order,
                "num_nodes": num_nodes,
                "edges": edges,
                "equation_type": equation_type,
                "manufacturing_distributions": manufacturing_distributions,
                "configuration": config,
                "root_percentage": root_percentage,
                "categorical_root_percentage": (len(categorical_nodes) / num_nodes) * 100,
                "seed": seed + i
            }
            
            # Create dataset name based on configuration
            config_id = config.get('config_id', f'config_{i:03d}')
            base_name = f"dataset_{i:03d}_{config_id}"
            dataset_dir = os.path.join(output_dir, base_name)
            
            save_dataset(df, adj_matrix, metadata, dataset_dir, base_name, index_file, "train")
            
            print(f"✅ Saved dataset: {base_name}")
            print(f"   Config: categorical={config['categorical_percentage']:.3f}, "
                  f"normal={config['continuous_distributions']['normal']:.3f}, "
                  f"noise={config['noise_level']:.4f}")
            
        except Exception as e:
            print(f"❌ Error generating dataset {i+1}: {e}")
            continue
    
    print(f"\n🎉 Generated {len(configs)} diverse datasets successfully!")

def generate_configurations_for_strategy(strategy, total_datasets, base_config, parameter_ranges, seed):
    """
    Generate configurations based on the specified strategy.
    """
    if strategy == "preset_variations":
        from config import MANUFACTURING_CONFIG
        # Create simple variations of the manufacturing config
        configs = []
        base_config = MANUFACTURING_CONFIG.copy()
        
        # Variation 1: More categorical
        config1 = base_config.copy()
        config1['categorical_percentage'] = 0.20
        config1['continuous_percentage'] = 0.80
        configs.append(config1)
        
        # Variation 2: More noise
        config2 = base_config.copy()
        config2['noise_level'] = 0.01
        configs.append(config2)
        
        # Variation 3: Different distribution mix
        config3 = base_config.copy()
        config3['continuous_distributions']['normal'] = 0.60
        config3['continuous_distributions']['truncated_normal'] = 0.30
        config3['continuous_distributions']['lognormal'] = 0.10
        configs.append(config3)
        
        return configs
    
    elif strategy == "random":
        generator = DynamicConfigGenerator(base_config, parameter_ranges, seed=seed)
        return generator.generate_config_batch(total_datasets)
    
    elif strategy == "gradient":
        generator = DynamicConfigGenerator(base_config, parameter_ranges, seed=seed)
        configs = []
        
        # Generate gradients for different parameters
        configs.extend(generator.generate_gradient_configs("categorical_percentage", 0.02, 0.40, 5))
        configs.extend(generator.generate_gradient_configs("normal_percentage", 0.50, 0.85, 5))
        configs.extend(generator.generate_gradient_configs("noise_level", 0.001, 0.02, 5))
        
        return configs[:total_datasets]  # Limit to requested number
    
    elif strategy == "mixed":
        from config import MANUFACTURING_CONFIG
        configs = []
        
        # Add preset variations (same as preset_variations strategy)
        base_config = MANUFACTURING_CONFIG.copy()
        
        # Variation 1: More categorical
        config1 = base_config.copy()
        config1['categorical_percentage'] = 0.20
        config1['continuous_percentage'] = 0.80
        configs.append(config1)
        
        # Variation 2: More noise
        config2 = base_config.copy()
        config2['noise_level'] = 0.01
        configs.append(config2)
        
        # Variation 3: Different distribution mix
        config3 = base_config.copy()
        config3['continuous_distributions']['normal'] = 0.60
        config3['continuous_distributions']['truncated_normal'] = 0.30
        config3['continuous_distributions']['lognormal'] = 0.10
        configs.append(config3)
        
        # Add random configs
        generator = DynamicConfigGenerator(base_config, parameter_ranges, seed=seed)
        random_configs = generator.generate_config_batch(max(0, total_datasets - len(preset_configs)))
        configs.extend(random_configs)
        
        return configs[:total_datasets]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def generate_single_dataset_with_manufacturing_distributions(
    num_nodes=10,
    root_nodes=3,
    edges=15,
    num_samples=500,
    equation_type='linear',
    manufacturing_config=None,
    seed=42
):
    """
    Generate a single dataset using manufacturing-specific distributions.
    
    Args:
        num_nodes: Number of nodes in the DAG
        root_nodes: Number of root nodes
        edges: Number of edges
        num_samples: Number of samples to generate
        equation_type: Type of equations ('linear', 'non_linear', 'random')
        manufacturing_config: Manufacturing configuration dict
        seed: Random seed
        
    Returns:
        tuple: (dataframe, adjacency_matrix, metadata)
    """
    
    # Use default manufacturing config if not provided
    if manufacturing_config is None:
        from config import MANUFACTURING_CONFIG
        manufacturing_config = MANUFACTURING_CONFIG
    
    # Generate the DAG structure
    cdg = CausalDataGenerator(num_samples=num_samples, seed=seed)
    G, roots = cdg.generate_random_graph(num_nodes, root_nodes, edges)
    
    # Use manufacturing distribution manager to assign distributions
    dist_manager = ManufacturingDistributionManager(manufacturing_config, seed=seed)
    manufacturing_distributions = dist_manager.assign_manufacturing_distributions(roots)
    
    # Convert manufacturing distributions to SCDG format
    scdg_distributions = {}
    for node, dist_info in manufacturing_distributions.items():
        if dist_info['dist'] in ['normal', 'truncated_normal', 'lognormal']:
            scdg_distributions[node] = dist_info
        elif dist_info['dist'] == 'categorical':
            # For categorical, we'll handle this separately if needed
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
    
    # Generate data using the pipeline
    df = cdg.generate_data_pipeline(
        total_nodes=num_nodes,
        root_nodes=root_nodes,
        edges=edges,
        equation_type=equation_type,
        root_distributions_override=scdg_distributions
    )
    
    # Create metadata
    temporal_order = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=temporal_order, dtype=int)
    root_percentage = (root_nodes / num_nodes) * 100
    
    metadata = {
        "temporal_order": temporal_order,
        "num_nodes": num_nodes,
        "root_nodes": root_nodes,
        "root_percentage": root_percentage,
        "edges": edges,
        "equation_type": equation_type,
        "manufacturing_distributions": manufacturing_distributions,
        "seed": seed
    }
    
    return df, adj_matrix, metadata

def test_manufacturing_distributions():
    """Test function to verify manufacturing distributions work correctly"""
    
    print("Testing manufacturing distributions integration...")
    
    # Test single dataset generation
    df, adj_matrix, metadata = generate_single_dataset_with_manufacturing_distributions(
        num_nodes=8,
        root_nodes=3,
        edges=10,
        num_samples=1000,
        equation_type='linear',
        seed=123
    )
    
    print(f"Generated dataset shape: {df.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Root nodes: {metadata['root_nodes']}")
    print(f"Manufacturing distributions:")
    for node, dist_info in metadata['manufacturing_distributions'].items():
        print(f"  {node}: {dist_info['dist']}")
        if dist_info['dist'] in ['normal', 'truncated_normal', 'lognormal']:
            print(f"    Mean: {df[node].mean():.3f}, Std: {df[node].std():.3f}")
            print(f"    Min: {df[node].min():.3f}, Max: {df[node].max():.3f}")
    
    print("\nManufacturing distributions integration test completed successfully!")
    return df, adj_matrix, metadata

if __name__ == "__main__":
    # Test the enhanced generator
    test_manufacturing_distributions() 