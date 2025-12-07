"""
Simple configuration loading for dataset generation.
Just load YAML and return a dict.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        return _normalize_config(raw_config)
    
    # Return default configuration
    return {
        'num_datasets': 5,
        'output_dir': 'causal_meta_dataset',
        'seed': 42,
        'nodes_range': [5, 10],
        'root_percentage_range': [0.10, 0.30],
        'edge_density_range': [0.70, 0.80],
        'samples_range': [1000, 50000],
        'categorical_percentage': 0.10,
        'continuous_distributions': {
            'normal': 0.60,
            'truncated_normal': 0.30,
            'lognormal': 0.10
        },
        'categorical_distributions': {
            'uniform': 0.50,
            'non_uniform': 0.50
        },
        'relationship_mix': {
            'linear': 0.60,
            'nonlinear': 0.30,
            'mixed': 0.10
        },
        'noise_level': 0.005,
        'extract_station_info': True,
        'num_stations': 10,
        'save_train_test_split': True,
        'train_ratio': 0.8
    }


def _normalize_range(value: Any) -> list:
    """
    Convert a list of values to [min, max] range format.
    If already a 2-element list, return as-is.
    If a longer list, return [min, max] of the list.
    """
    if not isinstance(value, list):
        return value
    if len(value) == 2:
        return value
    if len(value) > 2:
        return [min(value), max(value)]
    return value


def _normalize_config(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Translate structured configs into the flat schema expected by the generator
    while preserving the original keys when possible.
    """
    if not raw:
        return {}

    config = dict(raw)

    if "total_datasets" in raw and "num_datasets" not in config:
        config["num_datasets"] = raw["total_datasets"]
    if "random_seed" in raw and "seed" not in config:
        config["seed"] = raw["random_seed"]

    graph = raw.get("graph_structure", {})
    if "nodes_range" not in config:
        nodes_val = graph.get("num_nodes_range", config.get("nodes_range"))
        config["nodes_range"] = _normalize_range(nodes_val)
    if "root_percentage_range" not in config:
        root_val = graph.get("root_nodes_percentage_range", config.get("root_percentage_range"))
        config["root_percentage_range"] = _normalize_range(root_val)
    if "edge_density_range" not in config:
        edge_val = graph.get("edges_density_range", config.get("edge_density_range"))
        config["edge_density_range"] = _normalize_range(edge_val)

    data_generation = raw.get("data_generation", {})
    if "samples_range" not in config:
        samples_val = data_generation.get("num_samples_range", config.get("samples_range"))
        config["samples_range"] = _normalize_range(samples_val)
    if "num_samples" not in config and "default_num_samples" in data_generation:
        config["num_samples"] = data_generation["default_num_samples"]
    if "relationship_mix" not in config and "relationship_mix" in data_generation:
        config["relationship_mix"] = data_generation["relationship_mix"]

    manufacturing = raw.get("manufacturing", {})
    if "categorical_percentage" not in config:
        config["categorical_percentage"] = manufacturing.get("categorical_percentage", config.get("categorical_percentage"))
    if "continuous_distributions" not in config:
        config["continuous_distributions"] = manufacturing.get("continuous_distributions", config.get("continuous_distributions"))
    if "categorical_distributions" not in config:
        config["categorical_distributions"] = manufacturing.get("categorical_distributions", config.get("categorical_distributions"))
    if "noise_level" not in config:
        config["noise_level"] = manufacturing.get("noise_level", config.get("noise_level"))
    if "noise_type" not in config and "noise_type" in manufacturing:
        config["noise_type"] = manufacturing["noise_type"]
    if "noise_params" not in config and "noise_params" in manufacturing:
        config["noise_params"] = manufacturing["noise_params"]

    if "generation_ranges" not in config and "generation_ranges" in raw:
        config["generation_ranges"] = raw["generation_ranges"]

    output = raw.get("output", {})
    if "output_dir" not in config and "output_dir" in output:
        config["output_dir"] = output["output_dir"]
    if "save_metadata" not in config and "save_metadata" in output:
        config["save_metadata"] = output["save_metadata"]
    if "save_graphs" not in config and "save_graphs" in output:
        config["save_graphs"] = output["save_graphs"]
    if "save_adjacency_matrices" not in config and "save_adjacency_matrices" in output:
        config["save_adjacency_matrices"] = output["save_adjacency_matrices"]
    if "metadata_format" not in config and "metadata_format" in output:
        config["metadata_format"] = output["metadata_format"]
    if "graph_format" not in config and "graph_format" in output:
        config["graph_format"] = output["graph_format"]
    if "adjacency_format" not in config and "adjacency_format" in output:
        config["adjacency_format"] = output["adjacency_format"]

    strategy = raw.get("strategy")
    if strategy is not None and "strategy" not in config:
        config["strategy"] = strategy

    return config


def save_config_template(output_path: str, config: Dict[str, Any] = None) -> None:
    """
    Save a configuration template to file.
    
    Args:
        output_path: Path to save config
        config: Configuration to save (uses default if None)
    """
    if config is None:
        config = load_config()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)

