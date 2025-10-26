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
            return yaml.safe_load(f)
    
    # Return default configuration
    return {
        'num_datasets': 5,
        'output_dir': 'causal_meta_dataset',
        'seed': 42,
        'nodes_range': [30, 40],
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

