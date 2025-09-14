"""
Configuration for the data generation pipeline.

Exposes:
- TOTAL_DATASETS: number of datasets to generate
- OUTPUT_DIR: output directory for generated datasets
- MANUFACTURING_CONFIG: base config used to sample manufacturing-style root distributions
- GENERATION_CONFIG: parameter ranges used for dynamic configuration strategies
"""

# High-level controls
TOTAL_DATASETS = 100
OUTPUT_DIR = "causal_meta_dataset"

# Base manufacturing configuration used by ManufacturingDistributionManager
# Keys are consumed in data_generator/generator/manufacturing_distributions.py
MANUFACTURING_CONFIG = {
    # Fraction of root nodes that should be categorical (rest continuous)
    "categorical_percentage": 0.10,  # 10% categorical roots by default

    # Convenience mirror of the complement; some reporting uses this
    "continuous_percentage": 0.90,

    # Mixture over continuous root distributions
    # Must sum to 1.0
    "continuous_distributions": {
        "normal": 0.60,
        "truncated_normal": 0.30,
        "lognormal": 0.10,
    },

    # Mixture over categorical root distributions
    # Must sum to 1.0
    "categorical_distributions": {
        "uniform": 0.50,
        "non_uniform": 0.50,
    },

    # Noise injected downstream in SCDG pipeline for non-root nodes
    "noise_level": 0.005,
    "noise_type": "normal",
    "noise_params": {"mean": 0.0, "std": 0.005},
}

# Parameter ranges used by the DynamicConfigGenerator for random/gradient strategies
# Keys are consumed in data_generator/generator/dynamic_config_generator.py
GENERATION_CONFIG = {
    # Global categorical vs continuous split for roots
    # Sampled uniformly within [min, max]
    "categorical_percentage": (0.02, 0.40),

    # Continuous distribution mixture component ranges (before normalization)
    "normal_percentage": (0.50, 0.85),
    "truncated_normal_percentage": (0.10, 0.45),
    "lognormal_percentage": (0.05, 0.25),

    # Categorical distribution mixture component ranges (before normalization)
    "uniform_categorical_percentage": (0.30, 0.70),
    "non_uniform_categorical_percentage": (0.30, 0.70),

    # Noise level range for non-root nodes (std for Gaussian noise)
    "noise_level": (0.001, 0.02),

    # Graph structure controls used by the enhanced generator
    "graph_structure": {
        # Change these two values to control the allowed number of nodes
        # Used at runtime by dag_generator_enhanced.generate_meta_dataset_with_diverse_configurations
        "num_nodes_range": [10, 20],

        # Fraction of nodes that are roots (sampled uniformly within range)
        "root_nodes_percentage_range": [0.10, 0.30],

        # Target edge density relative to max acyclic edges
        "edges_density_range": [0.30, 0.80],
    },

    # Data generation controls (samples per dataset)
    "data_generation": {
        "num_samples_range": [1000, 50000],
        "default_num_samples": 1000,
    },
}


