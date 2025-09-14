# Data Generator Configuration System

This document describes the comprehensive configuration system for the data generator pipeline.

## Overview

The data generator now supports a flexible, validated configuration system that allows you to:
- Define all parameters in YAML, JSON, or Python files
- Use preset configurations for common use cases
- Override parameters via command line arguments or environment variables
- Validate configurations before running
- Maintain backward compatibility with the legacy `config.py` system

## Quick Start

### Using Default Configuration
```bash
python main.py
```

### Using a Configuration File
```bash
python main.py --config config.yaml
```

### Using a Preset
```bash
python main.py --preset minimal
```

### Overriding Parameters
```bash
python main.py --total-datasets 50 --strategy random --output-dir my_data
```

## Configuration Files

### YAML Configuration (Recommended)

The primary configuration format is YAML. Create a `config.yaml` file:

```yaml
# Core settings
total_datasets: 100
output_dir: "causal_meta_dataset"
random_seed: 42

# Graph structure
graph_structure:
  num_nodes_range: [10, 20]
  root_nodes_percentage_range: [0.10, 0.30]
  edges_density_range: [0.30, 0.80]

# Data generation
data_generation:
  num_samples_range: [1000, 50000]
  default_num_samples: 1000

# Manufacturing distributions
manufacturing:
  categorical_percentage: 0.10
  continuous_percentage: 0.90
  continuous_distributions:
    normal: 0.60
    truncated_normal: 0.30
    lognormal: 0.10
  categorical_distributions:
    uniform: 0.50
    non_uniform: 0.50
  noise_level: 0.005
  noise_type: "normal"
  noise_params:
    mean: 0.0
    std: 0.005

# Generation strategy
strategy:
  strategy: "random"  # random, preset_variations, gradient, mixed
  parallel_workers: 1
  progress_reporting: true
```

### JSON Configuration

You can also use JSON format:

```json
{
  "total_datasets": 100,
  "output_dir": "causal_meta_dataset",
  "graph_structure": {
    "num_nodes_range": [10, 20],
    "root_nodes_percentage_range": [0.10, 0.30],
    "edges_density_range": [0.30, 0.80]
  }
}
```

### Python Configuration (Legacy)

The legacy `config.py` format is still supported:

```python
TOTAL_DATASETS = 100
OUTPUT_DIR = "causal_meta_dataset"
MANUFACTURING_CONFIG = {
    "categorical_percentage": 0.10,
    # ... other parameters
}
```

## Preset Configurations

### Default Preset
```bash
python main.py --preset default
```
- 100 datasets
- 10-20 nodes per graph
- 1000-50000 samples per dataset
- Random generation strategy

### Minimal Preset
```bash
python main.py --preset minimal
```
- 5 datasets
- 5-10 nodes per graph
- 100-1000 samples per dataset
- Preset variations strategy
- Good for testing

### Large Scale Preset
```bash
python main.py --preset large_scale
```
- 1000 datasets
- 20-50 nodes per graph
- 10000-100000 samples per dataset
- Mixed generation strategy
- 4 parallel workers

## Command Line Interface

### Basic Usage
```bash
python main.py [options]
```

### Configuration Options
- `--config, -c PATH`: Path to configuration file
- `--preset, -p {default,minimal,large_scale}`: Use a preset configuration
- `--env`: Load configuration from environment variables

### Core Parameters
- `--total-datasets N`: Number of datasets to generate
- `--output-dir PATH`: Output directory
- `--seed N`: Random seed

### Generation Strategy
- `--strategy {random,preset_variations,gradient,mixed}`: Generation strategy
- `--workers N`: Number of parallel workers

### Utility Options
- `--dry-run`: Show configuration without generating data
- `--validate-only`: Only validate configuration
- `--verbose, -v`: Enable verbose output
- `--use-legacy`: Use legacy configuration system

### Examples
```bash
# Use specific config file
python main.py --config my_config.yaml

# Override specific parameters
python main.py --total-datasets 50 --strategy random --output-dir test_data

# Validate configuration only
python main.py --config config.yaml --validate-only

# Dry run to see what would be generated
python main.py --preset minimal --dry-run

# Use legacy configuration
python main.py --use-legacy
```

## Environment Variables

You can also configure the generator using environment variables:

```bash
export DATA_GEN_TOTAL_DATASETS=100
export DATA_GEN_OUTPUT_DIR="my_datasets"
export DATA_GEN_STRATEGY="random"
export DATA_GEN_SEED=42
export DATA_GEN_WORKERS=4

python main.py --env
```

## Configuration Validation

### Validate a Configuration File
```bash
python config_validator.py config.yaml
```

### Validate with Verbose Output
```bash
python config_validator.py config.yaml --verbose
```

### Programmatic Validation
```python
from data_generator.config_validator import validate_config_file

is_valid, errors, warnings = validate_config_file("config.yaml")
if not is_valid:
    print("Errors:", errors)
```

## Configuration Schema

### Core Settings
- `total_datasets`: Number of datasets to generate (positive integer)
- `output_dir`: Output directory path (string)
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `random_seed`: Random seed for reproducibility (non-negative integer)
- `validate_config`: Whether to validate configuration (boolean)

### Graph Structure (`graph_structure`)
- `num_nodes_range`: Range of nodes per graph [min, max]
- `root_nodes_percentage_range`: Range of root node percentage [min, max]
- `edges_density_range`: Range of edge density [min, max]

### Data Generation (`data_generation`)
- `num_samples_range`: Range of samples per dataset [min, max]
- `default_num_samples`: Default number of samples

### Manufacturing Distributions (`manufacturing`)
- `categorical_percentage`: Percentage of categorical root nodes
- `continuous_percentage`: Percentage of continuous root nodes
- `continuous_distributions`: Distribution weights for continuous nodes
- `categorical_distributions`: Distribution weights for categorical nodes
- `noise_level`: Noise level for non-root nodes
- `noise_type`: Type of noise (normal, uniform, exponential)
- `noise_params`: Parameters for noise generation

### Generation Ranges (`generation_ranges`)
- `categorical_percentage`: Range for categorical percentage
- `normal_percentage`: Range for normal distribution percentage
- `truncated_normal_percentage`: Range for truncated normal percentage
- `lognormal_percentage`: Range for lognormal percentage
- `uniform_categorical_percentage`: Range for uniform categorical percentage
- `non_uniform_categorical_percentage`: Range for non-uniform categorical percentage
- `noise_level`: Range for noise level

### Output Settings (`output`)
- `output_dir`: Output directory
- `save_metadata`: Whether to save metadata
- `save_graphs`: Whether to save graph structures
- `save_adjacency_matrices`: Whether to save adjacency matrices
- `metadata_format`: Metadata file format (json, yaml, pickle)
- `graph_format`: Graph file format (npy, pickle, json)
- `adjacency_format`: Adjacency matrix format (csv, json, npy)

### Generation Strategy (`strategy`)
- `strategy`: Generation strategy (random, preset_variations, gradient, mixed)
- `seed`: Random seed
- `parallel_workers`: Number of parallel workers
- `progress_reporting`: Whether to show progress
- `save_intermediate_results`: Whether to save intermediate results

## Creating Configuration Templates

### Create a Default Template
```bash
python config_loader.py create config_template.yaml
```

### Create a Minimal Template
```bash
python config_loader.py create minimal_template.yaml minimal
```

### Create a Large Scale Template
```bash
python config_loader.py create large_scale_template.yaml large_scale
```

## Migration from Legacy Configuration

If you're using the old `config.py` system, you can migrate gradually:

1. **Keep using legacy**: Use `--use-legacy` flag
2. **Convert to new format**: Use the config loader to convert
3. **Use both**: The new system can load legacy configurations

### Converting Legacy Config
```python
from data_generator.config_loader import ConfigLoader

# Load legacy config
loader = ConfigLoader()
config = loader.load_config(config_path="config.py")

# Save as YAML
config.to_yaml("new_config.yaml")
```

## Troubleshooting

### Common Issues

1. **Configuration not found**: Make sure the config file exists and path is correct
2. **Validation errors**: Check that all required parameters are present and valid
3. **Permission errors**: Ensure you have write access to the output directory
4. **Memory issues**: Reduce `total_datasets` or `num_samples_range` for large configurations

### Getting Help

- Use `--validate-only` to check configuration without generating data
- Use `--dry-run` to see what would be generated
- Use `--verbose` for detailed error messages
- Check the validation output for specific error messages

### Debug Mode

Enable debug logging for detailed information:

```yaml
log_level: "DEBUG"
```

Or use the command line:

```bash
python main.py --config config.yaml --verbose
```

## Best Practices

1. **Start with presets**: Use `--preset minimal` for testing
2. **Validate configurations**: Always validate before running large generations
3. **Use version control**: Keep your configuration files in version control
4. **Document custom configs**: Add comments to explain non-standard settings
5. **Test incrementally**: Start with small datasets and scale up
6. **Monitor resources**: Watch disk space and memory usage for large generations
