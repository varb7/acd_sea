# Causal Dataset Generator

Simple, clean generator for synthetic causal datasets using SCDG as the core.

## Quick Start

```bash
# Generate datasets with default config
python main.py

# Use a specific config file
python main.py --config configs/simple_config.yaml

# Generate CSuite-style datasets
python main.py --config configs/simple_config.yaml --mode csuite

# Create a config template
python main.py --save-config my_config.yaml
```

## Architecture

```
Config YAML → simple_config.py → dataset_generator.py → DCDI Output
```

### Files

**Core Files:**
- `main.py` - Entry point (108 lines)
- `simple_config.py` - YAML loader (66 lines)
- `generator/dataset_generator.py` - Simple unified generator (194 lines)
- `generator/csuite_dag_generator.py` - CSuite mode

**Essential Utilities (kept):**
- `generator/manufacturing_distributions.py` - Distribution logic ✅
- `generator/temporal_utils.py` - Station extraction ✅
- `generator/utils.py` - DCDI output saving ✅
- `generator/structural_patterns.py` - CSuite patterns

## Configuration

Config files are simple YAML:

```yaml
num_datasets: 100
output_dir: causal_meta_dataset
seed: 42

# Graph structure
nodes_range: [10, 20]
root_percentage_range: [0.10, 0.30]
edge_density_range: [0.30, 0.80]

# Data generation
samples_range: [1000, 50000]

# Distributions
categorical_percentage: 0.10
continuous_distributions:
  normal: 0.60
  truncated_normal: 0.30
  lognormal: 0.10

# Relationships
relationship_mix:
  linear: 0.60
  nonlinear: 0.30
  mixed: 0.10

noise_level: 0.005
extract_station_info: true
num_stations: 3
save_train_test_split: true
train_ratio: 0.8
```

## Features

- **Graph structure variation**: Nodes, edges, density ranges
- **Distribution types**: Categorical (uniform/non-uniform), continuous (normal/truncated-normal/lognormal)
- **Relationship types**: Linear, nonlinear, mixed
- **Station extraction**: Temporal/station info for prior knowledge
- **DCDI format output**: Same as your existing format
- **CSuite mode**: Benchmark-style small graphs

## What Was Removed

During refactoring, I removed:
- 9 duplicate/unnecessary files
- ~1000 lines of duplicate code
- Complex dynamic generation strategies
- Heavy validation systems

You now have:
- Clean, focused code
- Simple config-driven workflow
- All essential features intact

