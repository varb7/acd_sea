# CSuite DAG Generator

This module provides a DAG generator inspired by Microsoft's [CSuite](https://github.com/microsoft/csuite) benchmark datasets for causal discovery. It generates synthetic datasets with various structural patterns commonly used in causal inference research.

## Features

- **Multiple Structural Patterns**: Chain, collider, backdoor, mixed confounding, weak arrow, and large backdoor patterns
- **Flexible Node Counts**: Support for 2-5 nodes per graph
- **Mixed Variable Types**: Both continuous and discrete variables
- **Station-wise Ordering**: Consistent with the main data generator's temporal ordering
- **SCDG Integration**: Uses the existing SCDG framework for data generation

## Supported Patterns

### 1. Chain Pattern
- **Structure**: X0 → X1 → X2 → ... → Xn
- **Use Case**: Sequential causal relationships
- **Min Nodes**: 2

### 2. Collider Pattern
- **Structure**: X0 → X1 ← X2 (with optional children)
- **Use Case**: Collider bias scenarios
- **Min Nodes**: 3

### 3. Backdoor Pattern
- **Structure**: X0 → X1 ← X2 (confounding)
- **Use Case**: Backdoor adjustment scenarios
- **Min Nodes**: 3

### 4. Mixed Confounding Pattern
- **Structure**: Treatment → Outcome with multiple confounders
- **Use Case**: Complex confounding scenarios with mixed variable types
- **Min Nodes**: 4

### 5. Weak Arrow Pattern
- **Structure**: Chain with weak causal effects
- **Use Case**: Testing sensitivity to weak effects
- **Min Nodes**: 3

### 6. Large Backdoor Pattern
- **Structure**: Multiple confounders affecting treatment and outcome
- **Use Case**: High-dimensional confounding
- **Min Nodes**: 4

## Usage

### Command Line Interface

#### Generate CSuite datasets directly:
```bash
# Generate all patterns with 2-5 nodes
python data_generator/generate_csuite_datasets.py --all-patterns --nodes 2 5

# Generate specific patterns
python data_generator/generate_csuite_datasets.py --patterns chain collider --nodes 3 4

# Generate with custom parameters
python data_generator/generate_csuite_datasets.py --patterns mixed_confounding --nodes 4 --samples 2000 --output my_datasets
```

#### Use with main data generator:
```bash
# Generate using CSuite strategy
python data_generator/main.py --strategy csuite --total-datasets 10 --output-dir csuite_datasets
```

### Python API

#### Generate a single dataset:
```python
from data_generator.generator.csuite_dag_generator import CSuiteConfig, CSuiteDAGGenerator

# Create configuration
config = CSuiteConfig(
    pattern="chain",
    num_nodes=4,
    num_samples=1000,
    seed=42
)

# Generate dataset
generator = CSuiteDAGGenerator(config)
result = generator.generate_dataset("output_dir", "my_dataset")

print(f"Generated dataset with {result['dataframe'].shape[0]} samples")
print(f"Graph edges: {list(result['graph'].edges())}")
```

#### Generate a meta dataset:
```python
from data_generator.generator.csuite_dag_generator import generate_csuite_meta_dataset

# Generate multiple datasets
generate_csuite_meta_dataset(
    patterns=['chain', 'collider', 'backdoor'],
    num_nodes_range=(3, 5),
    num_samples=1000,
    output_dir="csuite_benchmark",
    seed=42
)
```

## Output Format

Each generated dataset includes:

- **Data CSV**: `{base_name}.csv` - Observational data
- **Adjacency Matrix**: `{base_name}_adj_matrix.csv` - True causal graph
- **Metadata**: `{base_name}_meta.pkl` and `{base_name}_meta.txt` - Dataset information
- **DCDI Format**: `data.npy`, `graph.npy`, `regimes.csv`, `interventions.csv`

### Metadata Fields

- `pattern`: CSuite pattern type
- `num_nodes`: Number of nodes
- `num_edges`: Number of edges
- `root_nodes`: List of root nodes
- `leaf_nodes`: List of leaf nodes
- `variable_types`: Dictionary mapping nodes to types ('continuous' or 'discrete')
- `equation_type`: Type of structural equations ('linear', 'non_linear', 'random')
- `temporal_order`: Station-wise node ordering
- `station_blocks`: List of node groups by station
- `station_names`: List of station names
- `station_map`: Dictionary mapping nodes to stations

## Examples

### Chain Pattern (4 nodes)
```
X0 → X1 → X2 → X3
```
- Root: X0
- Leaf: X3
- All variables: continuous

### Collider Pattern (3 nodes)
```
X0 → X1 ← X2
```
- Roots: X0, X2
- Leaf: X1
- All variables: continuous

### Mixed Confounding Pattern (4 nodes)
```
X2 → X0 → X1 ← X3
```
- Treatment: X0 (discrete)
- Outcome: X1 (continuous)
- Confounders: X2, X3 (mixed types)

## Integration with Causal Discovery

The generated datasets are compatible with the existing causal discovery pipeline:

```python
from inference_pipeline.utils.io_utils import load_datasets

# Load CSuite datasets
datasets = load_datasets("csuite_meta_dataset")

for dataset in datasets:
    data = dataset['data']
    true_adj = dataset['true_adj_matrix']
    metadata = dataset['metadata']
    
    print(f"Pattern: {metadata['pattern']}")
    print(f"Shape: {data.shape}")
    # Run your causal discovery algorithm here
```

## Testing

Run the test script to verify everything works:

```bash
python test_csuite_generator.py
```

This will:
1. Test individual pattern generation
2. Demonstrate different configurations
3. Generate a small meta dataset
4. Show usage examples

## Configuration Options

### CSuiteConfig Parameters

- `pattern`: Pattern type (required)
- `num_nodes`: Number of nodes (2-5, required)
- `num_samples`: Number of samples (default: 1000)
- `seed`: Random seed (default: 42)

### Pattern-specific Parameters

Some patterns support additional parameters through the `**kwargs` argument:

- `weak_effects`: For weak arrow pattern (boolean)
- `confounder_strength`: For backdoor patterns (float)
- `discrete_probability`: For mixed patterns (float)

## References

- [CSuite: A Suite of Benchmark Datasets for Causality](https://github.com/microsoft/csuite)
- [Microsoft Research CSuite Paper](https://github.com/microsoft/csuite) (if available)

## License

This module is part of the ACD-SEA project and follows the same license terms.
