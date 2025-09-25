# Prior Knowledge Integration for PyTetrad Algorithms

This document describes the prior knowledge integration system that allows PyTetrad algorithms to use metadata from our data generator to improve causal discovery performance.

## 🎯 Overview

The prior knowledge system extracts structural constraints from dataset metadata and formats them for use with PyTetrad algorithms. This includes:

- **Temporal Ordering**: Station-wise temporal precedence constraints
- **Station Constraints**: Nodes in later stations cannot cause nodes in earlier stations
- **Root Node Constraints**: Root nodes have no incoming edges
- **Forbidden/Required Edges**: Explicit edge constraints

## 📁 Files

### Core Components
- `utils/prior_knowledge.py` - Main prior knowledge utilities
- `format_prior_knowledge.py` - Helper script for formatting prior knowledge
- `test_prior_knowledge.py` - Test script for functionality

### Updated Algorithms
- `tetrad_pc.py` - Updated PC algorithm with prior knowledge support
- `tetrad_ges.py` - Updated GES algorithm with prior knowledge support
- `main.py` - Updated main pipeline with prior knowledge flag

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Run inference pipeline with prior knowledge
python inference_pipeline/main.py --use-prior-knowledge

# Run specific algorithms with prior knowledge
python inference_pipeline/main.py --use-prior-knowledge --algorithms pc ges fges

# Run on specific dataset directory
python inference_pipeline/main.py --use-prior-knowledge --input-dir my_datasets
```

### 2. Format Prior Knowledge

```bash
# Format prior knowledge for all datasets
python inference_pipeline/format_prior_knowledge.py --input-dir causal_meta_dataset --output prior_knowledge.json

# Format for specific algorithm
python inference_pipeline/format_prior_knowledge.py --input-dir causal_meta_dataset --algorithm pc --output pc_prior.json

# Validate prior knowledge
python inference_pipeline/format_prior_knowledge.py --input-dir causal_meta_dataset --validate-only
```

### 3. Test Functionality

```bash
# Run comprehensive tests
python test_prior_knowledge.py
```

## 🔧 API Usage

### Python API

```python
from inference_pipeline.utils.prior_knowledge import PriorKnowledgeFormatter, format_prior_knowledge_for_algorithm
from inference_pipeline.tetrad_pc import run_pc

# Load dataset metadata
metadata = dataset['metadata']

# Format prior knowledge
prior_knowledge = format_prior_knowledge_for_algorithm(metadata, "pc")

# Run algorithm with prior knowledge
result = run_pc(
    data.values,
    list(data.columns),
    alpha=0.05,
    use_prior_knowledge=True,
    prior_knowledge=prior_knowledge
)
```

### Direct Algorithm Usage

```python
from inference_pipeline.tetrad_pc import TetradPC

# Create algorithm instance
pc = TetradPC(
    alpha=0.05,
    use_prior_knowledge=True,
    prior_knowledge=prior_knowledge
)

# Run algorithm
result = pc.run(data, columns)
```

## 📊 Prior Knowledge Types

### 1. Temporal Ordering Constraints
- **Source**: `temporal_order` in metadata
- **Constraint**: Earlier nodes in temporal order cannot be caused by later nodes
- **Usage**: Forbidden edges from later to earlier nodes

### 2. Station Constraints
- **Source**: `station_blocks` and `station_names` in metadata
- **Constraint**: Nodes in later stations cannot cause nodes in earlier stations
- **Usage**: Forbidden edges between stations

### 3. Root Node Constraints
- **Source**: `root_nodes` in metadata
- **Constraint**: Root nodes have no incoming edges
- **Usage**: Forbidden edges to root nodes

### 4. Tier Ordering
- **Source**: `station_blocks` in metadata
- **Constraint**: Hierarchical ordering of nodes
- **Usage**: Algorithm-specific tier constraints

## 🎛️ Configuration

### Algorithm-Specific Formatting

Different algorithms support different types of prior knowledge:

- **PC, FCI, RFCI, GFCI, CFCI**: Focus on forbidden edges and tier ordering
- **GES, FGES**: Can use all constraint types
- **Other algorithms**: Use default formatting

### Prior Knowledge Dictionary Structure

```python
{
    'forbidden_edges': [('source', 'target'), ...],  # List of forbidden edges
    'required_edges': [('source', 'target'), ...],   # List of required edges
    'tier_ordering': [['node1', 'node2'], ...],      # List of tiers
    'root_nodes': ['node1', 'node2', ...],           # List of root nodes
    'temporal_order': ['node1', 'node2', ...],       # Temporal ordering
    'station_blocks': [['node1'], ['node2'], ...]    # Station blocks
}
```

## 🔍 Validation

The system includes comprehensive validation:

- **Node Name Validation**: Ensures all referenced nodes exist in the dataset
- **Constraint Validation**: Validates edge constraints are consistent
- **Metadata Validation**: Checks metadata completeness

```python
from inference_pipeline.utils.prior_knowledge import validate_prior_knowledge

is_valid = validate_prior_knowledge(prior_knowledge, node_names)
```

## 📈 Performance Impact

### Expected Benefits
- **Improved Accuracy**: Prior knowledge can guide algorithms to better solutions
- **Faster Convergence**: Constraints reduce search space
- **Better Edge Orientation**: Temporal and station constraints help with orientation

### When to Use
- **High-Confidence Metadata**: When you trust the temporal/station ordering
- **Complex Datasets**: Large datasets where constraints can significantly reduce search space
- **Domain Knowledge**: When you have specific structural constraints

### When NOT to Use
- **Uncertain Metadata**: When temporal ordering might be incorrect
- **Exploratory Analysis**: When you want to discover unexpected relationships
- **Small Datasets**: Where constraints might be too restrictive

## 🧪 Testing

### Test Scripts
- `test_prior_knowledge.py` - Comprehensive functionality tests
- `format_prior_knowledge.py --validate-only` - Validation tests

### Test Coverage
- Prior knowledge extraction from metadata
- Algorithm execution with/without prior knowledge
- Validation of constraints
- Error handling and edge cases

## 🔧 Troubleshooting

### Common Issues

1. **JVM Not Started**
   ```
   Error: JVM not started, cannot create Tetrad PriorKnowledge
   ```
   - **Solution**: Ensure PyTetrad is properly installed and JVM is available

2. **Invalid Node Names**
   ```
   Warning: Forbidden edge (node1, node2) references unknown nodes
   ```
   - **Solution**: Check that metadata node names match dataset column names

3. **Prior Knowledge Not Applied**
   ```
   Warning: Could not apply prior knowledge to PC: ...
   ```
   - **Solution**: Check that prior knowledge is properly formatted and valid

### Debug Mode

```bash
# Run with verbose logging
python inference_pipeline/main.py --use-prior-knowledge --verbose

# Test specific algorithm
python test_prior_knowledge.py
```

## 📚 Examples

### Example 1: Basic Usage

```python
# Load dataset
from inference_pipeline.utils.io_utils import load_datasets
datasets = load_datasets("causal_meta_dataset")
dataset = datasets[0]

# Extract prior knowledge
from inference_pipeline.utils.prior_knowledge import format_prior_knowledge_for_algorithm
prior_knowledge = format_prior_knowledge_for_algorithm(dataset['metadata'], "pc")

# Run PC with prior knowledge
from inference_pipeline.tetrad_pc import run_pc
result = run_pc(
    dataset['data'].values,
    list(dataset['data'].columns),
    use_prior_knowledge=True,
    prior_knowledge=prior_knowledge
)
```

### Example 2: Custom Prior Knowledge

```python
# Create custom prior knowledge
custom_prior = {
    'forbidden_edges': [('X1', 'X0'), ('X2', 'X0')],  # X0 cannot be caused by X1 or X2
    'required_edges': [('X0', 'X1')],                  # X0 must cause X1
    'root_nodes': ['X0'],                              # X0 is a root node
    'tier_ordering': [['X0'], ['X1', 'X2']]           # Two tiers
}

# Use with algorithm
result = run_pc(
    data.values,
    columns,
    use_prior_knowledge=True,
    prior_knowledge=custom_prior
)
```

### Example 3: Batch Processing

```python
# Process multiple datasets
for i, dataset in enumerate(datasets):
    prior_knowledge = format_prior_knowledge_for_algorithm(dataset['metadata'], "ges")
    
    result = run_ges(
        dataset['data'].values,
        list(dataset['data'].columns),
        use_prior_knowledge=True,
        prior_knowledge=prior_knowledge
    )
    
    print(f"Dataset {i}: {np.sum(result != 0)} edges found")
```

## 🔄 Integration with Existing Pipeline

The prior knowledge system is designed to be:

- **Backward Compatible**: Existing code works without changes
- **Optional**: Use `--use-prior-knowledge` flag to enable
- **Transparent**: Results include `use_prior_knowledge` flag for comparison
- **Extensible**: Easy to add new constraint types

## 📝 Notes

- Prior knowledge is extracted from metadata generated by our data generator
- Station-wise temporal ordering is the primary source of constraints
- Root node information comes from the DAG generation process
- The system gracefully handles missing or invalid prior knowledge
- All algorithms that support prior knowledge are automatically updated

## 🤝 Contributing

To add prior knowledge support to a new algorithm:

1. Add `use_prior_knowledge` and `prior_knowledge` parameters to the algorithm class
2. Import and use `create_tetrad_prior_knowledge` in the algorithm's run method
3. Apply the prior knowledge using `alg.setKnowledge(prior)`
4. Update the algorithm's convenience function to accept prior knowledge parameters

## 📄 License

This prior knowledge integration follows the same license as the main ACD-SEA project.

