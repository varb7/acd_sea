# CSuite Benchmark Collection

This directory contains CSuite-style datasets for causal discovery benchmarking, separate from the main causal meta dataset collection. These datasets are inspired by Microsoft's CSuite benchmark patterns and are designed for systematic evaluation of causal discovery algorithms.

## Directory Structure

```
csuite_benchmark/
├── README.md                           # This file
├── datasets/                           # Generated CSuite datasets
│   ├── chain/                         # Chain pattern datasets
│   ├── collider/                      # Collider pattern datasets
│   ├── backdoor/                      # Backdoor pattern datasets
│   ├── mixed_confounding/             # Mixed confounding datasets
│   ├── weak_arrow/                    # Weak arrow pattern datasets
│   └── large_backdoor/                # Large backdoor datasets
├── results/                           # Evaluation results
│   ├── causal_discovery_results.csv   # Main results file
│   ├── pattern_analysis/              # Pattern-specific analysis
│   └── algorithm_comparison/          # Cross-algorithm comparisons
├── configs/                           # CSuite-specific configurations
│   ├── csuite_config.yaml            # Main CSuite configuration
│   └── evaluation_config.yaml        # Evaluation parameters
└── scripts/                           # Evaluation and analysis scripts
    ├── run_csuite_evaluation.py      # Main evaluation script
    ├── analyze_patterns.py           # Pattern-specific analysis
    └── generate_benchmark_report.py  # Generate benchmark reports
```

## Dataset Organization

Each pattern directory contains datasets with the following naming convention:
- `csuite_{pattern}_{nodes}nodes_{id:03d}/`
  - `csuite_chain_3nodes_001/`
  - `csuite_collider_4nodes_002/`
  - etc.

## Usage

### Generate CSuite Datasets
```bash
# Generate all patterns
python data_generator/generate_csuite_datasets.py --all-patterns --output csuite_benchmark/datasets

# Generate specific patterns
python data_generator/generate_csuite_datasets.py --patterns chain collider --output csuite_benchmark/datasets
```

### Run Evaluations
```bash
# Run full CSuite evaluation
python csuite_benchmark/scripts/run_csuite_evaluation.py

# Run specific pattern evaluation
python csuite_benchmark/scripts/run_csuite_evaluation.py --patterns chain collider
```

### Generate Reports
```bash
# Generate comprehensive benchmark report
python csuite_benchmark/scripts/generate_benchmark_report.py
```

## Pattern Descriptions

### Chain Pattern
- **Structure**: Linear causal chain X0 → X1 → X2 → ... → Xn
- **Use Case**: Sequential causal relationships
- **Challenges**: Long-range dependencies, temporal ordering

### Collider Pattern  
- **Structure**: Collider structure X0 → X1 ← X2
- **Use Case**: Collider bias scenarios
- **Challenges**: Conditional independence, collider detection

### Backdoor Pattern
- **Structure**: Confounded treatment-outcome relationship
- **Use Case**: Backdoor adjustment scenarios
- **Challenges**: Confounder identification, adjustment strategies

### Mixed Confounding Pattern
- **Structure**: Complex confounding with mixed variable types
- **Use Case**: Real-world confounding scenarios
- **Challenges**: Mixed data types, high-dimensional confounding

### Weak Arrow Pattern
- **Structure**: Chain with weak causal effects
- **Use Case**: Sensitivity to weak effects
- **Challenges**: Effect size detection, statistical power

### Large Backdoor Pattern
- **Structure**: Multiple confounders affecting treatment and outcome
- **Use Case**: High-dimensional confounding
- **Challenges**: High-dimensional inference, confounder selection

## Evaluation Metrics

- **Structural Hamming Distance (SHD)**: Graph structure accuracy
- **F1 Score**: Edge detection performance
- **Precision/Recall**: Edge-wise performance
- **Pattern-specific Metrics**: Tailored to each pattern type
- **Algorithm Comparison**: Cross-algorithm performance analysis

## Integration with Main Pipeline

The CSuite collection is designed to work alongside the main causal meta dataset collection:

- **Separate Storage**: Independent directory structure
- **Shared Tools**: Uses same inference pipeline components
- **Cross-Collection Analysis**: Can compare with main collection results
- **Dedicated Reporting**: CSuite-specific evaluation reports
