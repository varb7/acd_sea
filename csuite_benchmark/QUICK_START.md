# CSuite Benchmark Quick Start Guide

This guide will help you quickly get started with the CSuite benchmark collection for causal discovery evaluation.

## 🚀 Quick Start

### 1. Generate CSuite Datasets
```bash
# Generate all patterns (2-5 nodes)
python data_generator/generate_csuite_datasets.py --all-patterns

# Generate specific patterns
python data_generator/generate_csuite_datasets.py --patterns chain collider --nodes 3 4

# Quick test with minimal datasets
python csuite_benchmark/run_full_benchmark.py --quick-test
```

### 2. Run Full Benchmark Pipeline
```bash
# Complete pipeline: generate → evaluate → analyze → report
python csuite_benchmark/run_full_benchmark.py --all-patterns

# Custom configuration
python csuite_benchmark/run_full_benchmark.py --patterns chain collider backdoor --algorithms pc ges fges
```

### 3. Run Individual Components

#### Generate Datasets Only
```bash
python data_generator/generate_csuite_datasets.py --all-patterns --output csuite_benchmark/datasets
```

#### Run Evaluation Only
```bash
python csuite_benchmark/scripts/run_csuite_evaluation.py --datasets-dir csuite_benchmark/datasets
```

#### Analyze Results Only
```bash
python csuite_benchmark/scripts/analyze_patterns.py --create-plots
```

#### Generate Reports Only
```bash
python csuite_benchmark/scripts/generate_benchmark_report.py
```

## 📊 Understanding the Results

### Directory Structure
```
csuite_benchmark/
├── datasets/                    # Generated CSuite datasets
│   ├── chain/                  # Chain pattern datasets
│   ├── collider/               # Collider pattern datasets
│   ├── backdoor/               # Backdoor pattern datasets
│   ├── mixed_confounding/      # Mixed confounding datasets
│   ├── weak_arrow/             # Weak arrow pattern datasets
│   └── large_backdoor/         # Large backdoor datasets
├── results/                    # Evaluation results
│   ├── causal_discovery_results.csv    # Main results
│   ├── pattern_analysis/               # Pattern-specific analysis
│   ├── algorithm_comparison/           # Cross-algorithm comparison
│   ├── csuite_benchmark_report.html   # HTML report
│   └── plots/                          # Visualization plots
└── configs/                    # Configuration files
```

### Key Files
- **`causal_discovery_results.csv`**: Main evaluation results with all metrics
- **`csuite_benchmark_report.html`**: Comprehensive HTML report
- **`pattern_analysis/`**: Detailed analysis by pattern type
- **`algorithm_comparison/`**: Cross-algorithm performance comparison

## 🔧 Configuration Options

### Dataset Generation
- **Patterns**: `chain`, `collider`, `backdoor`, `mixed_confounding`, `weak_arrow`, `large_backdoor`
- **Node Count**: 2-5 nodes per graph
- **Samples**: Number of data points per dataset
- **Seed**: Random seed for reproducibility

### Evaluation
- **Algorithms**: `pc`, `ges`, `fges`, `fci`, `rfci`, `gfci`, `boss`, `sam`, `cpc`, `cfci`, `fci_max`
- **Metrics**: SHD, F1 Score, Precision, Recall, Specificity, Sensitivity
- **Pattern Analysis**: Detailed performance by pattern type

### Reporting
- **Formats**: HTML, JSON, CSV
- **Visualizations**: Performance plots, heatmaps, scatter plots
- **Analysis**: Pattern-specific insights, algorithm comparison

## 📈 Interpreting Results

### Key Metrics
- **SHD (Structural Hamming Distance)**: Lower is better (0 = perfect)
- **F1 Score**: Higher is better (1.0 = perfect)
- **Precision**: Higher is better (1.0 = perfect)
- **Recall**: Higher is better (1.0 = perfect)

### Pattern-Specific Insights
- **Chain**: Tests sequential causal relationships
- **Collider**: Tests collider bias detection
- **Backdoor**: Tests confounding adjustment
- **Mixed Confounding**: Tests mixed data type handling
- **Weak Arrow**: Tests sensitivity to weak effects
- **Large Backdoor**: Tests high-dimensional confounding

## 🔄 Integration with Main Pipeline

The CSuite collection is designed to work alongside your main causal meta dataset collection:

```bash
# Use CSuite datasets with main inference pipeline
python csuite_benchmark/scripts/run_csuite_inference.py

# Compare with main collection results
python inference_pipeline/main.py --input-dir causal_meta_dataset
python inference_pipeline/main.py --input-dir csuite_benchmark/datasets
```

## 🐛 Troubleshooting

### Common Issues
1. **No datasets found**: Ensure datasets are generated first
2. **Algorithm errors**: Check that all required dependencies are installed
3. **Memory issues**: Reduce number of samples or datasets for testing

### Debug Mode
```bash
# Run with debug logging
python csuite_benchmark/run_full_benchmark.py --quick-test --log-level DEBUG
```

## 📚 Next Steps

1. **Explore Results**: Open the HTML report in your browser
2. **Customize**: Modify configurations for your specific needs
3. **Extend**: Add new patterns or algorithms
4. **Compare**: Run evaluations on both collections
5. **Analyze**: Use the analysis scripts for deeper insights

## 🆘 Getting Help

- Check the main README files for detailed documentation
- Review the configuration files for customization options
- Use the test scripts to verify everything is working
- Check the log files for detailed error information

Happy benchmarking! 🎉


