# CSuite Workflow Guide

This guide explains the 3-step process for working with CSuite datasets: **Generate → Run Experiments → Analyze Results**.

## Overview

The workflow consists of three main steps:

1. **Generate Datasets** - Create synthetic causal datasets using CSuite patterns
2. **Run Experiments** - Execute causal discovery algorithms on the generated datasets
3. **Analyze Results** - View performance metrics and comparisons

---

## Step 1: Generate Datasets

You have **three options** for generating datasets:

### Option A: Quick CSuite Generation (Single Pattern)

Generate a small set of CSuite datasets with one pattern:

```bash
cd data_generator
python main.py --mode csuite --config configs/csuite_config.yaml
```

**Output:** Datasets in `csuite_datasets_test/` (or as configured)
**Use case:** Quick testing or small experiments

### Option B: Full Experiment Grid (Recommended for Experiments)

Generate the complete experiment grid (Phase 1 + Phase 2):

```bash
cd data_generator
python experiments/generate_csuite_grid.py
```

Or with a custom config:
```bash
python experiments/generate_csuite_grid.py --config experiments/experiment_grid.yaml
```

**Output:** Datasets in `csuite_grid_datasets/` with `index.csv`
**Use case:** Full experimental evaluation

### Option C: Simple Random Datasets

Generate random synthetic datasets:

```bash
cd data_generator
python main.py --mode simple --config configs/simple_config.yaml
```

**Output:** Datasets in `simple_test_datasets/` (or as configured)
**Use case:** General causal discovery testing

---

## Step 2: Run Experiments

After generating datasets, run causal discovery algorithms on them.

### Option A: Using run_experiments.py (Recommended)

This script works with the experiment grid index:

```bash
cd inference_pipeline
python run_experiments.py \
    --index ../csuite_grid_datasets/index.csv \
    --results ../causal_discovery_results2/experiment_results.csv \
    --with-prior true
```

**What it does:**
- Reads datasets from the index.csv
- Runs all registered algorithms on each dataset
- Runs with and without prior knowledge (if `--with-prior true`)
- Appends results to a CSV file

### Option B: Using main.py (Legacy)

For datasets that match the old format:

```bash
cd inference_pipeline
python main.py --use-prior-knowledge --input-dir ../causal_meta_dataset --output-dir ../causal_discovery_results2
```

---

## Step 3: Analyze Results

View and analyze the experiment results:

```bash
cd inference_pipeline
python analyze_results.py --results-file ../causal_discovery_results2/experiment_results.csv
```

**What it shows:**
- Algorithm performance summary
- Comparison with vs without prior knowledge
- Best performing algorithms
- Dataset characteristic correlations

---

## Quick Start Example (Full Workflow)

```bash
# 1. Generate experiment grid datasets
cd data_generator
python experiments/generate_csuite_grid.py

# 2. Run experiments
cd ../inference_pipeline
python run_experiments.py \
    --index ../csuite_grid_datasets/index.csv \
    --results ../causal_discovery_results2/experiment_results.csv \
    --with-prior true

# 3. Analyze results
python analyze_results.py \
    --results-file ../causal_discovery_results2/experiment_results.csv
```

---

## File Locations

### Generated Datasets
- **Experiment grid**:** `csuite_grid_datasets/`
- **Quick CSuite test**: `data_generator/csuite_datasets_test/`
- **Simple datasets**: `data_generator/simple_test_datasets/`
- **Legacy format**: `causal_meta_dataset/`

### Results
- **Experiment results**: `causal_discovery_results2/experiment_results.csv`
- **Legacy results**: `causal_discovery_results2/causal_discovery_analysis.csv`

---

## Configuration Files

### Dataset Generation
- `data_generator/configs/csuite_config.yaml` - Quick CSuite generation
- `data_generator/experiments/experiment_grid.yaml` - Full experiment grid
- `data_generator/configs/simple_config.yaml` - Simple random datasets

### Inference
- `inference_pipeline/config.py` - Inference pipeline settings

---

## Understanding the Experiment Grid

The experiment grid (`experiment_grid.yaml`) generates datasets in two phases:

- **Phase 1:** Wide coverage (sizes 2-10, continuous variables, default equations)
- **Phase 2:** Deep coverage (select sizes, mixed variable types, linear/non-linear equations)

Each dataset is saved with:
- `data.npy` - The generated data
- `graph.npy` - The true adjacency matrix
- `*_meta.pkl` - Metadata (pattern, nodes, edges, etc.)
- `*_train.csv`, `*_test.csv` - Train/test splits
- `index.csv` - Master index of all datasets

---

## Troubleshooting

**Q: Which index.csv should I use?**
- Use the one in your dataset output directory (e.g., `csuite_grid_datasets/index.csv`)

**Q: Results file already exists?**
- `run_experiments.py` will append to existing results. Delete the file to start fresh.

**Q: Datasets not found?**
- Check that `index.csv` exists in your dataset directory
- Verify the `fp_graph` column points to valid directories

**Q: Prior knowledge not working?**
- Ensure datasets have `*_meta.pkl` files with temporal/station information
- Check that `--with-prior true` is set

