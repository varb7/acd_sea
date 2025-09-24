import pandas as pd
from config import INPUT_DIR, OUTPUT_DIR
from utils.io_utils import load_datasets, save_results
from utils.metrics_utils import compute_data_properties
from utils.algorithms import AlgorithmRegistry, compute_metrics, default_metrics
from utils.graph_utils import build_true_graph
from utils.prior_knowledge import format_prior_knowledge_for_algorithm, log_prior_knowledge_summary
import networkx as  nx
from typing import List
import mlflow
import numpy as np
import argparse
try:
    import torch
except ImportError:
    torch = None

mlflow.set_experiment("causal_discovery_experiments")

def make_serializable(obj):
    # Handle dicts
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    # Handle lists/tuples
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    # Handle PyTorch tensors
    elif torch and isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    # Handle numpy scalars/arrays
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle objects with .item() or .tolist() methods
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    elif hasattr(obj, 'tolist') and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    # Fallback: convert to string
    try:
        import json
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def main(use_prior_knowledge=False, algorithms=None):
    datasets = load_datasets(INPUT_DIR)
    results = []

    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)}")
        data = dataset['data']
        adj = dataset['true_adj_matrix']
        metadata = dataset['metadata']

        props = compute_data_properties(data, adj)

        # Prepare prior knowledge if enabled
        prior_knowledge = None
        if use_prior_knowledge:
            try:
                prior_knowledge = format_prior_knowledge_for_algorithm(metadata, "default")
                log_prior_knowledge_summary(prior_knowledge, f"dataset_{i}")
            except Exception as e:
                print(f"Warning: Could not extract prior knowledge for dataset {i}: {e}")
                prior_knowledge = None

        # Use the new AlgorithmRegistry system; enable fallbacks so we still run when Tetrad/JVM is unavailable
        registry = AlgorithmRegistry(enable_fallbacks=True)
        available_algorithms = registry.list_algorithms()
        
        # Filter algorithms if specified
        if algorithms:
            algorithms_to_run = [algo for algo in available_algorithms if algo in algorithms]
        else:
            algorithms_to_run = available_algorithms
        
        metrics = {}
        max_possible_edges = dataset['true_adj_matrix'].size
        
        for algo_name in algorithms_to_run:
            print(f"Running {algo_name}...")
            
            # Run algorithm with timing and prior knowledge
            result = registry.run_algorithm(
                algo_name, 
                dataset['data'].values, 
                list(dataset['data'].columns),
                use_prior_knowledge=use_prior_knowledge,
                prior_knowledge=prior_knowledge
            )
            
            if result is None:
                metrics[algo_name] = {**default_metrics(), 'execution_time': 0.0}
                continue
                
            # Compute metrics for raw result
            raw_metrics = compute_metrics(result.adjacency_matrix, dataset['true_adj_matrix'], max_possible_edges)
            metrics[algo_name] = {
                **raw_metrics,
                'execution_time': result.execution_time,
                'preprocessing_time': result.preprocessing_time,
                'postprocessing_time': result.postprocessing_time
            }
            

        for algo_name, vals in metrics.items():
            if vals:
                row = {
                    'dataset_name': f"dataset_{i}",
                    'algorithm': algo_name,
                    'use_prior_knowledge': use_prior_knowledge,
                    **{k: v for k, v in metadata.items() if k != 'temporal_order'},
                    **props,
                    **vals,
                    'unified_score': (vals['normalized_shd'] + vals['f1_score']) / 2
                }
                # MLflow logging
                with mlflow.start_run(run_name=f"dataset_{i}_{algo_name}"):
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("dataset_name", f"dataset_{i}")
                    # Log all metadata and props as params
                    for k, v in metadata.items():
                        if k != 'temporal_order':
                            mlflow.log_param(k, v)
                    for k, v in props.items():
                        mlflow.log_param(k, v)
                    # Log metrics
                    for k, v in vals.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)
                    mlflow.log_metric("unified_score", (vals['normalized_shd'] + vals['f1_score']) / 2)
                    # Optionally log result row as artifact
                    import json
                    mlflow.log_text(json.dumps(make_serializable(row), indent=2), "result_row.json")
                results.append(row)

    df = pd.DataFrame(results)
    save_results(df, f"{OUTPUT_DIR}/causal_discovery_analysis.csv")
    print("Analysis complete. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal discovery with optional prior knowledge")
    parser.add_argument('--use-prior-knowledge', action='store_true',
                       help='Use prior knowledge from metadata')
    parser.add_argument('--algorithms', nargs='+',
                       help='Specific algorithms to run (default: all)')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR,
                       help='Input directory containing datasets')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update global variables if specified
    if args.input_dir != INPUT_DIR:
        import sys
        sys.modules['config'].INPUT_DIR = args.input_dir
    if args.output_dir != OUTPUT_DIR:
        import sys
        sys.modules['config'].OUTPUT_DIR = args.output_dir
    
    main(use_prior_knowledge=args.use_prior_knowledge, algorithms=args.algorithms)
