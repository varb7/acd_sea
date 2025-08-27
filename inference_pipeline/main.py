import pandas as pd
from config import INPUT_DIR, OUTPUT_DIR
from utils.io_utils import load_datasets, save_results
from utils.metrics_utils import compute_data_properties
from utils.algorithms import AlgorithmRegistry, compute_metrics, default_metrics
from utils.graph_utils import build_true_graph
import networkx as  nx
from typing import List
import mlflow
import numpy as np
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

def main():
    datasets = load_datasets(INPUT_DIR)
    results = []

    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)}")
        data = dataset['data']
        adj = dataset['true_adj_matrix']
        metadata = dataset['metadata']

        props = compute_data_properties(data, adj)
        temporal_order = dataset['metadata'].get('temporal_order')

# Fallback in case temporal_order is missing
        if temporal_order is None:
            G_true = build_true_graph(dataset['true_adj_matrix'], dataset['data'].columns)
            temporal_order = list(nx.topological_sort(G_true))

        # Use the new AlgorithmRegistry system for proper timing
        registry = AlgorithmRegistry()
        algorithms = registry.list_algorithms()
        
        metrics = {}
        max_possible_edges = dataset['true_adj_matrix'].size
        
        for algo_name in algorithms:
            print(f"Running {algo_name}...")
            
            # Run algorithm with timing
            result = registry.run_algorithm(algo_name, dataset['data'].values, list(dataset['data'].columns))
            
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
            
            # Apply temporal pruning if temporal order is provided
            if temporal_order:
                try:
                    from utils.graph_utils import prune_temporal_violations
                    
                    # Convert to NetworkX graph
                    pred_graph = nx.DiGraph(result.adjacency_matrix)
                    pred_graph = nx.relabel_nodes(pred_graph, dict(enumerate(dataset['data'].columns)))
                    
                    # Apply pruning
                    pruned_graph = prune_temporal_violations(pred_graph, temporal_order)
                    pruned_adj = nx.to_numpy_array(pruned_graph, nodelist=dataset['data'].columns)
                    
                    # Compute metrics for pruned result
                    pruned_metrics = compute_metrics(pruned_adj, dataset['true_adj_matrix'], max_possible_edges)
                    metrics[f"{algo_name}_pruned"] = {
                        **pruned_metrics,
                        'execution_time': result.execution_time,
                        'preprocessing_time': result.preprocessing_time,
                        'postprocessing_time': result.postprocessing_time
                    }
                    
                except Exception as e:
                    print(f"[WARNING] Temporal pruning failed for {algo_name}: {e}")
                    metrics[f"{algo_name}_pruned"] = {**default_metrics(), 'execution_time': 0.0}

        for name, vals in metrics.items():
            if vals:
                algo_name = name.replace('_pruned', '')
                pruned = name.endswith('_pruned')
                row = {
                    'dataset_name': f"dataset_{i}",
                    'algorithm': algo_name,
                    'pruned': pruned,
                    # **metadata,  # We'll filter out temporal_order below
                    **{k: v for k, v in metadata.items() if k != 'temporal_order'},
                    **props,
                    **vals,
                    'unified_score': (vals['normalized_shd'] + vals['f1_score']) / 2
                }
                # MLflow logging
                with mlflow.start_run(run_name=f"dataset_{i}_{algo_name}_{'pruned' if pruned else 'raw'}"):
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("pruned", pruned)
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
    main()
