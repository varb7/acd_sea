"""
Isolated test for FGES to debug why it returns 0 edges.
This script loads a single dataset and runs FGES with detailed logging.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference_pipeline.tetrad_fges import TetradFGES
from inference_pipeline.utils.prior_knowledge import format_prior_knowledge_for_algorithm


def load_test_dataset(dataset_dir: Path):
    """Load a dataset from directory."""
    data_fp = dataset_dir / 'data.npy'
    graph_fp = dataset_dir / 'graph.npy'
    
    # Find metadata
    meta_files = list(dataset_dir.glob("*_meta.pkl"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.pkl found in {dataset_dir}")
    
    with open(meta_files[0], 'rb') as f:
        metadata = pickle.load(f)
    
    data = np.load(data_fp)
    cols = list(metadata.get('temporal_order', [f'v{i}' for i in range(data.shape[1])]))
    df = pd.DataFrame(data, columns=cols)
    true_adj = np.load(graph_fp)
    
    return df, true_adj, metadata


def test_fges_detailed(df: pd.DataFrame, true_adj: np.ndarray, metadata: dict, use_prior: bool = False):
    """Run FGES with detailed debugging."""
    print("\n" + "="*70)
    print("FGES ISOLATED TEST")
    print("="*70)
    
    print(f"\nData info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes: {df.dtypes.to_dict()}")
    print(f"  Sample:\n{df.head(3)}")
    
    print(f"\nTrue graph info:")
    print(f"  Shape: {true_adj.shape}")
    print(f"  Total edges: {np.sum(true_adj)}")
    print(f"  True adjacency:\n{true_adj}")
    
    # Check for NaNs
    if df.isnull().any().any():
        print(f"\n[WARNING] NaN values detected in data!")
        print(df.isnull().sum())
    
    # Prepare prior knowledge
    prior = None
    if use_prior:
        try:
            prior = format_prior_knowledge_for_algorithm(metadata, "TetradFGES")
            print(f"\nPrior knowledge:")
            print(f"  {prior}")
        except Exception as e:
            print(f"\n[WARNING] Could not format prior knowledge: {e}")
            prior = None
    
    # Run FGES with different parameters
    test_configs = [
        {"penalty_discount": 0.5, "max_degree": -1, "name": "Aggressive (penalty=0.5)"},
        {"penalty_discount": 1.0, "max_degree": -1, "name": "Moderate (penalty=1.0)"},
        {"penalty_discount": 2.0, "max_degree": -1, "name": "Default (penalty=2.0)"},
        {"penalty_discount": 0.1, "max_degree": 10, "name": "Very aggressive (penalty=0.1, max_degree=10)"},
    ]
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        try:
            fges = TetradFGES(
                penalty_discount=config['penalty_discount'],
                max_degree=config['max_degree'],
                orient_cpdag_to_dag=True
            )
            
            print(f"\nFGES parameters:")
            params = fges.get_parameters()
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            # Run FGES
            print(f"\nRunning FGES...")
            adj_matrix = fges.run(df, prior=prior)
            
            print(f"\nResult:")
            print(f"  Shape: {adj_matrix.shape}")
            print(f"  Total edges found: {int(np.sum(adj_matrix))}")
            print(f"  Adjacency matrix:\n{adj_matrix}")
            
            # Compare with true
            if np.sum(adj_matrix) == 0:
                print(f"\n[WARNING] FGES returned 0 edges!")
                print(f"  True edges: {int(np.sum(true_adj))}")
            else:
                # Simple comparison
                correct = np.sum((adj_matrix == 1) & (true_adj == 1))
                false_pos = np.sum((adj_matrix == 1) & (true_adj == 0))
                false_neg = np.sum((adj_matrix == 0) & (true_adj == 1))
                print(f"\nComparison:")
                print(f"  Correct edges: {correct}/{int(np.sum(true_adj))}")
                print(f"  False positives: {false_pos}")
                print(f"  False negatives: {false_neg}")
                
        except Exception as e:
            print(f"\n[ERROR] FGES failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fges_isolated.py <dataset_dir> [--with-prior]")
        print("\nExample:")
        print("  python test_fges_isolated.py csuite_grid_datasets/csuite_chain_5n_p1_continuous_1000_0042")
        return
    
    dataset_dir = Path(sys.argv[1])
    use_prior = "--with-prior" in sys.argv
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    try:
        df, true_adj, metadata = load_test_dataset(dataset_dir)
        test_fges_detailed(df, true_adj, metadata, use_prior=use_prior)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

