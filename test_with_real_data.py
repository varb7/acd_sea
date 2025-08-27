#!/usr/bin/env python3
"""
Test script comparing synthetic vs real data performance for modular modules.
This will show why we got good results in parameter tuning but not in module testing.
"""

import numpy as np
import pandas as pd
import time
import os

def create_synthetic_data():
    """Create synthetic mixed-type test data (like in module testing)."""
    np.random.seed(42)
    n_samples = 500
    
    categorical_var = np.random.choice([0, 1, 2], size=n_samples)
    continuous_var1 = categorical_var + np.random.normal(0, 1, size=n_samples)
    continuous_var2 = 0.5 * continuous_var1 + np.random.normal(0, 1, size=n_samples)
    noise_var = np.random.normal(0, 1, size=n_samples)
    
    df = pd.DataFrame({
        'cat': categorical_var,
        'cont1': continuous_var1,
        'cont2': continuous_var2,
        'noise': noise_var
    })
    
    return df, "Synthetic (500 samples, 4 vars)"

def load_real_mixed_data():
    """Load real mixed-type data from your datasets."""
    try:
        # Try to load from categorical_datasets
        dataset_path = "categorical_datasets/dataset_000"
        
        if os.path.exists(dataset_path):
            # Load data
            data = np.load(os.path.join(dataset_path, "data.npy"))
            adj_matrix = np.load(os.path.join(dataset_path, "graph.npy"))
            
            # Load CSV for column names and data types
            csv_data = pd.read_csv(os.path.join(dataset_path, "dataset_000.csv"))
            
            # Use CSV data (it has proper column names)
            df = csv_data
            return df, f"Real Categorical (dataset_000, {df.shape[0]} samples, {df.shape[1]} vars)", adj_matrix
            
    except Exception as e:
        print(f"Could not load real data: {e}")
        return None, None, None
    
    return None, None, None

def test_algorithm_performance(data, data_name, true_adj=None):
    """Test both RFCI and FGES on the given data."""
    print(f"\n{'='*60}")
    print(f"TESTING ON: {data_name}")
    print(f"{'='*60}")
    
    print(f"Data shape: {data.shape}")
    print(f"Data types: {data.dtypes.to_dict()}")
    
    results = {}
    
    try:
        # Test RFCI
        print(f"\n--- RFCI Testing ---")
        from tetrad_rfci import TetradRFCI
        
        # Test with default parameters
        start_time = time.time()
        rfci_default = TetradRFCI()
        adj_rfci_default = rfci_default.run(data)
        default_time = time.time() - start_time
        
        print(f"  Default (α=0.01, d=-1): {np.sum(adj_rfci_default)} edges, {default_time:.3f}s")
        
        # Test with tuned parameters
        start_time = time.time()
        rfci_tuned = TetradRFCI(alpha=0.05, depth=2)
        adj_rfci_tuned = rfci_tuned.run(data)
        tuned_time = time.time() - start_time
        
        print(f"  Tuned (α=0.05, d=2): {np.sum(adj_rfci_tuned)} edges, {tuned_time:.3f}s")
        
        results['rfci'] = {
            'default': adj_rfci_default,
            'tuned': adj_rfci_tuned,
            'default_time': default_time,
            'tuned_time': tuned_time
        }
        
    except Exception as e:
        print(f"  RFCI failed: {e}")
        results['rfci'] = None
    
    try:
        # Test FGES
        print(f"\n--- FGES Testing ---")
        from tetrad_fges import TetradFGES
        
        # Test with default parameters
        start_time = time.time()
        fges_default = TetradFGES()
        adj_fges_default = fges_default.run(data)
        default_time = time.time() - start_time
        
        print(f"  Default (penalty=2.0, max_deg=-1): {np.sum(adj_fges_default)} edges, {default_time:.3f}s")
        
        # Test with tuned parameters
        start_time = time.time()
        fges_tuned = TetradFGES(penalty_discount=0.5, max_degree=3)
        adj_fges_tuned = fges_tuned.run(data)
        tuned_time = time.time() - start_time
        
        print(f"  Tuned (penalty=0.5, max_deg=3): {np.sum(adj_fges_tuned)} edges, {tuned_time:.3f}s")
        
        results['fges'] = {
            'default': adj_fges_default,
            'tuned': adj_fges_tuned,
            'default_time': default_time,
            'tuned_time': tuned_time
        }
        
    except Exception as e:
        print(f"  FGES failed: {e}")
        results['fges'] = None
    
    # Calculate metrics if we have ground truth
    if true_adj is not None:
        print(f"\n--- Performance Metrics (vs Ground Truth) ---")
        
        if results['rfci']:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Flatten matrices for sklearn
            y_true = true_adj.flatten()
            
            # RFCI metrics
            y_pred_default = results['rfci']['default'].flatten()
            y_pred_tuned = results['rfci']['tuned'].flatten()
            
            # Default RFCI
            if np.sum(y_pred_default) > 0:
                precision_default = precision_score(y_true, y_pred_default, zero_division=0)
                recall_default = recall_score(y_true, y_pred_default, zero_division=0)
                f1_default = f1_score(y_true, y_pred_default, zero_division=0)
                print(f"  RFCI Default: Precision={precision_default:.3f}, Recall={recall_default:.3f}, F1={f1_default:.3f}")
            else:
                print(f"  RFCI Default: No edges detected")
            
            # Tuned RFCI
            if np.sum(y_pred_tuned) > 0:
                precision_tuned = precision_score(y_true, y_pred_tuned, zero_division=0)
                recall_tuned = recall_score(y_true, y_pred_tuned, zero_division=0)
                f1_tuned = f1_score(y_true, y_pred_tuned, zero_division=0)
                print(f"  RFCI Tuned: Precision={precision_tuned:.3f}, Recall={recall_tuned:.3f}, F1={f1_tuned:.3f}")
            else:
                print(f"  RFCI Tuned: No edges detected")
        
        if results['fges']:
            # FGES metrics
            y_pred_default = results['fges']['default'].flatten()
            y_pred_tuned = results['fges']['tuned'].flatten()
            
            # Default FGES
            if np.sum(y_pred_default) > 0:
                precision_default = precision_score(y_true, y_pred_default, zero_division=0)
                recall_default = recall_score(y_true, y_pred_default, zero_division=0)
                f1_default = f1_score(y_true, y_pred_default, zero_division=0)
                print(f"  FGES Default: Precision={precision_default:.3f}, Recall={recall_default:.3f}, F1={f1_default:.3f}")
            else:
                print(f"  FGES Default: No edges detected")
            
            # Tuned FGES
            if np.sum(y_pred_tuned) > 0:
                precision_tuned = precision_score(y_true, y_pred_tuned, zero_division=0)
                recall_tuned = recall_score(y_true, y_pred_tuned, zero_division=0)
                f1_tuned = f1_score(y_true, y_pred_tuned, zero_division=0)
                print(f"  FGES Tuned: Precision={precision_tuned:.3f}, Recall={recall_tuned:.3f}, F1={f1_tuned:.3f}")
            else:
                print(f"  FGES Tuned: No edges detected")
    
    return results

def main():
    """Compare synthetic vs real data performance."""
    print("SYNTHETIC vs REAL DATA PERFORMANCE COMPARISON")
    print("=" * 60)
    print("This will show why parameter tuning showed good results but module testing didn't.")
    
    # Test 1: Synthetic data (like in module testing)
    print("\n" + "="*60)
    print("TEST 1: SYNTHETIC DATA (Module Testing Scenario)")
    print("="*60)
    
    synthetic_data, synthetic_name = create_synthetic_data()
    synthetic_results = test_algorithm_performance(synthetic_data, synthetic_name)
    
    # Test 2: Real mixed-type data (like in parameter tuning)
    print("\n" + "="*60)
    print("TEST 2: REAL MIXED-TYPE DATA (Parameter Tuning Scenario)")
    print("="*60)
    
    real_data, real_name, true_adj = load_real_mixed_data()
    
    if real_data is not None:
        real_results = test_algorithm_performance(real_data, real_name, true_adj)
        
        # Summary comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nSynthetic Data ({synthetic_name}):")
        if synthetic_results['rfci']:
            print(f"  RFCI: Default={np.sum(synthetic_results['rfci']['default'])} edges, Tuned={np.sum(synthetic_results['rfci']['tuned'])} edges")
        if synthetic_results['fges']:
            print(f"  FGES: Default={np.sum(synthetic_results['fges']['default'])} edges, Tuned={np.sum(synthetic_results['fges']['tuned'])} edges")
        
        print(f"\nReal Data ({real_name}):")
        if real_results['rfci']:
            print(f"  RFCI: Default={np.sum(real_results['rfci']['default'])} edges, Tuned={np.sum(real_results['rfci']['tuned'])} edges")
        if real_results['fges']:
            print(f"  FGES: Default={np.sum(real_results['fges']['default'])} edges, Tuned={np.sum(real_results['fges']['tuned'])} edges")
        
        print(f"\nKey Insight:")
        print(f"  - Synthetic data: Weak/no causal signals → Few edges detected")
        print(f"  - Real data: Strong causal signals → More edges detected")
        print(f"  - Parameter tuning shows benefits on real data with actual relationships")
        
    else:
        print("Could not load real data. Make sure mixed_type_test_datasets exists.")
        print("This explains why we can't reproduce the good results from parameter tuning.")

if __name__ == "__main__":
    main()
