#!/usr/bin/env python3
"""
Test Tetrad algorithms directly on dataset_000 from causal_meta_dataset
"""

import numpy as np
import pandas as pd
import pickle
import os

def load_dataset_000():
    """Load dataset_000 from causal_meta_dataset"""
    base_dir = "causal_meta_dataset/dataset_000_config_000"
    
    # Load data
    data_path = os.path.join(base_dir, "dataset_000_config_000.csv")
    data = pd.read_csv(data_path)
    
    # Load true adjacency matrix
    adj_path = os.path.join(base_dir, "dataset_000_config_000_adj_matrix.csv")
    true_adj = pd.read_csv(adj_path, index_col=0).values
    
    # Load metadata
    meta_path = os.path.join(base_dir, "dataset_000_config_000_meta.pkl")
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset loaded:")
    print(f"  Data shape: {data.shape}")
    print(f"  Variables: {list(data.columns)}")
    print(f"  True adj shape: {true_adj.shape}")
    print(f"  True edges: {np.sum(true_adj != 0)}")
    print(f"  Metadata keys: {list(metadata.keys())}")
    
    return data, true_adj, metadata

def test_tetrad_algorithms(data, true_adj):
    """Test TetradRFCI and TetradFGES on the dataset with different parameters"""
    print("\n" + "="*60)
    print("TESTING TETRAD ALGORITHMS ON DATASET_000")
    print("="*60)
    
    try:
        # Import Tetrad modules
        from inference_pipeline.tetrad_rfci import TetradRFCI
        from inference_pipeline.tetrad_fges import TetradFGES
        
        print("✓ Tetrad modules imported successfully")
        
        # Debug: Check data types and conversion
        print("\n--- DEBUG: DATA ANALYSIS ---")
        print(f"Original data types: {data.dtypes.to_dict()}")
        print(f"Column 'a' unique values: {data['a'].unique()}")
        print(f"Column 'b' range: {data['b'].min():.3f} to {data['b'].max():.3f}")
        print(f"Column 'c' range: {data['c'].min():.3f} to {data['c'].max():.3f}")
        
        # Test data conversion manually
        print("\n--- DEBUG: TESTING DATA CONVERSION ---")
        try:
            rfci_debug = TetradRFCI(alpha=0.05, depth=2)
            categorical_cols, continuous_cols = rfci_debug._detect_data_types(data)
            print(f"Detected categorical columns: {categorical_cols}")
            print(f"Detected continuous columns: {continuous_cols}")
            
            # Test conversion
            tetrad_data = rfci_debug._convert_to_tetrad_format(data)
            print(f"Tetrad data type: {type(tetrad_data)}")
            print(f"Tetrad data methods: {[m for m in dir(tetrad_data) if not m.startswith('_')][:10]}")
            
        except Exception as e:
            print(f"Debug conversion failed: {e}")
        
        # Test TetradRFCI with different parameters
        print("\n--- Testing TetradRFCI ---")
        
        # Conservative parameters (original)
        print("1. Conservative: alpha=0.05, depth=2")
        rfci_conservative = TetradRFCI(alpha=0.05, depth=2)
        rfci_result_cons = rfci_conservative.run(data)
        print(f"   Edges detected: {np.sum(rfci_result_cons != 0)}")
        print(f"   Matrix:\n{rfci_result_cons}")
        
        # Moderate parameters
        print("\n2. Moderate: alpha=0.1, depth=3")
        rfci_moderate = TetradRFCI(alpha=0.1, depth=3)
        rfci_result_mod = rfci_moderate.run(data)
        print(f"   Edges detected: {np.sum(rfci_result_mod != 0)}")
        print(f"   Matrix:\n{rfci_result_mod}")
        
        # Aggressive parameters
        print("\n3. Aggressive: alpha=0.2, depth=4")
        rfci_aggressive = TetradRFCI(alpha=0.2, depth=4)
        rfci_result_agg = rfci_aggressive.run(data)
        print(f"   Edges detected: {np.sum(rfci_result_agg != 0)}")
        print(f"   Matrix:\n{rfci_result_agg}")
        
        # Extremely aggressive parameters
        print("\n4. Extremely Aggressive: alpha=0.5, depth=6")
        rfci_extreme = TetradRFCI(alpha=0.5, depth=6)
        rfci_result_extreme = rfci_extreme.run(data)
        print(f"   Edges detected: {np.sum(rfci_result_extreme != 0)}")
        print(f"   Matrix:\n{rfci_result_extreme}")
        
        # Test TetradFGES with different parameters
        print("\n--- Testing TetradFGES ---")
        
        # Conservative parameters (original)
        print("1. Conservative: penalty_discount=0.5, max_degree=-1")
        fges_conservative = TetradFGES(penalty_discount=0.5, max_degree=-1)
        fges_result_cons = fges_conservative.run(data)
        print(f"   Edges detected: {np.sum(fges_result_cons != 0)}")
        print(f"   Matrix:\n{fges_result_cons}")
        
        # Moderate parameters
        print("\n2. Moderate: penalty_discount=1.0, max_degree=5")
        fges_moderate = TetradFGES(penalty_discount=1.0, max_degree=5)
        fges_result_mod = fges_moderate.run(data)
        print(f"   Edges detected: {np.sum(fges_result_mod != 0)}")
        print(f"   Matrix:\n{fges_result_mod}")
        
        # Aggressive parameters
        print("\n3. Aggressive: penalty_discount=2.0, max_degree=10")
        fges_aggressive = TetradFGES(penalty_discount=2.0, max_degree=10)
        fges_result_agg = fges_aggressive.run(data)
        print(f"   Edges detected: {np.sum(fges_result_agg != 0)}")
        print(f"   Matrix:\n{fges_result_agg}")
        
        # Extremely aggressive parameters
        print("\n4. Extremely Aggressive: penalty_discount=5.0, max_degree=20")
        fges_extreme = TetradFGES(penalty_discount=5.0, max_degree=20)
        fges_result_extreme = fges_extreme.run(data)
        print(f"   Edges detected: {np.sum(fges_result_extreme != 0)}")
        print(f"   Matrix:\n{fges_result_extreme}")
        
        # Compare with true graph
        print("\n--- COMPARISON WITH TRUE GRAPH ---")
        print(f"True edges: {np.sum(true_adj != 0)}")
        print(f"True matrix:\n{true_adj}")
        
        # Calculate metrics for best results
        print("\n--- PERFORMANCE COMPARISON ---")
        
        # Find best RFCI result
        rfci_results = [
            ("Conservative", rfci_result_cons),
            ("Moderate", rfci_result_mod), 
            ("Aggressive", rfci_result_agg),
            ("Extreme", rfci_result_extreme)
        ]
        
        best_rfci = None
        best_rfci_score = -1
        best_rfci_name = ""
        
        for name, result in rfci_results:
            if result.shape == true_adj.shape:
                correct = np.sum((result != 0) == (true_adj != 0))
                total = result.size
                accuracy = correct / total
                print(f"RFCI {name}: {accuracy:.3f} ({correct}/{total}) - Edges: {np.sum(result != 0)}")
                
                if accuracy > best_rfci_score:
                    best_rfci_score = accuracy
                    best_rfci = result
                    best_rfci_name = name
        
        # Find best FGES result
        fges_results = [
            ("Conservative", fges_result_cons),
            ("Moderate", fges_result_mod),
            ("Aggressive", fges_result_agg),
            ("Extreme", fges_result_extreme)
        ]
        
        best_fges = None
        best_fges_score = -1
        best_fges_name = ""
        
        for name, result in fges_results:
            if result.shape == true_adj.shape:
                correct = np.sum((result != 0) == (true_adj != 0))
                total = result.size
                accuracy = correct / total
                print(f"FGES {name}: {accuracy:.3f} ({correct}/{total}) - Edges: {np.sum(result != 0)}")
                
                if accuracy > best_fges_score:
                    best_fges_score = accuracy
                    best_fges = result
                    best_fges_name = name
        
        print(f"\n🎯 Best RFCI: {best_rfci_name} (Accuracy: {best_rfci_score:.3f})")
        print(f"🎯 Best FGES: {best_fges_name} (Accuracy: {best_fges_score:.3f})")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import Tetrad modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Algorithm execution failed: {e}")
        return False

def main():
    """Main test function"""
    print("TETRAD ALGORITHM TEST ON DATASET_000")
    print("="*60)
    
    # Load dataset
    try:
        data, true_adj, metadata = load_dataset_000()
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Test algorithms
    success = test_tetrad_algorithms(data, true_adj)
    
    if success:
        print("\n🎉 Tetrad algorithm test completed successfully!")
    else:
        print("\n⚠️  Tetrad algorithm test failed!")

if __name__ == "__main__":
    main()
