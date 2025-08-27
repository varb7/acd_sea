#!/usr/bin/env python3
"""
Test Castle algorithms vs Tetrad algorithms on dataset_000 for comparison
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
    
    return data, true_adj, metadata

def test_castle_algorithms(data, true_adj):
    """Test Castle PC and GES algorithms"""
    print("\n" + "="*60)
    print("TESTING CASTLE ALGORITHMS ON DATASET_000")
    print("="*60)
    
    try:
        from castle.algorithms import PC, GES
        
        print("✓ Castle modules imported successfully")
        
        # Test Castle PC
        print("\n--- Testing Castle PC ---")
        try:
            pc = PC()
            pc.learn(data.values)
            pc_result = pc.causal_matrix
            
            print(f"PC Result shape: {pc_result.shape}")
            print(f"PC Edges detected: {np.sum(pc_result != 0)}")
            print(f"PC Matrix:\n{pc_result}")
            
        except Exception as e:
            print(f"PC failed: {e}")
            pc_result = np.zeros((data.shape[1], data.shape[1]))
        
        # Test Castle GES
        print("\n--- Testing Castle GES ---")
        try:
            ges = GES()
            ges.learn(data.values)
            ges_result = ges.causal_matrix
            
            print(f"GES Result shape: {ges_result.shape}")
            print(f"GES Edges detected: {np.sum(ges_result != 0)}")
            print(f"GES Matrix:\n{ges_result}")
            
        except Exception as e:
            print(f"GES failed: {e}")
            ges_result = np.zeros((data.shape[1], data.shape[1]))
        
        return pc_result, ges_result
        
    except ImportError as e:
        print(f"✗ Failed to import Castle modules: {e}")
        return None, None

def test_tetrad_algorithms(data, true_adj):
    """Test TetradRFCI and TetradFGES algorithms"""
    print("\n" + "="*60)
    print("TESTING TETRAD ALGORITHMS ON DATASET_000")
    print("="*60)
    
    try:
        from inference_pipeline.tetrad_rfci import TetradRFCI
        from inference_pipeline.tetrad_fges import TetradFGES
        
        print("✓ Tetrad modules imported successfully")
        
        # Test TetradRFCI with moderate parameters
        print("\n--- Testing TetradRFCI ---")
        print("Parameters: alpha=0.1, depth=3")
        rfci = TetradRFCI(alpha=0.1, depth=3)
        rfci_result = rfci.run(data)
        
        print(f"RFCI Result shape: {rfci_result.shape}")
        print(f"RFCI Edges detected: {np.sum(rfci_result != 0)}")
        print(f"RFCI Matrix:\n{rfci_result}")
        
        # Test TetradFGES with moderate parameters
        print("\n--- Testing TetradFGES ---")
        print("Parameters: penalty_discount=1.0, max_degree=5")
        fges = TetradFGES(penalty_discount=1.0, max_degree=5)
        fges_result = fges.run(data)
        
        print(f"FGES Result shape: {fges_result.shape}")
        print(f"FGES Edges detected: {np.sum(fges_result != 0)}")
        print(f"FGES Matrix:\n{fges_result}")
        
        return rfci_result, fges_result
        
    except ImportError as e:
        print(f"✗ Failed to import Tetrad modules: {e}")
        return None, None

def compare_results(true_adj, pc_result, ges_result, rfci_result, fges_result):
    """Compare all algorithm results"""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*60)
    
    print(f"True edges: {np.sum(true_adj != 0)}")
    print(f"True matrix:\n{true_adj}")
    
    algorithms = [
        ("Castle PC", pc_result),
        ("Castle GES", ges_result),
        ("Tetrad RFCI", rfci_result),
        ("Tetrad FGES", fges_result)
    ]
    
    print(f"\n{'Algorithm':<15} {'Edges':<6} {'Accuracy':<10} {'Status'}")
    print("-" * 50)
    
    for name, result in algorithms:
        if result is not None and result.shape == true_adj.shape:
            edges = np.sum(result != 0)
            correct = np.sum((result != 0) == (true_adj != 0))
            total = result.size
            accuracy = correct / total
            
            if edges > 0:
                status = "✅ Found edges"
            else:
                status = "❌ No edges"
                
            print(f"{name:<15} {edges:<6} {accuracy:<10.3f} {status}")
        else:
            print(f"{name:<15} {'N/A':<6} {'N/A':<10} ❌ Failed")

def main():
    """Main test function"""
    print("CASTLE vs TETRAD ALGORITHM COMPARISON")
    print("="*60)
    
    # Load dataset
    try:
        data, true_adj, metadata = load_dataset_000()
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Test Castle algorithms
    pc_result, ges_result = test_castle_algorithms(data, true_adj)
    
    # Test Tetrad algorithms
    rfci_result, fges_result = test_tetrad_algorithms(data, true_adj)
    
    # Compare results
    compare_results(true_adj, pc_result, ges_result, rfci_result, fges_result)
    
    print("\n🎉 Algorithm comparison completed!")

if __name__ == "__main__":
    main()
