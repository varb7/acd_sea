#!/usr/bin/env python3
"""
Test script for the modular RFCI and FGES implementations.

This script tests both modules to ensure they work correctly with different
parameter combinations and data formats.
"""

import numpy as np
import pandas as pd
import time

def create_test_data():
    """Create synthetic mixed-type test data."""
    np.random.seed(42)
    n_samples = 10000  # Smaller dataset for faster testing
    
    # Generate synthetic mixed-type data
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
    
    return df

def test_rfci_module():
    """Test the RFCI module with different parameters."""
    print("=" * 60)
    print("TESTING RFCI MODULE")
    print("=" * 60)
    
    try:
        from tetrad_rfci import TetradRFCI, run_rfci
        
        df = create_test_data()
        print(f"Test data shape: {df.shape}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        # Test 1: Default parameters
        print("\nTest 1: Default parameters (alpha=0.01, depth=-1)")
        start_time = time.time()
        rfci = TetradRFCI()
        adj1 = rfci.run(df)
        default_time = time.time() - start_time
        
        print(f"  Result shape: {adj1.shape}")
        print(f"  Edges detected: {np.sum(adj1)}")
        print(f"  Execution time: {default_time:.3f}s")
        print(f"  Parameters used: {rfci.get_parameters()}")
        
        # Test 2: Tuned parameters (more aggressive)
        print("\nTest 2: Tuned parameters (alpha=0.05, depth=2)")
        start_time = time.time()
        rfci2 = TetradRFCI(alpha=0.05, depth=2)
        adj2 = rfci2.run(df)
        tuned_time = time.time() - start_time
        
        print(f"  Result shape: {adj2.shape}")
        print(f"  Edges detected: {np.sum(adj2)}")
        print(f"  Execution time: {tuned_time:.3f}s")
        print(f"  Parameters used: {rfci2.get_parameters()}")
        
        # Test 3: Convenience function
        print("\nTest 3: Convenience function")
        start_time = time.time()
        adj3 = run_rfci(df, alpha=0.1, depth=1)
        conv_time = time.time() - start_time
        
        print(f"  Result shape: {adj3.shape}")
        print(f"  Edges detected: {np.sum(adj3)}")
        print(f"  Execution time: {conv_time:.3f}s")
        
        # Compare results
        print(f"\nParameter tuning comparison:")
        print(f"  Default (α=0.01, d=-1): {np.sum(adj1)} edges")
        print(f"  Tuned  (α=0.05, d=2):  {np.sum(adj2)} edges")
        print(f"  Aggressive (α=0.1, d=1): {np.sum(adj3)} edges")
        
        return True
        
    except Exception as e:
        print(f"RFCI module test failed: {e}")
        return False

def test_fges_module():
    """Test the FGES module with different parameters."""
    print("\n" + "=" * 60)
    print("TESTING FGES MODULE")
    print("=" * 60)
    
    try:
        from tetrad_fges import TetradFGES, run_fges
        
        df = create_test_data()
        print(f"Test data shape: {df.shape}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        # Test 1: Default parameters
        print("\nTest 1: Default parameters (penalty=2.0, max_degree=-1)")
        start_time = time.time()
        fges = TetradFGES()
        adj1 = fges.run(df)
        default_time = time.time() - start_time
        
        print(f"  Result shape: {adj1.shape}")
        print(f"  Edges detected: {np.sum(adj1)}")
        print(f"  Execution time: {default_time:.3f}s")
        print(f"  Parameters used: {fges.get_parameters()}")
        
        # Test 2: Tuned parameters (more aggressive)
        print("\nTest 2: Tuned parameters (penalty=0.5, max_degree=3)")
        start_time = time.time()
        fges2 = TetradFGES(penalty_discount=0.5, max_degree=3)
        adj2 = fges2.run(df)
        tuned_time = time.time() - start_time
        
        print(f"  Result shape: {adj2.shape}")
        print(f"  Edges detected: {np.sum(adj2)}")
        print(f"  Execution time: {tuned_time:.3f}s")
        print(f"  Parameters used: {fges2.get_parameters()}")
        
        # Test 3: Convenience function
        print("\nTest 3: Convenience function")
        start_time = time.time()
        adj3 = run_fges(df, penalty_discount=1.0, max_degree=2)
        conv_time = time.time() - start_time
        
        print(f"  Result shape: {adj3.shape}")
        print(f"  Edges detected: {np.sum(adj3)}")
        print(f"  Execution time: {conv_time:.3f}s")
        
        # Compare results
        print(f"\nParameter tuning comparison:")
        print(f"  Default (penalty=2.0, max_deg=-1): {np.sum(adj1)} edges")
        print(f"  Tuned  (penalty=0.5, max_deg=3):  {np.sum(adj2)} edges")
        print(f"  Moderate (penalty=1.0, max_deg=2): {np.sum(adj3)} edges")
        
        return True
        
    except Exception as e:
        print(f"FGES module test failed: {e}")
        return False

def test_data_format_compatibility():
    """Test that modules work with both DataFrame and numpy array inputs."""
    print("\n" + "=" * 60)
    print("TESTING DATA FORMAT COMPATIBILITY")
    print("=" * 60)
    
    try:
        from tetrad_rfci import run_rfci
        from tetrad_fges import run_fges
        
        df = create_test_data()
        data_array = df.values
        columns = list(df.columns)
        
        print("Testing DataFrame input...")
        adj_df_rfci = run_rfci(df, alpha=0.05, depth=2)
        adj_df_fges = run_fges(df, penalty_discount=1.0, max_degree=2)
        
        print("Testing numpy array input...")
        adj_np_rfci = run_rfci(data_array, columns=columns, alpha=0.05, depth=2)
        adj_np_fges = run_fges(data_array, columns=columns, penalty_discount=1.0, max_degree=2)
        
        # Verify results are identical
        rfci_match = np.array_equal(adj_df_rfci, adj_np_rfci)
        fges_match = np.array_equal(adj_df_fges, adj_np_fges)
        
        print(f"  RFCI DataFrame vs Numpy: {'✓ MATCH' if rfci_match else '✗ DIFFERENT'}")
        print(f"  FGES DataFrame vs Numpy: {'✓ MATCH' if fges_match else '✗ DIFFERENT'}")
        
        if rfci_match and fges_match:
            print("  ✓ Both modules handle both input formats correctly!")
            return True
        else:
            print("  ✗ Input format compatibility issues detected!")
            return False
            
    except Exception as e:
        print(f"Data format compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MODULAR TETRAD ALGORITHMS TEST SUITE")
    print("=" * 60)
    print("Testing RFCI and FGES modules for integration readiness...")
    
    results = []
    
    # Test RFCI module
    results.append(("RFCI Module", test_rfci_module()))
    
    # Test FGES module
    results.append(("FGES Module", test_fges_module()))
    
    # Test data format compatibility
    results.append(("Data Format Compatibility", test_data_format_compatibility()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modules are ready for integration.")
        print("\nNext steps:")
        print("  1. Integrate into your inference pipeline")
        print("  2. Use the parameter tuning insights from our earlier analysis")
        print("  3. Configure based on your specific use case")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("Common issues:")
        print("  - PyTetrad not properly installed")
        print("  - JVM startup problems")
        print("  - Missing dependencies")

if __name__ == "__main__":
    main()
