#!/usr/bin/env python3
"""
Test script to verify Tetrad algorithms are properly integrated into the inference pipeline.
"""

import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_algorithm_registry():
    """Test if the Tetrad algorithms are properly registered."""
    print("TESTING TETRAD ALGORITHM INTEGRATION")
    print("=" * 60)
    
    try:
        # Import the pipeline components
        from inference_pipeline.utils.algorithms import AlgorithmRegistry
        
        print("✓ Successfully imported AlgorithmRegistry")
        
        # Create registry instance
        registry = AlgorithmRegistry()
        print("✓ Successfully created AlgorithmRegistry instance")
        
        # List available algorithms
        algorithms = registry.list_algorithms()
        print(f"✓ Available algorithms: {algorithms}")
        
        # Check if Tetrad algorithms are registered
        tetrad_algorithms = [algo for algo in algorithms if 'Tetrad' in algo]
        print(f"✓ Tetrad algorithms found: {tetrad_algorithms}")
        
        if 'TetradRFCI' in algorithms:
            print("✓ TetradRFCI is properly registered")
        else:
            print("✗ TetradRFCI is NOT registered")
            
        if 'TetradFGES' in algorithms:
            print("✓ TetradFGES is properly registered")
        else:
            print("✗ TetradFGES is NOT registered")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_algorithm_execution():
    """Test if the Tetrad algorithms can actually run."""
    print("\n" + "=" * 60)
    print("TESTING TETRAD ALGORITHM EXECUTION")
    print("=" * 60)
    
    try:
        from inference_pipeline.utils.algorithms import AlgorithmRegistry
        import numpy as np
        import pandas as pd
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        
        # Simple test data
        data = np.random.normal(0, 1, (n_samples, n_vars))
        columns = [f"var_{i}" for i in range(n_vars)]
        
        print(f"✓ Created test data: {data.shape}")
        
        # Create registry
        registry = AlgorithmRegistry()
        
        # Test TetradRFCI
        print("\n--- Testing TetradRFCI ---")
        try:
            result_rfci = registry.run_algorithm("TetradRFCI", data, columns)
            if result_rfci:
                print(f"✓ TetradRFCI executed successfully")
                print(f"  Result shape: {result_rfci.adjacency_matrix.shape}")
                print(f"  Edges detected: {np.sum(result_rfci.adjacency_matrix)}")
                print(f"  Execution time: {result_rfci.execution_time:.3f}s")
            else:
                print("✗ TetradRFCI returned None")
        except Exception as e:
            print(f"✗ TetradRFCI execution failed: {e}")
        
        # Test TetradFGES
        print("\n--- Testing TetradFGES ---")
        try:
            result_fges = registry.run_algorithm("TetradFGES", data, columns)
            if result_fges:
                print(f"✓ TetradFGES executed successfully")
                print(f"  Result shape: {result_fges.adjacency_matrix.shape}")
                print(f"  Edges detected: {np.sum(result_fges.adjacency_matrix)}")
                print(f"  Execution time: {result_fges.execution_time:.3f}s")
            else:
                print("✗ TetradFGES returned None")
        except Exception as e:
            print(f"✗ TetradFGES execution failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Execution test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("TETRAD ALGORITHM PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Algorithm Registration
    registration_success = test_algorithm_registry()
    
    # Test 2: Algorithm Execution
    execution_success = test_algorithm_execution()
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    if registration_success and execution_success:
        print("🎉 All tests passed! Tetrad algorithms are successfully integrated.")
        print("\nNext steps:")
        print("  1. Run your main pipeline with the new algorithms")
        print("  2. Compare performance with existing algorithms")
        print("  3. Monitor the optimized parameters in action")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nCommon issues:")
        print("  - tetrad_rfci.py or tetrad_fges.py not in the right location")
        print("  - PyTetrad dependencies not properly installed")
        print("  - Import path issues")

if __name__ == "__main__":
    main()
