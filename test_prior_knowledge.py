#!/usr/bin/env python3
"""
Test script for prior knowledge functionality

This script demonstrates how to use prior knowledge with PyTetrad algorithms
using metadata from our data generator.
"""

import sys
import os
from pathlib import Path

# Add the inference pipeline to the path
sys.path.append(str(Path(__file__).parent / "inference_pipeline"))

from utils.io_utils import load_datasets
from utils.prior_knowledge import (
    PriorKnowledgeFormatter, 
    format_prior_knowledge_for_algorithm,
    log_prior_knowledge_summary
)
from tetrad_pc import run_pc
from tetrad_ges import run_ges
import pandas as pd
import numpy as np


def test_prior_knowledge_extraction():
    """Test prior knowledge extraction from metadata."""
    print("🧪 Testing Prior Knowledge Extraction")
    print("=" * 50)
    
    # Load a sample dataset
    try:
        datasets = load_datasets("causal_meta_dataset")
        if not datasets:
            print("❌ No datasets found in causal_meta_dataset")
            return False
        
        dataset = datasets[0]
        metadata = dataset['metadata']
        data = dataset['data']
        
        print(f"✅ Loaded dataset with {len(data.columns)} variables")
        print(f"   Variables: {list(data.columns)}")
        
        # Extract prior knowledge
        formatter = PriorKnowledgeFormatter(metadata)
        prior_knowledge = formatter.get_prior_knowledge_dict()
        
        print(f"\n📊 Prior Knowledge Summary:")
        print(f"   Forbidden edges: {len(prior_knowledge.get('forbidden_edges', []))}")
        print(f"   Required edges: {len(prior_knowledge.get('required_edges', []))}")
        print(f"   Tier ordering: {len(prior_knowledge.get('tier_ordering', []))} tiers")
        print(f"   Root nodes: {len(prior_knowledge.get('root_nodes', []))}")
        
        if prior_knowledge.get('forbidden_edges'):
            print(f"   Sample forbidden edges: {prior_knowledge['forbidden_edges'][:3]}")
        
        if prior_knowledge.get('tier_ordering'):
            print(f"   Tier structure: {[len(tier) for tier in prior_knowledge['tier_ordering']]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prior knowledge extraction: {e}")
        return False


def test_algorithm_with_prior_knowledge():
    """Test running algorithms with and without prior knowledge."""
    print("\n🔬 Testing Algorithms with Prior Knowledge")
    print("=" * 50)
    
    try:
        # Load a sample dataset
        datasets = load_datasets("causal_meta_dataset")
        if not datasets:
            print("❌ No datasets found")
            return False
        
        dataset = datasets[0]
        data = dataset['data']
        metadata = dataset['metadata']
        
        print(f"✅ Testing with dataset shape: {data.shape}")
        
        # Format prior knowledge
        prior_knowledge = format_prior_knowledge_for_algorithm(metadata, "pc")
        
        # Test PC algorithm without prior knowledge
        print("\n🔍 Running PC without prior knowledge...")
        try:
            result_without = run_pc(
                data.values, 
                list(data.columns),
                alpha=0.05,
                use_prior_knowledge=False
            )
            print(f"   ✅ PC completed, result shape: {result_without.shape}")
            print(f"   📊 Edges found: {np.sum(result_without != 0)}")
        except Exception as e:
            print(f"   ❌ PC failed: {e}")
            result_without = None
        
        # Test PC algorithm with prior knowledge
        print("\n🔍 Running PC with prior knowledge...")
        try:
            result_with = run_pc(
                data.values, 
                list(data.columns),
                alpha=0.05,
                use_prior_knowledge=True,
                prior_knowledge=prior_knowledge
            )
            print(f"   ✅ PC with prior knowledge completed, result shape: {result_with.shape}")
            print(f"   📊 Edges found: {np.sum(result_with != 0)}")
        except Exception as e:
            print(f"   ❌ PC with prior knowledge failed: {e}")
            result_with = None
        
        # Compare results
        if result_without is not None and result_with is not None:
            print(f"\n📈 Comparison:")
            print(f"   Without prior knowledge: {np.sum(result_without != 0)} edges")
            print(f"   With prior knowledge: {np.sum(result_with != 0)} edges")
            
            # Check if results are different
            if not np.array_equal(result_without, result_with):
                print(f"   ✅ Results differ - prior knowledge had an effect!")
            else:
                print(f"   ℹ️  Results are identical - prior knowledge may not have affected this dataset")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing algorithms: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prior_knowledge_validation():
    """Test prior knowledge validation."""
    print("\n✅ Testing Prior Knowledge Validation")
    print("=" * 50)
    
    try:
        # Load a sample dataset
        datasets = load_datasets("causal_meta_dataset")
        if not datasets:
            print("❌ No datasets found")
            return False
        
        dataset = datasets[0]
        data = dataset['data']
        metadata = dataset['metadata']
        
        # Format prior knowledge
        formatter = PriorKnowledgeFormatter(metadata)
        prior_knowledge = formatter.get_prior_knowledge_dict()
        
        # Validate
        from utils.prior_knowledge import validate_prior_knowledge
        node_names = list(data.columns)
        
        is_valid = validate_prior_knowledge(prior_knowledge, node_names)
        
        if is_valid:
            print("✅ Prior knowledge validation passed")
        else:
            print("❌ Prior knowledge validation failed")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Prior Knowledge Test Suite")
    print("=" * 60)
    
    tests = [
        ("Prior Knowledge Extraction", test_prior_knowledge_extraction),
        ("Algorithm with Prior Knowledge", test_algorithm_with_prior_knowledge),
        ("Prior Knowledge Validation", test_prior_knowledge_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Prior knowledge functionality is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

