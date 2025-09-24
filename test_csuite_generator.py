#!/usr/bin/env python3
"""
Test script for CSuite DAG Generator

This script demonstrates how to use the CSuite DAG generator to create
various structural patterns inspired by Microsoft's CSuite benchmark datasets.
"""

import sys
from pathlib import Path

# Add the data_generator module to the path
sys.path.append(str(Path(__file__).parent / "data_generator"))

from generator.csuite_dag_generator import (
    CSuiteConfig, 
    CSuiteDAGGenerator, 
    generate_csuite_meta_dataset,
    test_csuite_generator
)


def demo_single_dataset():
    """Demonstrate generating a single dataset."""
    print("🔬 Demo: Single Dataset Generation")
    print("=" * 40)
    
    # Create a chain configuration
    config = CSuiteConfig(
        pattern="chain",
        num_nodes=4,
        num_samples=500,
        seed=42
    )
    
    # Generate the dataset
    generator = CSuiteDAGGenerator(config)
    G, metadata = generator.generate_dag()
    df, data_metadata = generator.generate_data(G)
    
    print(f"Pattern: {metadata['pattern']}")
    print(f"Nodes: {metadata['num_nodes']}")
    print(f"Edges: {metadata['num_edges']}")
    print(f"Root nodes: {metadata['root_nodes']}")
    print(f"Variable types: {metadata['variable_types']}")
    print(f"Data shape: {df.shape}")
    print(f"Temporal order: {data_metadata['temporal_order']}")
    print(f"Station blocks: {data_metadata['station_blocks']}")


def demo_different_patterns():
    """Demonstrate different CSuite patterns."""
    print("\n🎯 Demo: Different Patterns")
    print("=" * 40)
    
    patterns = ['chain', 'collider', 'backdoor', 'mixed_confounding']
    
    for pattern in patterns:
        try:
            print(f"\nPattern: {pattern}")
            
            # Determine appropriate node count
            if pattern in ['collider', 'backdoor']:
                num_nodes = 3
            elif pattern in ['mixed_confounding']:
                num_nodes = 4
            else:
                num_nodes = 3
            
            config = CSuiteConfig(
                pattern=pattern,
                num_nodes=num_nodes,
                num_samples=200,
                seed=42
            )
            
            generator = CSuiteDAGGenerator(config)
            G, metadata = generator.generate_dag()
            
            print(f"  Nodes: {list(G.nodes())}")
            print(f"  Edges: {list(G.edges())}")
            print(f"  Root nodes: {metadata['root_nodes']}")
            print(f"  Variable types: {metadata['variable_types']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def demo_meta_dataset():
    """Demonstrate generating a meta dataset."""
    print("\n📊 Demo: Meta Dataset Generation")
    print("=" * 40)
    
    # Generate a small meta dataset
    generate_csuite_meta_dataset(
        patterns=['chain', 'collider'],
        num_nodes_range=(3, 4),
        num_samples=100,
        output_dir="demo_csuite_datasets",
        seed=42
    )


def main():
    """Run all demonstrations."""
    print("🚀 CSuite DAG Generator Demo")
    print("=" * 50)
    
    # Run the built-in test
    test_csuite_generator()
    
    # Run custom demos
    demo_single_dataset()
    demo_different_patterns()
    demo_meta_dataset()
    
    print("\n✅ All demonstrations completed successfully!")
    print("\nTo generate your own datasets, use:")
    print("  python data_generator/generate_csuite_datasets.py --help")
    print("  python data_generator/main.py --strategy csuite")


if __name__ == "__main__":
    main()
