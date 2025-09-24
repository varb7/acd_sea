#!/usr/bin/env python3
"""
CSuite Dataset Generator

This script generates datasets using CSuite-style configurations for benchmarking
causal discovery algorithms. It supports various structural patterns and can
generate graphs with 2-5 nodes.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the generator module to the path
sys.path.append(str(Path(__file__).parent))

from generator.csuite_dag_generator import (
    CSuiteConfig, 
    CSuiteDAGGenerator, 
    generate_csuite_meta_dataset
)


def main():
    """Main entry point for CSuite dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate CSuite-style datasets for causal discovery benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_csuite_datasets.py --patterns chain collider --nodes 3 4 5
  python generate_csuite_datasets.py --all-patterns --nodes 2 5 --samples 2000
  python generate_csuite_datasets.py --pattern mixed_confounding --nodes 4 --output my_datasets
        """
    )
    
    # Pattern selection
    pattern_group = parser.add_mutually_exclusive_group(required=True)
    pattern_group.add_argument('--patterns', nargs='+', 
                              choices=['chain', 'collider', 'backdoor', 'mixed_confounding', 'weak_arrow', 'large_backdoor'],
                              help='Specific patterns to generate')
    pattern_group.add_argument('--all-patterns', action='store_true',
                              help='Generate all available patterns')
    
    # Node configuration
    parser.add_argument('--nodes', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       default=[2, 5], help='Range of number of nodes (default: 2 5)')
    
    # Data configuration
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per dataset (default: 1000)')
    parser.add_argument('--output', type=str, default='csuite_meta_dataset',
                       help='Output directory (default: csuite_meta_dataset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Additional options
    parser.add_argument('--test', action='store_true',
                       help='Run test generation first')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.nodes[0] < 2 or args.nodes[1] > 5:
        parser.error("Node count must be between 2 and 5")
    if args.nodes[0] > args.nodes[1]:
        parser.error("Minimum nodes must be <= maximum nodes")
    
    # Determine patterns to use
    if args.all_patterns:
        patterns = ['chain', 'collider', 'backdoor', 'mixed_confounding', 'weak_arrow', 'large_backdoor']
    else:
        patterns = args.patterns
    
    # Filter patterns based on node count constraints
    filtered_patterns = []
    for pattern in patterns:
        if pattern in ['collider', 'backdoor'] and args.nodes[0] < 3:
            logging.warning(f"Skipping {pattern} - requires at least 3 nodes")
            continue
        if pattern in ['mixed_confounding', 'large_backdoor'] and args.nodes[0] < 4:
            logging.warning(f"Skipping {pattern} - requires at least 4 nodes")
            continue
        filtered_patterns.append(pattern)
    
    if not filtered_patterns:
        logging.error("No valid patterns for the specified node range")
        return 1
    
    logging.info("CSuite Dataset Generator")
    logging.info("=" * 50)
    logging.info(f"Patterns: {', '.join(filtered_patterns)}")
    logging.info(f"Node range: {args.nodes[0]}-{args.nodes[1]}")
    logging.info(f"Samples per dataset: {args.samples}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Seed: {args.seed}")
    logging.info("=" * 50)
    
    # Run test if requested
    if args.test:
        logging.info("Running test generation...")
        from generator.csuite_dag_generator import test_csuite_generator
        test_csuite_generator()
        logging.info("Test completed successfully")
    
    # Generate datasets
    try:
        generate_csuite_meta_dataset(
            patterns=filtered_patterns,
            num_nodes_range=tuple(args.nodes),
            num_samples=args.samples,
            output_dir=args.output,
            seed=args.seed
        )
        
        logging.info(f"Successfully generated CSuite-style datasets in '{args.output}'")
        return 0
        
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    sys.exit(main())
