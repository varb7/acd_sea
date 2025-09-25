#!/usr/bin/env python3
"""
Prior Knowledge Formatting Helper Script

This script helps format prior knowledge from dataset metadata for use with
PyTetrad algorithms. It can be used to test and validate prior knowledge
extraction before running the full inference pipeline.
"""

import argparse
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List

# Add the inference pipeline to the path
sys.path.append(str(Path(__file__).parent))

from utils.io_utils import load_datasets
from utils.prior_knowledge import (
    PriorKnowledgeFormatter, 
    format_prior_knowledge_for_algorithm,
    validate_prior_knowledge,
    log_prior_knowledge_summary
)


def load_dataset_metadata(dataset_path: str) -> List[Dict]:
    """Load metadata from a dataset directory."""
    datasets = load_datasets(dataset_path)
    return [dataset['metadata'] for dataset in datasets]


def format_prior_knowledge_for_dataset(metadata: Dict, algorithm: str = None) -> Dict:
    """Format prior knowledge for a single dataset."""
    formatter = PriorKnowledgeFormatter(metadata)
    prior_knowledge = formatter.get_prior_knowledge_dict()
    
    if algorithm:
        prior_knowledge = format_prior_knowledge_for_algorithm(metadata, algorithm)
    
    return prior_knowledge


def main():
    """Main function for prior knowledge formatting."""
    parser = argparse.ArgumentParser(
        description="Format prior knowledge from dataset metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format prior knowledge for all datasets
  python format_prior_knowledge.py --input-dir causal_meta_dataset --output prior_knowledge.json

  # Format for specific algorithm
  python format_prior_knowledge.py --input-dir causal_meta_dataset --algorithm pc --output pc_prior.json

  # Format for specific dataset
  python format_prior_knowledge.py --dataset-path causal_meta_dataset/dataset_000 --output dataset_000_prior.json

  # Validate prior knowledge
  python format_prior_knowledge.py --input-dir causal_meta_dataset --validate-only
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=str,
                            help='Directory containing datasets')
    input_group.add_argument('--dataset-path', type=str,
                            help='Path to specific dataset directory')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for formatted prior knowledge')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for multiple files')
    
    # Algorithm options
    parser.add_argument('--algorithm', type=str,
                       choices=['pc', 'ges', 'fges', 'fci', 'rfci', 'gfci', 'cfci'],
                       help='Specific algorithm to format for')
    
    # Processing options
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate prior knowledge without formatting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'pickle'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        if args.dataset_path:
            # Process single dataset
            logger.info(f"Processing single dataset: {args.dataset_path}")
            datasets = load_datasets(args.dataset_path)
            if not datasets:
                logger.error(f"No datasets found in {args.dataset_path}")
                return 1
            
            metadata = datasets[0]['metadata']
            prior_knowledge = format_prior_knowledge_for_dataset(metadata, args.algorithm)
            
            # Validate
            node_names = list(datasets[0]['data'].columns)
            if not validate_prior_knowledge(prior_knowledge, node_names):
                logger.error("Prior knowledge validation failed")
                return 1
            
            if args.validate_only:
                logger.info("Validation passed")
                return 0
            
            # Save output
            if args.output:
                if args.format == 'json':
                    with open(args.output, 'w') as f:
                        json.dump(prior_knowledge, f, indent=2, default=str)
                else:
                    with open(args.output, 'wb') as f:
                        pickle.dump(prior_knowledge, f)
                logger.info(f"Prior knowledge saved to {args.output}")
            
            # Log summary
            log_prior_knowledge_summary(prior_knowledge, args.dataset_path)
            
        else:
            # Process multiple datasets
            logger.info(f"Processing datasets from: {args.input_dir}")
            datasets = load_datasets(args.input_dir)
            if not datasets:
                logger.error(f"No datasets found in {args.input_dir}")
                return 1
            
            logger.info(f"Found {len(datasets)} datasets")
            
            all_prior_knowledge = {}
            
            for i, dataset in enumerate(datasets):
                dataset_name = f"dataset_{i}"
                metadata = dataset['metadata']
                node_names = list(dataset['data'].columns)
                
                # Format prior knowledge
                prior_knowledge = format_prior_knowledge_for_dataset(metadata, args.algorithm)
                
                # Validate
                if not validate_prior_knowledge(prior_knowledge, node_names):
                    logger.warning(f"Prior knowledge validation failed for {dataset_name}")
                    continue
                
                all_prior_knowledge[dataset_name] = prior_knowledge
                
                if args.verbose:
                    log_prior_knowledge_summary(prior_knowledge, dataset_name)
            
            if args.validate_only:
                logger.info(f"Validation passed for {len(all_prior_knowledge)} datasets")
                return 0
            
            # Save output
            if args.output:
                if args.format == 'json':
                    with open(args.output, 'w') as f:
                        json.dump(all_prior_knowledge, f, indent=2, default=str)
                else:
                    with open(args.output, 'wb') as f:
                        pickle.dump(all_prior_knowledge, f)
                logger.info(f"Prior knowledge for {len(all_prior_knowledge)} datasets saved to {args.output}")
            
            elif args.output_dir:
                # Save individual files
                from pathlib import Path
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                
                for dataset_name, prior_knowledge in all_prior_knowledge.items():
                    output_file = Path(args.output_dir) / f"{dataset_name}_prior.{args.format}"
                    
                    if args.format == 'json':
                        with open(output_file, 'w') as f:
                            json.dump(prior_knowledge, f, indent=2, default=str)
                    else:
                        with open(output_file, 'wb') as f:
                            pickle.dump(prior_knowledge, f)
                
                logger.info(f"Prior knowledge saved to {args.output_dir}")
            
            # Print summary
            logger.info(f"Successfully processed {len(all_prior_knowledge)} datasets")
            
            # Print overall statistics
            total_forbidden = sum(len(pk.get('forbidden_edges', [])) for pk in all_prior_knowledge.values())
            total_required = sum(len(pk.get('required_edges', [])) for pk in all_prior_knowledge.values())
            total_tiers = sum(len(pk.get('tier_ordering', [])) for pk in all_prior_knowledge.values())
            
            logger.info(f"Overall statistics:")
            logger.info(f"  Total forbidden edges: {total_forbidden}")
            logger.info(f"  Total required edges: {total_required}")
            logger.info(f"  Total tiers: {total_tiers}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing prior knowledge: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

