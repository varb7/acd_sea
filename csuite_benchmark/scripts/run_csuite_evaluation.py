#!/usr/bin/env python3
"""
CSuite Evaluation Script

This script runs causal discovery algorithms on CSuite datasets and generates
comprehensive evaluation results with pattern-specific analysis.
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add the inference pipeline to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "inference_pipeline"))

from utils.io_utils import load_datasets
from utils.metrics_utils import calculate_metrics
from utils.graph_utils import build_true_graph
from tetrad_pc import run_pc
from tetrad_ges import run_ges
from tetrad_fges import run_fges
from tetrad_fci import run_fci
from tetrad_rfci import run_rfci
from tetrad_gfci import run_gfci
# Removed: tetrad_boss and tetrad_sam per request
from tetrad_cpc import run_cpc
from tetrad_cfci import run_cfci
from tetrad_fci_max import run_fci_max


def setup_logging(log_file: str, level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_csuite_datasets(datasets_dir: str) -> List[Dict]:
    """Load CSuite datasets from the specified directory."""
    datasets = []
    
    # Look for pattern directories
    pattern_dirs = [d for d in os.listdir(datasets_dir) 
                   if os.path.isdir(os.path.join(datasets_dir, d))]
    
    for pattern_dir in pattern_dirs:
        pattern_path = os.path.join(datasets_dir, pattern_dir)
        pattern_datasets = load_datasets(pattern_path)
        
        # Add pattern information to each dataset
        for dataset in pattern_datasets:
            dataset['pattern'] = pattern_dir
            datasets.append(dataset)
    
    return datasets


def run_algorithm_on_dataset(algorithm: str, dataset: Dict) -> Dict[str, Any]:
    """Run a specific algorithm on a dataset and return results."""
    data = dataset['data']
    true_adj_matrix = dataset['true_adj_matrix']
    metadata = dataset['metadata']
    pattern = dataset.get('pattern', 'unknown')
    
    try:
        # Run the specified algorithm
        if algorithm == 'pc':
            result = run_pc(data)
        elif algorithm == 'ges':
            result = run_ges(data)
        elif algorithm == 'fges':
            result = run_fges(data)
        elif algorithm == 'fci':
            result = run_fci(data)
        elif algorithm == 'rfci':
            result = run_rfci(data)
        elif algorithm == 'gfci':
            result = run_gfci(data)
        # 'boss' and 'sam' removed per request
        elif algorithm == 'cpc':
            result = run_cpc(data)
        elif algorithm == 'cfci':
            result = run_cfci(data)
        elif algorithm == 'fci_max':
            result = run_fci_max(data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Calculate metrics
        metrics = calculate_metrics(true_adj_matrix, result['adjacency_matrix'])
        
        # Add metadata
        result.update({
            'algorithm': algorithm,
            'pattern': pattern,
            'dataset_name': metadata.get('pattern', 'unknown'),
            'num_nodes': metadata.get('num_nodes', 0),
            'num_edges': metadata.get('num_edges', 0),
            'equation_type': metadata.get('equation_type', 'unknown'),
            'variable_types': metadata.get('variable_types', {}),
            'temporal_order': metadata.get('temporal_order', []),
            'station_blocks': metadata.get('station_blocks', []),
            'station_names': metadata.get('station_names', []),
            'station_map': metadata.get('station_map', {}),
            'seed': metadata.get('seed', 0),
            'num_samples': metadata.get('num_samples', 0),
            **metrics
        })
        
        return result
        
    except Exception as e:
        logging.error(f"Error running {algorithm} on {pattern} dataset: {e}")
        return {
            'algorithm': algorithm,
            'pattern': pattern,
            'error': str(e),
            'shd': float('inf'),
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }


def run_csuite_evaluation(
    datasets_dir: str,
    results_dir: str,
    algorithms: List[str],
    patterns: List[str] = None,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """Run comprehensive CSuite evaluation."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Starting CSuite evaluation...")
    
    # Load datasets
    logger.info(f"Loading datasets from {datasets_dir}")
    datasets = load_csuite_datasets(datasets_dir)
    
    if not datasets:
        logger.error("No datasets found!")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    # Filter by patterns if specified
    if patterns:
        datasets = [d for d in datasets if d.get('pattern') in patterns]
        logger.info(f"Filtered to {len(datasets)} datasets for patterns: {patterns}")
    
    # Run evaluations
    results = []
    total_evaluations = len(datasets) * len(algorithms)
    current_evaluation = 0
    
    for dataset in datasets:
        pattern = dataset.get('pattern', 'unknown')
        dataset_name = f"{pattern}_{dataset['metadata'].get('num_nodes', 0)}nodes"
        
        logger.info(f"Evaluating {pattern} dataset: {dataset_name}")
        
        for algorithm in algorithms:
            current_evaluation += 1
            logger.info(f"Running {algorithm} ({current_evaluation}/{total_evaluations})")
            
            result = run_algorithm_on_dataset(algorithm, dataset)
            results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "causal_discovery_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    return results_df


def generate_pattern_analysis(results_df: pd.DataFrame, results_dir: str, logger: logging.Logger):
    """Generate pattern-specific analysis."""
    logger.info("Generating pattern-specific analysis...")
    
    pattern_dir = os.path.join(results_dir, "pattern_analysis")
    os.makedirs(pattern_dir, exist_ok=True)
    
    # Analyze each pattern
    patterns = results_df['pattern'].unique()
    
    for pattern in patterns:
        pattern_data = results_df[results_df['pattern'] == pattern]
        
        # Calculate pattern-specific statistics
        pattern_stats = pattern_data.groupby('algorithm').agg({
            'shd': ['mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        
        # Save pattern analysis
        pattern_file = os.path.join(pattern_dir, f"{pattern}_analysis.csv")
        pattern_stats.to_csv(pattern_file)
        
        logger.info(f"Pattern analysis for {pattern} saved to {pattern_file}")


def generate_algorithm_comparison(results_df: pd.DataFrame, results_dir: str, logger: logging.Logger):
    """Generate algorithm comparison analysis."""
    logger.info("Generating algorithm comparison...")
    
    comparison_dir = os.path.join(results_dir, "algorithm_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Overall algorithm performance
    overall_stats = results_df.groupby('algorithm').agg({
        'shd': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(4)
    
    # Save overall comparison
    overall_file = os.path.join(comparison_dir, "overall_algorithm_performance.csv")
    overall_stats.to_csv(overall_file)
    
    # Pattern-specific algorithm performance
    pattern_comparison = results_df.groupby(['pattern', 'algorithm']).agg({
        'shd': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).round(4)
    
    pattern_file = os.path.join(comparison_dir, "pattern_algorithm_performance.csv")
    pattern_comparison.to_csv(pattern_file)
    
    logger.info(f"Algorithm comparison saved to {comparison_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Run CSuite evaluation on causal discovery algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--datasets-dir', type=str, 
                       default='csuite_benchmark/datasets',
                       help='Directory containing CSuite datasets')
    parser.add_argument('--results-dir', type=str,
                       default='csuite_benchmark/results',
                       help='Directory to save evaluation results')
    parser.add_argument('--algorithms', nargs='+',
                       default=['pc', 'ges', 'fges', 'fci', 'rfci', 'gfci'],
                       help='Algorithms to evaluate')
    parser.add_argument('--patterns', nargs='+',
                       help='Specific patterns to evaluate (default: all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = os.path.join(args.results_dir, "csuite_evaluation.log")
    logger = setup_logging(log_file, args.log_level)
    
    logger.info("CSuite Evaluation Starting")
    logger.info(f"Datasets directory: {args.datasets_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Patterns: {args.patterns or 'all'}")
    
    try:
        # Run evaluation
        results_df = run_csuite_evaluation(
            datasets_dir=args.datasets_dir,
            results_dir=args.results_dir,
            algorithms=args.algorithms,
            patterns=args.patterns,
            logger=logger
        )
        
        if results_df.empty:
            logger.error("No results generated!")
            return 1
        
        # Generate analysis
        generate_pattern_analysis(results_df, args.results_dir, logger)
        generate_algorithm_comparison(results_df, args.results_dir, logger)
        
        # Summary statistics
        logger.info("Evaluation Summary:")
        logger.info(f"Total evaluations: {len(results_df)}")
        logger.info(f"Algorithms tested: {len(results_df['algorithm'].unique())}")
        logger.info(f"Patterns tested: {len(results_df['pattern'].unique())}")
        
        # Best performing algorithm
        best_algorithm = results_df.groupby('algorithm')['f1_score'].mean().idxmax()
        best_f1 = results_df.groupby('algorithm')['f1_score'].mean().max()
        logger.info(f"Best performing algorithm: {best_algorithm} (F1: {best_f1:.4f})")
        
        logger.info("CSuite evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
