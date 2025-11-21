"""
Analyze causal discovery results and compute detailed metrics.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_results(results_file: str) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(results_file)

def analyze_algorithm_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics per algorithm."""
    metrics_cols = ['f1_score', 'precision', 'recall', 'shd', 'unified_score']
    
    summary = df.groupby(['algorithm', 'use_prior_knowledge']).agg({
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'unified_score': ['mean', 'std'],
        'execution_time': 'mean'
    }).round(3)
    
    return summary

def compare_with_without_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Compare performance with vs without prior knowledge."""
    comparison = []
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        if 'use_prior_knowledge' in algo_data.columns:
            with_prior = algo_data[algo_data['use_prior_knowledge'] == True]
            without_prior = algo_data[algo_data['use_prior_knowledge'] == False]
            
            if len(with_prior) > 0 and len(without_prior) > 0:
                comparison.append({
                    'algorithm': algo,
                    'f1_with_prior': with_prior['f1_score'].mean(),
                    'f1_without_prior': without_prior['f1_score'].mean(),
                    'f1_improvement': with_prior['f1_score'].mean() - without_prior['f1_score'].mean(),
                    'precision_with_prior': with_prior['precision'].mean(),
                    'precision_without_prior': without_prior['precision'].mean(),
                    'precision_improvement': with_prior['precision'].mean() - without_prior['precision'].mean(),
                    'shd_with_prior': with_prior['shd'].mean(),
                    'shd_without_prior': without_prior['shd'].mean(),
                    'shd_reduction': without_prior['shd'].mean() - with_prior['shd'].mean(),
                })
            elif len(with_prior) > 0:
                # Only have with_prior data
                comparison.append({
                    'algorithm': algo,
                    'f1_with_prior': with_prior['f1_score'].mean(),
                    'f1_without_prior': None,
                    'f1_improvement': None,
                    'precision_with_prior': with_prior['precision'].mean(),
                    'precision_without_prior': None,
                    'precision_improvement': None,
                    'shd_with_prior': with_prior['shd'].mean(),
                    'shd_without_prior': None,
                    'shd_reduction': None,
                })
    
    return pd.DataFrame(comparison).round(3)

def analyze_by_dataset_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how dataset characteristics affect performance."""
    
    # Correlate metrics with dataset properties
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metric_cols = ['f1_score', 'precision', 'recall', 'unified_score']
    
    correlations = []
    for metric in metric_cols:
        for col in ['num_nodes', 'num_edges', 'edge_density', 'num_samples', 'num_roots']:
            if col in df.columns:
                corr = df[metric].corr(df[col])
                if not np.isnan(corr):
                    correlations.append({
                        'metric': metric,
                        'characteristic': col,
                        'correlation': corr
                    })
    
    return pd.DataFrame(correlations).round(3)

def main():
    parser = argparse.ArgumentParser(description='Analyze causal discovery results')
    parser.add_argument('--results-file', type=str, 
                       default='causal_discovery_results2/causal_discovery_analysis.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for summary statistics')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_file)
    print(f"\n{'='*60}")
    print(f"RESULTS ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"Total datasets: {df['dataset_name'].nunique()}")
    print(f"Total algorithm runs: {len(df)}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}\n")
    
    # Algorithm performance summary
    print(f"{'='*60}")
    print("ALGORITHM PERFORMANCE SUMMARY")
    print(f"{'='*60}\n")
    
    summary = analyze_algorithm_performance(df)
    print(summary)
    
    # Compare with/without prior knowledge
    if 'use_prior_knowledge' in df.columns:
        print(f"\n{'='*60}")
        print("COMPARISON: WITH vs WITHOUT PRIOR KNOWLEDGE")
        print(f"{'='*60}\n")
        
        comparison = compare_with_without_prior(df)
        if not comparison.empty:
            print(comparison.to_string(index=False))
    
    # Best performing algorithm
    print(f"\n{'='*60}")
    print("BEST PERFORMING ALGORITHMS (by Unified Score)")
    print(f"{'='*60}\n")
    
    best_algorithms = df.groupby('algorithm')['unified_score'].mean().sort_values(ascending=False)
    print(best_algorithms.round(3))
    
    # Dataset characteristics analysis
    print(f"\n{'='*60}")
    print("DATASET CHARACTERISTIC CORRELATIONS")
    print(f"{'='*60}\n")
    
    characteristics = analyze_by_dataset_characteristics(df)
    if not characteristics.empty:
        print(characteristics.to_string(index=False))
    
    # Save summary if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("ALGORITHM PERFORMANCE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            if not comparison.empty:
                f.write("COMPARISON: WITH vs WITHOUT PRIOR KNOWLEDGE\n")
                f.write("="*60 + "\n\n")
                f.write(comparison.to_string(index=False))
        
        print(f"\nSummary saved to: {args.output}")

if __name__ == "__main__":
    main()

