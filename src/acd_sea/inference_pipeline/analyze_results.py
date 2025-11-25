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
    metrics_cols = ['f1_score', 'precision', 'recall', 'shd', 'normalized_shd', 'sid', 'gscore', 'pred_edge_count']
    
    # Only include SID and gscore if they exist in the dataframe
    agg_dict = {
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean',
        'execution_time': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby(['algorithm', 'use_prior']).agg(agg_dict).round(3)
    
    return summary

def compare_with_without_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Compare performance with vs without prior knowledge."""
    comparison = []
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        if 'use_prior' in algo_data.columns:
            with_prior = algo_data[algo_data['use_prior'] == True]
            without_prior = algo_data[algo_data['use_prior'] == False]
            
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
    metric_cols = ['f1_score', 'precision', 'recall', 'normalized_shd']
    
    # Add SID and gscore if available
    if 'sid' in df.columns:
        metric_cols.append('sid')
    if 'gscore' in df.columns:
        metric_cols.append('gscore')
    
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

def analyze_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by causal structure pattern."""
    if 'pattern' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby('pattern').agg(agg_dict).round(3)
    
    return summary

def analyze_by_var_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by variable type (continuous vs mixed)."""
    if 'var_type_tag' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby('var_type_tag').agg(agg_dict).round(3)
    
    return summary

def analyze_by_equation_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by equation type (linear vs non-linear)."""
    if 'equation_type' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby('equation_type').agg(agg_dict).round(3)
    
    return summary

def analyze_by_noise_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by noise type (for Phase 3 experiments)."""
    if 'noise_type' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby('noise_type').agg(agg_dict).round(3)
    
    return summary

def analyze_by_root_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by root node distribution type (for Phase 3 experiments)."""
    if 'root_distribution_type' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean'
    }
    if 'sid' in df.columns:
        agg_dict['sid'] = ['mean', 'std']
    if 'gscore' in df.columns:
        agg_dict['gscore'] = ['mean', 'std']
    
    summary = df.groupby('root_distribution_type').agg(agg_dict).round(3)
    
    return summary

def analyze_by_edge_density(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by edge density (graph sparsity for Phase 3 experiments)."""
    if 'edge_density' not in df.columns:
        return pd.DataFrame()
    
    # Create density bins for better grouping
    density_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    density_labels = ['Very Sparse (0-0.2)', 'Sparse (0.2-0.4)', 'Medium (0.4-0.6)', 
                     'Dense (0.6-0.8)', 'Very Dense (0.8-1.0)']
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    df_copy['density_category'] = pd.cut(df_copy['edge_density'], 
                                         bins=density_bins, 
                                         labels=density_labels,
                                         include_lowest=True)
    
    summary = df_copy.groupby('density_category', observed=True).agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'normalized_shd': ['mean', 'std'],
        'pred_edge_count': 'mean',
        'edge_density': 'mean'  # Show actual mean density in each bin
    }).round(3)
    
    # Add SID and gscore if available
    if 'sid' in df.columns and 'sid' not in summary.columns.get_level_values(0):
        sid_stats = df_copy.groupby('density_category', observed=True)['sid'].agg(['mean', 'std']).round(3)
        summary = pd.concat([summary, sid_stats], axis=1, keys=[summary.columns.names[0], 'sid'])
    if 'gscore' in df.columns and 'gscore' not in summary.columns.get_level_values(0):
        gscore_stats = df_copy.groupby('density_category', observed=True)['gscore'].agg(['mean', 'std']).round(3)
        summary = pd.concat([summary, gscore_stats], axis=1, keys=[summary.columns.names[0], 'gscore'])
    
    return summary


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
    
    print(f"Total datasets: {df['dataset_dir'].nunique()}")
    print(f"Total algorithm runs: {len(df)}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}\n")
    
    # Algorithm performance summary
    print(f"{'='*60}")
    print("ALGORITHM PERFORMANCE SUMMARY")
    print(f"{'='*60}\n")
    
    summary = analyze_algorithm_performance(df)
    print(summary)
    
    # Compare with/without prior knowledge
    if 'use_prior' in df.columns:
        print(f"\n{'='*60}")
        print("COMPARISON: WITH vs WITHOUT PRIOR KNOWLEDGE")
        print(f"{'='*60}\n")
        
        comparison = compare_with_without_prior(df)
        if not comparison.empty:
            print(comparison.to_string(index=False))
    
    # Best performing algorithm by F1 Score
    print(f"\n{'='*60}")
    print("BEST PERFORMING ALGORITHMS (by F1 Score)")
    print(f"{'='*60}\n")
    
    best_algorithms = df.groupby('algorithm')['f1_score'].mean().sort_values(ascending=False)
    print(best_algorithms.round(3))
    
    # Best performing algorithm by Normalized SHD
    print(f"\n{'='*60}")
    print("BEST PERFORMING ALGORITHMS (by Normalized SHD)")
    print(f"{'='*60}\n")
    
    best_nshd = df.groupby('algorithm')['normalized_shd'].mean().sort_values(ascending=False)
    print(best_nshd.round(3))
    
    # Best performing algorithm by SID (if available)
    if 'sid' in df.columns:
        print(f"\n{'='*60}")
        print("BEST PERFORMING ALGORITHMS (by SID - Lower is Better)")
        print(f"{'='*60}\n")
        
        best_sid = df.groupby('algorithm')['sid'].mean().sort_values(ascending=True)
        print(best_sid.round(3))
    
    # Best performing algorithm by gscore (if available)
    if 'gscore' in df.columns:
        print(f"\n{'='*60}")
        print("BEST PERFORMING ALGORITHMS (by G-Score)")
        print(f"{'='*60}\n")
        
        best_gscore = df.groupby('algorithm')['gscore'].mean().sort_values(ascending=False)
        print(best_gscore.round(3))
    
    # Performance by Pattern Type
    print(f"\n{'='*60}")
    print("PERFORMANCE BY CAUSAL STRUCTURE PATTERN")
    print(f"{'='*60}\n")
    
    pattern_analysis = analyze_by_pattern(df)
    if not pattern_analysis.empty:
        print(pattern_analysis)
    else:
        print("No pattern data available")
    
    # Performance by Variable Type
    print(f"\n{'='*60}")
    print("PERFORMANCE BY VARIABLE TYPE (Continuous vs Mixed)")
    print(f"{'='*60}\n")
    
    var_type_analysis = analyze_by_var_type(df)
    if not var_type_analysis.empty:
        print(var_type_analysis)
    else:
        print("No variable type data available")
    
    # Performance by Equation Type
    print(f"\n{'='*60}")
    print("PERFORMANCE BY EQUATION TYPE (Linear vs Non-Linear)")
    print(f"{'='*60}\n")
    
    equation_analysis = analyze_by_equation_type(df)
    if not equation_analysis.empty:
        print(equation_analysis)
    else:
        print("No equation type data available")
    
    # Performance by Noise Type (Phase 3)
    print(f"\n{'='*60}")
    print("PERFORMANCE BY NOISE TYPE (Phase 3)")
    print(f"{'='*60}\n")
    
    noise_analysis = analyze_by_noise_type(df)
    if not noise_analysis.empty:
        print(noise_analysis)
    else:
        print("No noise type data available (Phase 3 only)")
    
    # Performance by Root Distribution (Phase 3)
    print(f"\n{'='*60}")
    print("PERFORMANCE BY ROOT DISTRIBUTION TYPE (Phase 3)")
    print(f"{'='*60}\n")
    
    root_dist_analysis = analyze_by_root_distribution(df)
    if not root_dist_analysis.empty:
        print(root_dist_analysis)
    else:
        print("No root distribution data available (Phase 3 only)")
    
    # Performance by Edge Density (Phase 3)
    print(f"\n{'='*60}")
    print("PERFORMANCE BY EDGE DENSITY (Phase 3)")
    print(f"{'='*60}\n")
    
    edge_density_analysis = analyze_by_edge_density(df)
    if not edge_density_analysis.empty:
        print(edge_density_analysis)
    else:
        print("No edge density variation data available")
    
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
            # Header
            f.write("="*70 + "\n")
            f.write("CAUSAL DISCOVERY EXPERIMENT ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total datasets: {df['dataset_dir'].nunique()}\n")
            f.write(f"Total algorithm runs: {len(df)}\n")
            f.write(f"Algorithms: {', '.join(df['algorithm'].unique().tolist())}\n\n")
            
            # Algorithm Performance Summary
            f.write("="*70 + "\n")
            f.write("ALGORITHM PERFORMANCE SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            # Prior Knowledge Comparison
            if 'use_prior' in df.columns and not comparison.empty:
                f.write("="*70 + "\n")
                f.write("COMPARISON: WITH vs WITHOUT PRIOR KNOWLEDGE\n")
                f.write("="*70 + "\n\n")
                f.write(comparison.to_string(index=False))
                f.write("\n\n")
            
            # Best Algorithms by F1 Score
            f.write("="*70 + "\n")
            f.write("BEST PERFORMING ALGORITHMS (by F1 Score)\n")
            f.write("="*70 + "\n\n")
            f.write(best_algorithms.to_string())
            f.write("\n\n")
            
            # Best Algorithms by Normalized SHD
            f.write("="*70 + "\n")
            f.write("BEST PERFORMING ALGORITHMS (by Normalized SHD)\n")
            f.write("="*70 + "\n\n")
            f.write(best_nshd.to_string())
            f.write("\n\n")
            
            # Best Algorithms by SID (if available)
            if 'sid' in df.columns:
                f.write("="*70 + "\n")
                f.write("BEST PERFORMING ALGORITHMS (by SID - Lower is Better)\n")
                f.write("="*70 + "\n\n")
                best_sid = df.groupby('algorithm')['sid'].mean().sort_values(ascending=True)
                f.write(best_sid.to_string())
                f.write("\n\n")
            
            # Best Algorithms by gscore (if available)
            if 'gscore' in df.columns:
                f.write("="*70 + "\n")
                f.write("BEST PERFORMING ALGORITHMS (by G-Score)\n")
                f.write("="*70 + "\n\n")
                best_gscore = df.groupby('algorithm')['gscore'].mean().sort_values(ascending=False)
                f.write(best_gscore.to_string())
                f.write("\n\n")
            
            # Performance by Pattern
            if not pattern_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY CAUSAL STRUCTURE PATTERN\n")
                f.write("="*70 + "\n\n")
                f.write(pattern_analysis.to_string())
                f.write("\n\n")
            
            # Performance by Variable Type
            if not var_type_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY VARIABLE TYPE (Continuous vs Mixed)\n")
                f.write("="*70 + "\n\n")
                f.write(var_type_analysis.to_string())
                f.write("\n\n")
            
            # Performance by Equation Type
            if not equation_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY EQUATION TYPE (Linear vs Non-Linear)\n")
                f.write("="*70 + "\n\n")
                f.write(equation_analysis.to_string())
                f.write("\n\n")
            
            # Performance by Noise Type (Phase 3)
            if not noise_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY NOISE TYPE (Phase 3)\n")
                f.write("="*70 + "\n\n")
                f.write(noise_analysis.to_string())
                f.write("\n\n")
            
            # Performance by Root Distribution (Phase 3)
            if not root_dist_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY ROOT DISTRIBUTION TYPE (Phase 3)\n")
                f.write("="*70 + "\n\n")
                f.write(root_dist_analysis.to_string())
                f.write("\n\n")
            
            # Performance by Edge Density (Phase 3)
            if not edge_density_analysis.empty:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE BY EDGE DENSITY (Phase 3)\n")
                f.write("="*70 + "\n\n")
                f.write(edge_density_analysis.to_string())
                f.write("\n\n")
            
            # Dataset Characteristic Correlations
            if not characteristics.empty:
                f.write("="*70 + "\n")
                f.write("DATASET CHARACTERISTIC CORRELATIONS\n")
                f.write("="*70 + "\n\n")
                f.write(characteristics.to_string(index=False))
                f.write("\n\n")
        
        print(f"\n{'='*60}")
        print(f"✓ Complete analysis saved to: {args.output}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

