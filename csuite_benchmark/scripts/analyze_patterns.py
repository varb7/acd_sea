#!/usr/bin/env python3
"""
CSuite Pattern Analysis Script

This script provides detailed analysis of causal discovery performance
across different CSuite patterns and structural characteristics.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add the inference pipeline to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "inference_pipeline"))


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_results(results_file: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} evaluation results")
    return df


def analyze_pattern_performance(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze performance by pattern type."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing pattern-specific performance...")
    
    pattern_analysis = {}
    
    for pattern in df['pattern'].unique():
        pattern_data = df[df['pattern'] == pattern]
        
        # Calculate statistics for each algorithm
        stats = pattern_data.groupby('algorithm').agg({
            'shd': ['count', 'mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'specificity': ['mean', 'std'],
            'sensitivity': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns]
        
        # Add pattern-specific insights
        stats['pattern'] = pattern
        stats['num_datasets'] = len(pattern_data)
        
        pattern_analysis[pattern] = stats
        
        logger.info(f"Pattern {pattern}: {len(pattern_data)} datasets, "
                   f"best F1: {stats['f1_score_mean'].max():.4f}")
    
    return pattern_analysis


def analyze_structural_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by structural characteristics."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing structural characteristics...")
    
    # Create structural features
    df['num_nodes'] = df['num_nodes'].astype(int)
    df['num_edges'] = df['num_edges'].astype(int)
    df['edge_density'] = df['num_edges'] / (df['num_nodes'] * (df['num_nodes'] - 1))
    df['is_linear'] = df['equation_type'] == 'linear'
    df['is_mixed'] = df['variable_types'].astype(str).str.contains('discrete')
    
    # Analyze by node count
    node_analysis = df.groupby('num_nodes').agg({
        'shd': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).round(4)
    
    # Analyze by edge density
    df['density_bin'] = pd.cut(df['edge_density'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    density_analysis = df.groupby('density_bin').agg({
        'shd': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).round(4)
    
    # Analyze by equation type
    equation_analysis = df.groupby('equation_type').agg({
        'shd': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).round(4)
    
    logger.info("Structural analysis completed")
    
    return {
        'node_count': node_analysis,
        'edge_density': density_analysis,
        'equation_type': equation_analysis
    }


def create_performance_visualizations(df: pd.DataFrame, output_dir: str):
    """Create performance visualization plots."""
    logger = logging.getLogger(__name__)
    logger.info("Creating performance visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Algorithm performance comparison
    plt.figure(figsize=(12, 8))
    algorithm_perf = df.groupby('algorithm')['f1_score'].mean().sort_values(ascending=False)
    algorithm_perf.plot(kind='bar')
    plt.title('Algorithm Performance (F1 Score)')
    plt.xlabel('Algorithm')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pattern-specific performance
    plt.figure(figsize=(14, 8))
    pattern_perf = df.groupby(['pattern', 'algorithm'])['f1_score'].mean().unstack()
    pattern_perf.plot(kind='bar', figsize=(14, 8))
    plt.title('Pattern-Specific Algorithm Performance')
    plt.xlabel('Pattern')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_algorithm_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance vs Node Count
    plt.figure(figsize=(10, 6))
    node_perf = df.groupby(['num_nodes', 'algorithm'])['f1_score'].mean().unstack()
    node_perf.plot(kind='line', marker='o')
    plt.title('Performance vs Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('F1 Score')
    plt.legend(title='Algorithm')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_nodes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SHD vs F1 Score scatter plot
    plt.figure(figsize=(10, 8))
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        plt.scatter(alg_data['shd'], alg_data['f1_score'], 
                   label=algorithm, alpha=0.6, s=50)
    
    plt.xlabel('Structural Hamming Distance (SHD)')
    plt.ylabel('F1 Score')
    plt.title('SHD vs F1 Score by Algorithm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shd_vs_f1_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap of pattern-algorithm performance
    plt.figure(figsize=(12, 8))
    heatmap_data = df.groupby(['pattern', 'algorithm'])['f1_score'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Pattern-Algorithm Performance Heatmap')
    plt.xlabel('Algorithm')
    plt.ylabel('Pattern')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_algorithm_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def generate_pattern_insights(pattern_analysis: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Generate insights about each pattern's characteristics."""
    insights = {}
    
    for pattern, stats in pattern_analysis.items():
        best_algorithm = stats['f1_score_mean'].idxmax()
        best_f1 = stats['f1_score_mean'].max()
        worst_algorithm = stats['f1_score_mean'].idxmin()
        worst_f1 = stats['f1_score_mean'].min()
        
        # Calculate performance spread
        f1_std = stats['f1_score_mean'].std()
        
        insight = f"""
Pattern: {pattern}
- Best Algorithm: {best_algorithm} (F1: {best_f1:.4f})
- Worst Algorithm: {worst_algorithm} (F1: {worst_f1:.4f})
- Performance Spread: {f1_std:.4f}
- Number of Datasets: {stats['num_datasets'].iloc[0]}
- Average SHD: {stats['shd_mean'].mean():.4f}
        """.strip()
        
        insights[pattern] = insight
    
    return insights


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze CSuite pattern performance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-file', type=str,
                       default='csuite_benchmark/results/causal_discovery_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='csuite_benchmark/results/pattern_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting CSuite pattern analysis...")
    
    try:
        # Load results
        df = load_results(args.results_file)
        
        # Analyze pattern performance
        pattern_analysis = analyze_pattern_performance(df)
        
        # Analyze structural characteristics
        structural_analysis = analyze_structural_characteristics(df)
        
        # Generate insights
        insights = generate_pattern_insights(pattern_analysis)
        
        # Save analysis results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save pattern analysis
        for pattern, stats in pattern_analysis.items():
            pattern_file = os.path.join(args.output_dir, f"{pattern}_detailed_analysis.csv")
            stats.to_csv(pattern_file)
            logger.info(f"Pattern analysis for {pattern} saved to {pattern_file}")
        
        # Save structural analysis
        for analysis_type, analysis_data in structural_analysis.items():
            struct_file = os.path.join(args.output_dir, f"structural_{analysis_type}_analysis.csv")
            analysis_data.to_csv(struct_file)
            logger.info(f"Structural {analysis_type} analysis saved to {struct_file}")
        
        # Save insights
        insights_file = os.path.join(args.output_dir, "pattern_insights.txt")
        with open(insights_file, 'w') as f:
            for pattern, insight in insights.items():
                f.write(f"{insight}\n\n")
        logger.info(f"Pattern insights saved to {insights_file}")
        
        # Create visualizations if requested
        if args.create_plots:
            plots_dir = os.path.join(args.output_dir, "plots")
            create_performance_visualizations(df, plots_dir)
        
        # Print summary
        logger.info("Analysis Summary:")
        for pattern, insight in insights.items():
            print(f"\n{insight}")
        
        logger.info("Pattern analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
