#!/usr/bin/env python3
"""
CSuite Benchmark Report Generator

This script generates comprehensive benchmark reports for CSuite evaluation results,
including HTML reports, summary statistics, and comparative analysis.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json


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


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics."""
    logger = logging.getLogger(__name__)
    logger.info("Generating summary statistics...")
    
    summary = {
        'overview': {
            'total_evaluations': len(df),
            'unique_algorithms': df['algorithm'].nunique(),
            'unique_patterns': df['pattern'].nunique(),
            'unique_datasets': df.groupby(['pattern', 'num_nodes']).size().sum(),
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'algorithm_performance': {},
        'pattern_performance': {},
        'structural_analysis': {}
    }
    
    # Algorithm performance summary
    alg_stats = df.groupby('algorithm').agg({
        'shd': ['count', 'mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    alg_stats.columns = ['_'.join(col).strip() for col in alg_stats.columns]
    summary['algorithm_performance'] = alg_stats.to_dict('index')
    
    # Pattern performance summary
    pattern_stats = df.groupby('pattern').agg({
        'shd': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(4)
    
    pattern_stats.columns = ['_'.join(col).strip() for col in pattern_stats.columns]
    summary['pattern_performance'] = pattern_stats.to_dict('index')
    
    # Structural analysis
    df['num_nodes'] = df['num_nodes'].astype(int)
    df['num_edges'] = df['num_edges'].astype(int)
    
    node_stats = df.groupby('num_nodes').agg({
        'f1_score': 'mean',
        'shd': 'mean'
    }).round(4)
    
    summary['structural_analysis'] = {
        'by_node_count': node_stats.to_dict('index'),
        'overall_edge_density': (df['num_edges'] / (df['num_nodes'] * (df['num_nodes'] - 1))).mean(),
        'equation_type_distribution': df['equation_type'].value_counts().to_dict()
    }
    
    logger.info("Summary statistics generated")
    return summary


def generate_html_report(df: pd.DataFrame, summary: Dict, output_file: str):
    """Generate HTML benchmark report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating HTML report...")
    
    # Get best performing algorithm
    best_algorithm = df.groupby('algorithm')['f1_score'].mean().idxmax()
    best_f1 = df.groupby('algorithm')['f1_score'].mean().max()
    
    # Get pattern-specific best algorithms
    pattern_best = df.groupby(['pattern', 'algorithm'])['f1_score'].mean().groupby('pattern').idxmax()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CSuite Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .best {{ background-color: #d4edda; }}
        .worst {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CSuite Benchmark Report</h1>
        <p>Generated on: {summary['overview']['date_generated']}</p>
    </div>
    
    <div class="section">
        <h2>Overview</h2>
        <div class="metric">Total Evaluations: {summary['overview']['total_evaluations']}</div>
        <div class="metric">Algorithms: {summary['overview']['unique_algorithms']}</div>
        <div class="metric">Patterns: {summary['overview']['unique_patterns']}</div>
        <div class="metric">Datasets: {summary['overview']['unique_datasets']}</div>
    </div>
    
    <div class="section">
        <h2>Best Performing Algorithm</h2>
        <p><strong>{best_algorithm}</strong> with F1 Score: {best_f1:.4f}</p>
    </div>
    
    <div class="section">
        <h2>Algorithm Performance Summary</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>F1 Score (Mean)</th>
                <th>F1 Score (Std)</th>
                <th>SHD (Mean)</th>
                <th>SHD (Std)</th>
                <th>Precision (Mean)</th>
                <th>Recall (Mean)</th>
            </tr>
"""
    
    # Add algorithm performance rows
    for algorithm, stats in summary['algorithm_performance'].items():
        f1_mean = stats['f1_score_mean']
        f1_std = stats['f1_score_std']
        shd_mean = stats['shd_mean']
        shd_std = stats['shd_std']
        precision = stats['precision_mean']
        recall = stats['recall_mean']
        
        row_class = "best" if algorithm == best_algorithm else ""
        
        html_content += f"""
            <tr class="{row_class}">
                <td>{algorithm}</td>
                <td>{f1_mean:.4f}</td>
                <td>{f1_std:.4f}</td>
                <td>{shd_mean:.4f}</td>
                <td>{shd_std:.4f}</td>
                <td>{precision:.4f}</td>
                <td>{recall:.4f}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Pattern-Specific Performance</h2>
        <table>
            <tr>
                <th>Pattern</th>
                <th>F1 Score (Mean)</th>
                <th>F1 Score (Std)</th>
                <th>SHD (Mean)</th>
                <th>Best Algorithm</th>
            </tr>
"""
    
    # Add pattern performance rows
    for pattern, stats in summary['pattern_performance'].items():
        f1_mean = stats['f1_score_mean']
        f1_std = stats['f1_score_std']
        shd_mean = stats['shd_mean']
        best_alg = pattern_best[pattern][1] if pattern in pattern_best else "N/A"
        
        html_content += f"""
            <tr>
                <td>{pattern}</td>
                <td>{f1_mean:.4f}</td>
                <td>{f1_std:.4f}</td>
                <td>{shd_mean:.4f}</td>
                <td>{best_alg}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Structural Analysis</h2>
        <h3>Performance by Node Count</h3>
        <table>
            <tr>
                <th>Node Count</th>
                <th>F1 Score (Mean)</th>
                <th>SHD (Mean)</th>
            </tr>
"""
    
    # Add node count analysis
    for node_count, stats in summary['structural_analysis']['by_node_count'].items():
        f1_mean = stats['f1_score']
        shd_mean = stats['shd']
        
        html_content += f"""
            <tr>
                <td>{node_count}</td>
                <td>{f1_mean:.4f}</td>
                <td>{shd_mean:.4f}</td>
            </tr>
"""
    
    html_content += f"""
        </table>
        
        <h3>Additional Statistics</h3>
        <p>Overall Edge Density: {summary['structural_analysis']['overall_edge_density']:.4f}</p>
        <p>Equation Type Distribution: {summary['structural_analysis']['equation_type_distribution']}</p>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li><strong>Best Overall Algorithm:</strong> {best_algorithm} performs best across all patterns</li>
            <li><strong>Pattern-Specific:</strong> Consider using different algorithms for different pattern types</li>
            <li><strong>Node Count Impact:</strong> Performance generally decreases with increasing node count</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_file}")


def generate_json_report(summary: Dict, output_file: str):
    """Generate JSON benchmark report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating JSON report...")
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"JSON report saved to {output_file}")


def generate_csv_summary(df: pd.DataFrame, output_file: str):
    """Generate CSV summary of results."""
    logger = logging.getLogger(__name__)
    logger.info("Generating CSV summary...")
    
    # Create summary by algorithm
    alg_summary = df.groupby('algorithm').agg({
        'shd': ['count', 'mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    alg_summary.columns = ['_'.join(col).strip() for col in alg_summary.columns]
    alg_summary.to_csv(output_file)
    
    logger.info(f"CSV summary saved to {output_file}")


def main():
    """Main report generation function."""
    parser = argparse.ArgumentParser(
        description="Generate CSuite benchmark reports",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-file', type=str,
                       default='csuite_benchmark/results/causal_discovery_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='csuite_benchmark/results',
                       help='Output directory for reports')
    parser.add_argument('--formats', nargs='+',
                       default=['html', 'json', 'csv'],
                       choices=['html', 'json', 'csv'],
                       help='Report formats to generate')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting CSuite benchmark report generation...")
    
    try:
        # Load results
        df = load_results(args.results_file)
        
        # Generate summary statistics
        summary = generate_summary_statistics(df)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate reports in requested formats
        if 'html' in args.formats:
            html_file = os.path.join(args.output_dir, "csuite_benchmark_report.html")
            generate_html_report(df, summary, html_file)
        
        if 'json' in args.formats:
            json_file = os.path.join(args.output_dir, "csuite_benchmark_summary.json")
            generate_json_report(summary, json_file)
        
        if 'csv' in args.formats:
            csv_file = os.path.join(args.output_dir, "algorithm_performance_summary.csv")
            generate_csv_summary(df, csv_file)
        
        # Print key findings
        logger.info("Key Findings:")
        best_algorithm = df.groupby('algorithm')['f1_score'].mean().idxmax()
        best_f1 = df.groupby('algorithm')['f1_score'].mean().max()
        logger.info(f"Best Algorithm: {best_algorithm} (F1: {best_f1:.4f})")
        
        worst_algorithm = df.groupby('algorithm')['f1_score'].mean().idxmin()
        worst_f1 = df.groupby('algorithm')['f1_score'].mean().min()
        logger.info(f"Worst Algorithm: {worst_algorithm} (F1: {worst_f1:.4f})")
        
        logger.info("Benchmark report generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
