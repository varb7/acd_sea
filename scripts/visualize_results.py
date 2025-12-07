"""
Comprehensive Visualization Script for Causal Discovery Experiments

This script generates publication-quality figures for analyzing:
1. Algorithm performance comparison
2. Factor effects (linearity, noise, distribution, edge density)
3. Prior knowledge effectiveness
4. Algorithm robustness across challenging conditions

Usage:
    # Generate all standard figures
    python visualize_results.py --results path/to/results.csv --output figures/

    # Generate specific figure type
    python visualize_results.py --results path/to/results.csv --figure algorithm_comparison

    # Interactive mode
    python visualize_results.py --results path/to/results.csv --interactive
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color palette for algorithms
ALGO_COLORS = {
    # PyTetrad algorithms (blues/greens)
    'TetradFGES': '#1f77b4',
    'TetradRFCI': '#2ca02c',
    'TetradGFCI': '#17becf',
    'TetradFCI': '#9467bd',
    'TetradCFCI': '#8c564b',
    'TetradCPC': '#e377c2',
    'TetradPC': '#7f7f7f',
    'TetradFCIMax': '#bcbd22',
    'TetradBossFCI': '#ff7f0e',
    'TetradGraspFCI': '#d62728',
    'TetradSpFCI': '#aec7e8',
    # Causal-learn baselines (reds/oranges)
    'CausalLearnGES': '#ff9896',
    'CausalLearnFCI': '#c49c94',
}

# Metrics configuration
METRICS = {
    'f1_score': {'label': 'F1 Score', 'higher_better': True, 'format': '.3f'},
    'precision': {'label': 'Precision', 'higher_better': True, 'format': '.3f'},
    'recall': {'label': 'Recall', 'higher_better': True, 'format': '.3f'},
    'shd': {'label': 'SHD', 'higher_better': False, 'format': '.1f'},
    'normalized_shd': {'label': 'Normalized SHD', 'higher_better': True, 'format': '.3f'},
    'execution_time': {'label': 'Execution Time (s)', 'higher_better': False, 'format': '.2f'},
}

# Factor configuration
FACTORS = {
    'equation_type': {'label': 'Equation Type', 'order': ['linear', 'non_linear']},
    'var_type_tag': {'label': 'Variable Type', 'order': ['continuous', 'mixed']},
    'noise_type': {'label': 'Noise Type', 'order': ['normal', 'uniform', 'exponential']},
    'root_distribution_type': {'label': 'Root Distribution', 'order': ['normal', 'exponential', 'uniform', 'beta']},
    'pattern': {'label': 'Graph Pattern', 'order': ['chain', 'backdoor', 'random_dag']},
    'edge_density': {'label': 'Edge Density', 'order': None},  # Numeric, no fixed order
    'edge_density_bin': {'label': 'Edge Density', 'order': ['Sparse (0-0.33)', 'Medium (0.33-0.66)', 'Dense (0.66-1.0)']},
}


class ResultsAnalyzer:
    """Analyze and visualize causal discovery experiment results."""
    
    def __init__(self, results_path: str):
        """Load results from CSV."""
        self.df = pd.read_csv(results_path)
        self._preprocess()
        print(f"Loaded {len(self.df)} experiment results")
        print(f"Algorithms: {self.df['algorithm'].unique().tolist()}")
        print(f"Datasets: {self.df['dataset_dir'].nunique()}")
        
    def _preprocess(self):
        """Preprocess the dataframe."""
        # Convert edge_density to numeric if possible
        if 'edge_density' in self.df.columns:
            self.df['edge_density'] = pd.to_numeric(self.df['edge_density'], errors='coerce')
            
            # Create binned edge_density for categorical analysis
            valid_density = self.df['edge_density'].dropna()
            if len(valid_density) > 0:
                self.df['edge_density_bin'] = pd.cut(
                    self.df['edge_density'],
                    bins=[0, 0.33, 0.66, 1.0],
                    labels=['Sparse (0-0.33)', 'Medium (0.33-0.66)', 'Dense (0.66-1.0)'],
                    include_lowest=True
                )
        
        # Ensure use_prior is boolean
        if 'use_prior' in self.df.columns:
            self.df['use_prior'] = self.df['use_prior'].astype(bool)
        
        # Create algorithm family column
        self.df['algo_family'] = self.df['algorithm'].apply(self._get_algo_family)
        
        # Create baseline indicator
        self.df['is_baseline'] = self.df['algorithm'].isin(['CausalLearnGES', 'CausalLearnFCI'])
        
    def _get_algo_family(self, algo: str) -> str:
        """Categorize algorithm into family."""
        if 'FCI' in algo or 'fci' in algo.lower():
            return 'FCI-based'
        elif 'GES' in algo or 'FGES' in algo:
            return 'Score-based'
        elif 'PC' in algo or 'CPC' in algo:
            return 'Constraint-based'
        else:
            return 'Other'
    
    # =========================================================================
    # Core Analysis Functions
    # =========================================================================
    
    def get_algorithm_summary(self, metric: str = 'f1_score') -> pd.DataFrame:
        """Get summary statistics for each algorithm."""
        summary = self.df.groupby(['algorithm', 'use_prior'])[metric].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        return summary.reset_index()
    
    def get_factor_effect(self, factor: str, metric: str = 'f1_score') -> pd.DataFrame:
        """Get the effect of a factor on performance."""
        return self.df.groupby(['algorithm', factor])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    def compute_robustness_scores(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compute robustness score for each algorithm.
        Robustness = performance on challenging conditions / baseline performance
        """
        # Define baseline condition
        baseline_mask = (
            (self.df['equation_type'] == 'linear') &
            (self.df['noise_type'] == 'normal') &
            (self.df['root_distribution_type'] == 'normal')
        )
        
        # Add edge density to baseline if present (sparse = baseline)
        if 'edge_density_bin' in self.df.columns:
            sparse_mask = self.df['edge_density_bin'] == 'Sparse (0-0.33)'
            # Only apply if we have sparse data
            if sparse_mask.any():
                baseline_mask = baseline_mask & (sparse_mask | self.df['edge_density_bin'].isna())
        
        results = []
        for algo in self.df['algorithm'].unique():
            algo_data = self.df[self.df['algorithm'] == algo]
            
            baseline_perf = algo_data[baseline_mask][metric].mean()
            
            # Calculate degradation for each factor
            factors_to_check = ['equation_type', 'noise_type', 'root_distribution_type']
            baseline_levels = {
                'equation_type': 'linear',
                'noise_type': 'normal', 
                'root_distribution_type': 'normal',
                'edge_density_bin': 'Sparse (0-0.33)'
            }
            
            # Add edge_density_bin if present
            if 'edge_density_bin' in algo_data.columns and algo_data['edge_density_bin'].notna().any():
                factors_to_check.append('edge_density_bin')
            
            for factor in factors_to_check:
                for level in algo_data[factor].dropna().unique():
                    # Skip baseline level
                    if level == baseline_levels.get(factor):
                        continue
                    
                    challenged_perf = algo_data[algo_data[factor] == level][metric].mean()
                    
                    if pd.notna(baseline_perf) and baseline_perf > 0:
                        robustness = challenged_perf / baseline_perf
                    else:
                        robustness = np.nan
                    
                    results.append({
                        'algorithm': algo,
                        'factor': factor,
                        'level': str(level),
                        'baseline_perf': baseline_perf,
                        'challenged_perf': challenged_perf,
                        'robustness': robustness
                    })
        
        return pd.DataFrame(results)
    
    def compute_prior_effectiveness(self, metric: str = 'f1_score') -> pd.DataFrame:
        """Compute the effectiveness of prior knowledge for each algorithm."""
        results = []
        
        for algo in self.df['algorithm'].unique():
            algo_data = self.df[self.df['algorithm'] == algo]
            
            with_prior = algo_data[algo_data['use_prior'] == True][metric].mean()
            without_prior = algo_data[algo_data['use_prior'] == False][metric].mean()
            
            improvement = with_prior - without_prior
            pct_improvement = (improvement / without_prior * 100) if without_prior > 0 else np.nan
            
            # Statistical significance
            prior_vals = algo_data[algo_data['use_prior'] == True][metric].dropna()
            no_prior_vals = algo_data[algo_data['use_prior'] == False][metric].dropna()
            
            if len(prior_vals) > 1 and len(no_prior_vals) > 1:
                _, p_value = stats.ttest_ind(prior_vals, no_prior_vals)
            else:
                p_value = np.nan
            
            results.append({
                'algorithm': algo,
                'with_prior': with_prior,
                'without_prior': without_prior,
                'improvement': improvement,
                'pct_improvement': pct_improvement,
                'p_value': p_value,
                'significant': p_value < 0.05 if pd.notna(p_value) else False
            })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # Visualization Functions
    # =========================================================================
    
    def plot_algorithm_comparison(self, metric: str = 'f1_score', 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 1: Overall algorithm comparison bar chart.
        Shows mean performance with error bars, split by prior knowledge.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data
        summary = self.df.groupby(['algorithm', 'use_prior'])[metric].agg(['mean', 'std']).reset_index()
        
        algorithms = sorted(self.df['algorithm'].unique())
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Plot bars
        for i, use_prior in enumerate([False, True]):
            data = summary[summary['use_prior'] == use_prior]
            means = [data[data['algorithm'] == a]['mean'].values[0] if a in data['algorithm'].values else 0 
                    for a in algorithms]
            stds = [data[data['algorithm'] == a]['std'].values[0] if a in data['algorithm'].values else 0 
                   for a in algorithms]
            
            label = 'With Prior' if use_prior else 'Without Prior'
            color = '#2ecc71' if use_prior else '#3498db'
            
            bars = ax.bar(x + i * width, means, width, label=label, color=color, 
                         yerr=stds, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(METRICS[metric]['label'])
        ax.set_title(f'Algorithm Comparison: {METRICS[metric]["label"]}')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
        
        # Add baseline markers
        for i, algo in enumerate(algorithms):
            if algo in ['CausalLearnGES', 'CausalLearnFCI']:
                ax.axvline(x=i + width/2, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_factor_effect(self, factor: str, metric: str = 'f1_score',
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 2: Effect of a specific factor on algorithm performance.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get factor levels
        levels = self.df[factor].dropna().unique()
        if FACTORS.get(factor, {}).get('order'):
            levels = [l for l in FACTORS[factor]['order'] if l in levels]
        else:
            levels = sorted(levels)
        
        # Prepare data
        algorithms = sorted(self.df['algorithm'].unique())
        x = np.arange(len(levels))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            algo_data = self.df[self.df['algorithm'] == algo]
            means = []
            stds = []
            for level in levels:
                level_data = algo_data[algo_data[factor] == level][metric]
                means.append(level_data.mean())
                stds.append(level_data.std())
            
            color = ALGO_COLORS.get(algo, f'C{i}')
            ax.bar(x + i * width, means, width, label=algo, color=color, 
                  yerr=stds, capsize=2, alpha=0.8)
        
        ax.set_xlabel(FACTORS.get(factor, {}).get('label', factor))
        ax.set_ylabel(METRICS[metric]['label'])
        ax.set_title(f'Effect of {FACTORS.get(factor, {}).get("label", factor)} on {METRICS[metric]["label"]}')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(levels, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_factor_heatmap(self, factor: str, metric: str = 'f1_score',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 3: Heatmap showing algorithm performance across factor levels.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot table
        pivot = self.df.pivot_table(
            values=metric, 
            index='algorithm', 
            columns=factor, 
            aggfunc='mean'
        )
        
        # Sort by mean performance
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': METRICS[metric]['label']})
        
        ax.set_xlabel(FACTORS.get(factor, {}).get('label', factor))
        ax.set_ylabel('Algorithm')
        ax.set_title(f'{METRICS[metric]["label"]} by Algorithm and {FACTORS.get(factor, {}).get("label", factor)}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_robustness_heatmap(self, metric: str = 'f1_score',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 4: Robustness heatmap showing performance retention under challenging conditions.
        """
        robustness_df = self.compute_robustness_scores(metric)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pivot table
        robustness_df['condition'] = robustness_df['factor'] + ': ' + robustness_df['level'].astype(str)
        pivot = robustness_df.pivot_table(
            values='robustness', 
            index='algorithm', 
            columns='condition', 
            aggfunc='mean'
        )
        
        # Sort by mean robustness
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        
        # Create heatmap with custom colormap
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, center=1.0,
                   ax=ax, cbar_kws={'label': 'Robustness (1.0 = no degradation)'},
                   vmin=0.5, vmax=1.2)
        
        ax.set_xlabel('Challenging Condition')
        ax.set_ylabel('Algorithm')
        ax.set_title(f'Algorithm Robustness: {METRICS[metric]["label"]} Retention Under Challenging Conditions')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_prior_effectiveness(self, metric: str = 'f1_score',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 5: Prior knowledge effectiveness comparison.
        """
        prior_df = self.compute_prior_effectiveness(metric)
        prior_df = prior_df.sort_values('improvement', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Absolute improvement
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in prior_df['improvement']]
        bars = ax1.barh(prior_df['algorithm'], prior_df['improvement'], color=colors, alpha=0.8)
        
        # Add significance markers
        for i, (_, row) in enumerate(prior_df.iterrows()):
            if row['significant']:
                ax1.text(row['improvement'], i, ' *', va='center', fontsize=14, fontweight='bold')
        
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.set_xlabel(f'Improvement in {METRICS[metric]["label"]}')
        ax1.set_ylabel('Algorithm')
        ax1.set_title('Prior Knowledge Effect (Absolute)')
        
        # Right: Percentage improvement
        pct_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in prior_df['pct_improvement'].fillna(0)]
        ax2.barh(prior_df['algorithm'], prior_df['pct_improvement'].fillna(0), color=pct_colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Percentage Improvement (%)')
        ax2.set_title('Prior Knowledge Effect (Relative)')
        
        plt.suptitle(f'Effectiveness of Prior Knowledge on {METRICS[metric]["label"]}', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_edge_density_effect(self, metric: str = 'f1_score',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 6: Effect of edge density on performance.
        Works with any pattern that has varying edge_density values.
        """
        # Filter to rows with valid numeric edge_density
        density_data = self.df[self.df['edge_density'].notna()].copy()
        
        # Check if we have varying edge densities
        unique_densities = density_data['edge_density'].nunique()
        
        if density_data.empty or unique_densities < 2:
            print(f"No edge density variation in data (unique values: {unique_densities})")
            print("Edge density plot requires data from experiment_grid_comprehensive_random.yaml")
            print("Run: python generate_csuite_grid.py --config experiment_grid_comprehensive_random.yaml")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if we have random_dag pattern
        patterns_with_density = density_data['pattern'].unique()
        title_pattern = 'Random DAG' if 'random_dag' in patterns_with_density else ', '.join(patterns_with_density)
        
        for algo in sorted(density_data['algorithm'].unique()):
            algo_data = density_data[density_data['algorithm'] == algo]
            grouped = algo_data.groupby('edge_density')[metric].agg(['mean', 'std']).reset_index()
            
            if len(grouped) < 2:
                continue  # Skip if not enough data points
            
            color = ALGO_COLORS.get(algo, None)
            ax.errorbar(grouped['edge_density'], grouped['mean'], yerr=grouped['std'],
                       marker='o', label=algo, color=color, capsize=3, linewidth=2, markersize=6)
        
        ax.set_xlabel('Edge Density')
        ax.set_ylabel(METRICS[metric]['label'])
        ax.set_title(f'Effect of Edge Density on {METRICS[metric]["label"]} ({title_pattern})')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_tetrad_vs_baseline(self, metric: str = 'f1_score',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 7: Comparison of PyTetrad variants vs causal-learn baselines.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # GES comparison
        ax1 = axes[0]
        ges_algos = ['TetradFGES', 'CausalLearnGES']
        ges_data = self.df[self.df['algorithm'].isin(ges_algos)]
        
        for algo in ges_algos:
            data = ges_data[ges_data['algorithm'] == algo][metric]
            color = ALGO_COLORS.get(algo, None)
            ax1.hist(data, bins=20, alpha=0.6, label=algo, color=color)
        
        ax1.set_xlabel(METRICS[metric]['label'])
        ax1.set_ylabel('Count')
        ax1.set_title('GES Variants: TetradFGES vs CausalLearnGES')
        ax1.legend()
        
        # Add mean lines
        for algo in ges_algos:
            mean_val = ges_data[ges_data['algorithm'] == algo][metric].mean()
            color = ALGO_COLORS.get(algo, None)
            ax1.axvline(mean_val, color=color, linestyle='--', linewidth=2)
        
        # FCI comparison
        ax2 = axes[1]
        fci_tetrad = ['TetradFCI', 'TetradRFCI', 'TetradGFCI', 'TetradCFCI', 
                      'TetradBossFCI', 'TetradGraspFCI', 'TetradSpFCI', 'TetradFCIMax']
        fci_algos = [a for a in fci_tetrad if a in self.df['algorithm'].unique()] + ['CausalLearnFCI']
        
        summary = self.df[self.df['algorithm'].isin(fci_algos)].groupby('algorithm')[metric].agg(['mean', 'std'])
        summary = summary.sort_values('mean', ascending=False)
        
        colors = [ALGO_COLORS.get(a, 'gray') for a in summary.index]
        bars = ax2.barh(summary.index, summary['mean'], xerr=summary['std'], 
                       color=colors, alpha=0.8, capsize=3)
        
        # Highlight baseline
        for i, algo in enumerate(summary.index):
            if algo == 'CausalLearnFCI':
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)
        
        ax2.set_xlabel(METRICS[metric]['label'])
        ax2.set_title('FCI Variants Comparison')
        
        plt.suptitle(f'PyTetrad vs Causal-Learn Baselines: {METRICS[metric]["label"]}', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_execution_time(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 8: Execution time comparison.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate mean execution time per algorithm
        time_data = self.df.groupby('algorithm')['execution_time'].agg(['mean', 'std']).reset_index()
        time_data = time_data.sort_values('mean', ascending=True)
        
        colors = [ALGO_COLORS.get(a, 'gray') for a in time_data['algorithm']]
        bars = ax.barh(time_data['algorithm'], time_data['mean'], 
                      xerr=time_data['std'], color=colors, alpha=0.8, capsize=3)
        
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_ylabel('Algorithm')
        ax.set_title('Algorithm Execution Time Comparison')
        
        # Log scale if there's high variance
        if time_data['mean'].max() / time_data['mean'].min() > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_multi_metric_radar(self, algorithms: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 9: Radar chart comparing algorithms across multiple metrics.
        """
        if algorithms is None:
            # Select top 5 by F1 score
            top_algos = self.df.groupby('algorithm')['f1_score'].mean().nlargest(5).index.tolist()
            algorithms = top_algos
        
        metrics_to_plot = ['f1_score', 'precision', 'recall', 'normalized_shd']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        for algo in algorithms:
            values = []
            for m in metrics_to_plot:
                val = self.df[self.df['algorithm'] == algo][m].mean()
                values.append(val)
            values += values[:1]  # Complete the loop
            
            color = ALGO_COLORS.get(algo, None)
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRICS[m]['label'] for m in metrics_to_plot])
        ax.set_title('Multi-Metric Algorithm Comparison', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_node_sample_interaction(self, metric: str = 'f1_score',
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 10: Node × Sample interaction heatmap for top algorithms.
        Shows how each algorithm performs across different node counts and sample sizes.
        Critical for understanding SEA Framework operating conditions.
        """
        # Check if we have variation in both dimensions
        unique_nodes = self.df['num_nodes'].nunique()
        unique_samples = self.df['num_samples'].nunique()
        
        if unique_nodes < 2 or unique_samples < 2:
            print(f"Insufficient variation for node×sample analysis:")
            print(f"  Unique node counts: {unique_nodes} (need ≥2)")
            print(f"  Unique sample sizes: {unique_samples} (need ≥2)")
            print("Run experiment_grid_sea_aligned.yaml to generate appropriate data.")
            return None
        
        # Get top 6 algorithms by overall performance
        top_algos = self.df.groupby('algorithm')[metric].mean().nlargest(6).index.tolist()
        
        # Create subplots
        n_algos = len(top_algos)
        n_cols = 3
        n_rows = (n_algos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_algos > 1 else [axes]
        
        for idx, algo in enumerate(top_algos):
            ax = axes[idx]
            algo_data = self.df[self.df['algorithm'] == algo]
            
            pivot = algo_data.pivot_table(
                values=metric,
                index='num_nodes',
                columns='num_samples',
                aggfunc='mean'
            )
            
            if pivot.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(algo)
                continue
            
            # Sort index and columns
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot[sorted(pivot.columns)]
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax, cbar_kws={'label': METRICS[metric]['label']},
                       vmin=0, vmax=1 if metric in ['f1_score', 'precision', 'recall', 'normalized_shd'] else None)
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Number of Nodes')
            ax.set_title(f'{algo}')
        
        # Hide unused subplots
        for idx in range(len(top_algos), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{METRICS[metric]["label"]} by Node Count × Sample Size\n(Top 6 Algorithms)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_best_algorithm_by_condition(self, metric: str = 'f1_score',
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 11: Which algorithm is best for each node×sample combination.
        Directly answers: "What algorithm should SEA Framework use?"
        """
        # Check if we have variation in both dimensions
        unique_nodes = self.df['num_nodes'].nunique()
        unique_samples = self.df['num_samples'].nunique()
        
        if unique_nodes < 2 or unique_samples < 2:
            print(f"Insufficient variation for best-algorithm analysis.")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Best algorithm heatmap
        ax1 = axes[0]
        
        # Find best algorithm for each (nodes, samples) combination
        best_algo = self.df.groupby(['num_nodes', 'num_samples']).apply(
            lambda x: x.loc[x[metric].idxmax(), 'algorithm'] if len(x) > 0 else None
        ).unstack()
        
        # Create numeric encoding for algorithms
        all_algos = self.df['algorithm'].unique()
        algo_to_num = {algo: i for i, algo in enumerate(sorted(all_algos))}
        
        best_algo_numeric = best_algo.applymap(lambda x: algo_to_num.get(x, -1) if pd.notna(x) else -1)
        
        # Plot heatmap with algorithm names as annotations
        im = ax1.imshow(best_algo_numeric.values, cmap='tab20', aspect='auto')
        
        # Add text annotations
        for i in range(len(best_algo.index)):
            for j in range(len(best_algo.columns)):
                algo_name = best_algo.iloc[i, j]
                if pd.notna(algo_name):
                    # Shorten algorithm name for display
                    short_name = algo_name.replace('Tetrad', 'T').replace('CausalLearn', 'CL')
                    ax1.text(j, i, short_name, ha='center', va='center', fontsize=8,
                            color='white' if algo_to_num.get(algo_name, 0) < len(all_algos)/2 else 'black')
        
        ax1.set_xticks(range(len(best_algo.columns)))
        ax1.set_xticklabels(best_algo.columns)
        ax1.set_yticks(range(len(best_algo.index)))
        ax1.set_yticklabels(best_algo.index)
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title(f'Best Algorithm per Condition\n(by {METRICS[metric]["label"]})')
        
        # Right: Performance of best algorithm in each condition
        ax2 = axes[1]
        
        best_perf = self.df.groupby(['num_nodes', 'num_samples'])[metric].max().unstack()
        
        sns.heatmap(best_perf, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2,
                   cbar_kws={'label': METRICS[metric]['label']})
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title(f'Best Achievable {METRICS[metric]["label"]}\nper Condition')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_scalability_analysis(self, metric: str = 'f1_score',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Figure 12: How algorithm performance scales with complexity.
        Shows performance degradation as nodes and samples change.
        """
        # Check if we have variation
        unique_nodes = sorted(self.df['num_nodes'].unique())
        unique_samples = sorted(self.df['num_samples'].unique())
        
        if len(unique_nodes) < 2 and len(unique_samples) < 2:
            print("Insufficient data for scalability analysis.")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Performance vs Node Count (averaged over samples)
        ax1 = axes[0]
        for algo in self.df['algorithm'].unique():
            algo_data = self.df[self.df['algorithm'] == algo]
            grouped = algo_data.groupby('num_nodes')[metric].agg(['mean', 'std']).reset_index()
            
            if len(grouped) < 2:
                continue
            
            color = ALGO_COLORS.get(algo, None)
            ax1.errorbar(grouped['num_nodes'], grouped['mean'], yerr=grouped['std'],
                        marker='o', label=algo, color=color, capsize=3, linewidth=2)
        
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel(METRICS[metric]['label'])
        ax1.set_title(f'{METRICS[metric]["label"]} vs Node Count')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Right: Performance vs Sample Size (averaged over nodes)
        ax2 = axes[1]
        for algo in self.df['algorithm'].unique():
            algo_data = self.df[self.df['algorithm'] == algo]
            grouped = algo_data.groupby('num_samples')[metric].agg(['mean', 'std']).reset_index()
            
            if len(grouped) < 2:
                continue
            
            color = ALGO_COLORS.get(algo, None)
            ax2.errorbar(grouped['num_samples'], grouped['mean'], yerr=grouped['std'],
                        marker='o', label=algo, color=color, capsize=3, linewidth=2)
        
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel(METRICS[metric]['label'])
        ax2.set_title(f'{METRICS[metric]["label"]} vs Sample Size')
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Algorithm Scalability Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    def print_sea_recommendation(self, metric: str = 'f1_score'):
        """
        Print recommendation for SEA Framework based on node×sample analysis.
        """
        print("\n" + "=" * 70)
        print("SEA FRAMEWORK ALGORITHM RECOMMENDATION")
        print("=" * 70)
        
        # SEA default: 5 nodes, 500 samples
        sea_default = self.df[(self.df['num_nodes'] == 5) & (self.df['num_samples'] == 500)]
        if not sea_default.empty:
            print("\n📍 SEA Default Condition (5 nodes, 500 samples):")
            default_ranking = sea_default.groupby('algorithm')[metric].mean().sort_values(ascending=False)
            for i, (algo, score) in enumerate(default_ranking.head(5).items(), 1):
                print(f"   {i}. {algo}: {score:.4f}")
        else:
            print("\n⚠️  No data for SEA default condition (5 nodes, 500 samples)")
        
        # SEA max: 10 nodes, 1000 samples
        sea_max = self.df[(self.df['num_nodes'] == 10) & (self.df['num_samples'] == 1000)]
        if not sea_max.empty:
            print("\n📍 SEA Maximum Condition (10 nodes, 1000 samples):")
            max_ranking = sea_max.groupby('algorithm')[metric].mean().sort_values(ascending=False)
            for i, (algo, score) in enumerate(max_ranking.head(5).items(), 1):
                print(f"   {i}. {algo}: {score:.4f}")
        else:
            print("\n⚠️  No data for SEA maximum condition (10 nodes, 1000 samples)")
        
        # Overall recommendation
        print("\n📊 Overall Ranking (across all conditions):")
        overall = self.df.groupby('algorithm')[metric].mean().sort_values(ascending=False)
        for i, (algo, score) in enumerate(overall.head(5).items(), 1):
            print(f"   {i}. {algo}: {score:.4f}")
        
        # Consistency check
        print("\n🔍 Consistency Analysis:")
        best_at_default = sea_default.groupby('algorithm')[metric].mean().idxmax() if not sea_default.empty else None
        best_at_max = sea_max.groupby('algorithm')[metric].mean().idxmax() if not sea_max.empty else None
        best_overall = overall.idxmax()
        
        if best_at_default and best_at_max:
            if best_at_default == best_at_max == best_overall:
                print(f"   ✅ CONSISTENT: {best_overall} is best across all conditions")
            else:
                print(f"   ⚠️  INCONSISTENT rankings detected:")
                print(f"      - Best at 5/500:  {best_at_default}")
                print(f"      - Best at 10/1000: {best_at_max}")
                print(f"      - Best overall:    {best_overall}")
                print(f"   💡 Consider using condition-specific algorithms in SEA Framework")
        
        print("=" * 70)
    
    def generate_all_figures(self, output_dir: str, metric: str = 'f1_score'):
        """Generate all standard figures and save to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating figures in {output_dir}...")
        print("=" * 50)
        
        # Figure 1: Algorithm comparison
        self.plot_algorithm_comparison(metric, output_path / 'fig1_algorithm_comparison.png')
        
        # Figure 2-3: Factor effects
        factors_to_plot = [
            'equation_type', 'noise_type', 'root_distribution_type', 
            'var_type_tag', 'pattern', 'edge_density_bin'
        ]
        for factor in factors_to_plot:
            if factor in self.df.columns and self.df[factor].nunique() > 1:
                self.plot_factor_effect(factor, metric, 
                    output_path / f'fig2_{factor}_effect.png')
                self.plot_factor_heatmap(factor, metric,
                    output_path / f'fig3_{factor}_heatmap.png')
        
        # Figure 4: Robustness heatmap
        self.plot_robustness_heatmap(metric, output_path / 'fig4_robustness_heatmap.png')
        
        # Figure 5: Prior effectiveness
        self.plot_prior_effectiveness(metric, output_path / 'fig5_prior_effectiveness.png')
        
        # Figure 6: Edge density effect
        self.plot_edge_density_effect(metric, output_path / 'fig6_edge_density.png')
        
        # Figure 7: Tetrad vs baseline
        self.plot_tetrad_vs_baseline(metric, output_path / 'fig7_tetrad_vs_baseline.png')
        
        # Figure 8: Execution time
        self.plot_execution_time(output_path / 'fig8_execution_time.png')
        
        # Figure 9: Radar chart
        self.plot_multi_metric_radar(save_path=output_path / 'fig9_radar_chart.png')
        
        # Figure 10-12: Node × Sample interaction (for SEA Framework analysis)
        self.plot_node_sample_interaction(metric, output_path / 'fig10_node_sample_interaction.png')
        self.plot_best_algorithm_by_condition(metric, output_path / 'fig11_best_algorithm_per_condition.png')
        self.plot_scalability_analysis(metric, output_path / 'fig12_scalability_analysis.png')
        
        # Print SEA Framework recommendation
        self.print_sea_recommendation(metric)
        
        print("=" * 50)
        print(f"All figures saved to {output_dir}")
    
    def print_summary_table(self, metric: str = 'f1_score'):
        """Print summary statistics table."""
        print("\n" + "=" * 80)
        print(f"ALGORITHM PERFORMANCE SUMMARY ({METRICS[metric]['label']})")
        print("=" * 80)
        
        summary = self.df.groupby('algorithm').agg({
            metric: ['mean', 'std', 'min', 'max'],
            'execution_time': 'mean'
        }).round(4)
        
        summary.columns = [f'{metric}_mean', f'{metric}_std', f'{metric}_min', 
                          f'{metric}_max', 'time_mean']
        summary = summary.sort_values(f'{metric}_mean', ascending=False)
        
        print(summary.to_string())
        
        # Prior knowledge effectiveness
        print("\n" + "=" * 80)
        print("PRIOR KNOWLEDGE EFFECTIVENESS")
        print("=" * 80)
        
        prior_df = self.compute_prior_effectiveness(metric)
        prior_df = prior_df.sort_values('improvement', ascending=False)
        print(prior_df[['algorithm', 'without_prior', 'with_prior', 'improvement', 'significant']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Visualize causal discovery experiment results')
    parser.add_argument('--results', '-r', type=str, 
                       default='causal_discovery_results2/experiment_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', '-o', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--metric', '-m', type=str, default='f1_score',
                       choices=list(METRICS.keys()),
                       help='Primary metric for analysis')
    parser.add_argument('--figure', '-f', type=str, default='all',
                       help='Specific figure to generate (or "all")')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode (requires tkinter)')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Print summary table')
    
    args = parser.parse_args()
    
    # Load data
    analyzer = ResultsAnalyzer(args.results)
    
    if args.summary:
        analyzer.print_summary_table(args.metric)
    
    if args.interactive:
        # Import and run interactive visualizer
        try:
            from visualize_experiment_v2 import InteractiveViz
            import tkinter as tk
            root = tk.Tk()
            app = InteractiveViz(root)
            root.mainloop()
        except ImportError:
            print("Interactive mode requires visualize_experiment_v2.py")
            print("Falling back to batch generation...")
            analyzer.generate_all_figures(args.output, args.metric)
    elif args.figure == 'all':
        analyzer.generate_all_figures(args.output, args.metric)
    else:
        # Generate specific figure
        figure_methods = {
            'algorithm_comparison': analyzer.plot_algorithm_comparison,
            'robustness': analyzer.plot_robustness_heatmap,
            'prior': analyzer.plot_prior_effectiveness,
            'edge_density': analyzer.plot_edge_density_effect,
            'tetrad_vs_baseline': analyzer.plot_tetrad_vs_baseline,
            'execution_time': analyzer.plot_execution_time,
            'radar': analyzer.plot_multi_metric_radar,
            'node_sample': analyzer.plot_node_sample_interaction,
            'best_algorithm': analyzer.plot_best_algorithm_by_condition,
            'scalability': analyzer.plot_scalability_analysis,
        }
        
        # Special handling for SEA recommendation (no figure, just print)
        if args.figure == 'sea_recommendation':
            analyzer.print_sea_recommendation(args.metric)
        elif args.figure in figure_methods:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            figure_methods[args.figure](args.metric, output_path / f'{args.figure}.png')
        else:
            print(f"Unknown figure type: {args.figure}")
            print(f"Available: {list(figure_methods.keys()) + ['sea_recommendation']}")


if __name__ == "__main__":
    main()

