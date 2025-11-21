import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from abc import ABC, abstractmethod
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json

# Load the data
df = pd.read_csv('causal_discovery_results2/experiment_results_phase_test_with_meta.csv')

# Aggregate data
group_cols = [
    'algorithm',
    'use_prior',
    'pattern',
    'num_nodes',
    'num_edges',
    'num_samples',
    'equation_type',
    'var_type_tag',
    'root_distribution_type',
    'root_variation_level',
    'root_mean_bias',
    'noise_type',
    'noise_intensity_level',
]
metric_cols = ['shd', 'normalized_shd', 'f1_score', 'precision', 'recall', 'execution_time']
agg_dict = {m: ['mean', 'std', 'count'] for m in metric_cols}
df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
df_agg.columns = [
    '_'.join([c for c in col if c]) if isinstance(col, tuple) else col
    for col in df_agg.columns.values
]


class AggregationMetadata:
    """Tracks what data went into aggregation"""
    def __init__(self):
        self.n_configs = 0
        self.n_samples = 0
        self.aggregated_over = []
        self.fixed_at = {}
        self.varying_dims = []
    
    def to_string(self):
        parts = [f"Configs: {self.n_configs}"]
        if self.aggregated_over:
            parts.append(f"Averaged over: {', '.join(self.aggregated_over)}")
        if self.fixed_at:
            fixed_str = ', '.join([f"{k}={v}" for k, v in self.fixed_at.items()])
            parts.append(f"Fixed: {fixed_str}")
        return " | ".join(parts)


class PlotStrategy(ABC):
    """Abstract base for different plot types"""
    
    @abstractmethod
    def plot(self, ax, data: pd.DataFrame, metadata: AggregationMetadata, 
             metric: str, x_param: str, show_ci: bool, show_significance: bool):
        pass
    
    @abstractmethod
    def requires_x_axis(self) -> bool:
        pass


class LinePlotStrategy(PlotStrategy):
    """Line plot with confidence intervals"""
    
    def requires_x_axis(self) -> bool:
        return True
    
    def plot(self, ax, data, metadata, metric, x_param, show_ci, show_significance):
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"
        metric_count_col = f"{metric}_count"
        
        algorithms = sorted(data['algorithm'].unique())
        
        for algo in algorithms:
            df_algo = data[data['algorithm'] == algo].sort_values(x_param)
            if df_algo.empty:
                continue
            
            x_vals = df_algo[x_param].tolist()
            y_vals = df_algo[metric_mean_col].tolist()
            
            if show_ci and metric_count_col in df_algo.columns:
                # Calculate 95% CI using standard error
                counts = df_algo[metric_count_col].fillna(1)
                stds = df_algo[metric_std_col].fillna(0)
                se = stds / np.sqrt(counts)
                ci = 1.96 * se  # 95% CI
                
                ax.errorbar(x_vals, y_vals, yerr=ci, marker='o',
                           label=algo, capsize=4, capthick=1.5, linewidth=2,
                           markersize=6, alpha=0.8)
            else:
                ax.plot(x_vals, y_vals, marker='o', label=algo, 
                       linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add sample size annotation
        if metadata.n_configs > 0:
            ax.text(0.02, 0.98, f"n={metadata.n_configs} configs", 
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


class BarPlotStrategy(PlotStrategy):
    """Bar plot with error bars"""
    
    def requires_x_axis(self) -> bool:
        return True
    
    def plot(self, ax, data, metadata, metric, x_param, show_ci, show_significance):
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"
        metric_count_col = f"{metric}_count"
        
        algorithms = sorted(data['algorithm'].unique())
        unique_x = sorted(data[x_param].unique())
        
        width = 0.8 / len(algorithms) if len(algorithms) > 0 else 0.8
        
        for idx, algo in enumerate(algorithms):
            df_algo = data[data['algorithm'] == algo]
            values = []
            errors = []
            
            for xv in unique_x:
                row = df_algo[df_algo[x_param] == xv]
                if not row.empty:
                    values.append(row[metric_mean_col].iloc[0])
                    if show_ci and metric_count_col in row.columns:
                        count = row[metric_count_col].iloc[0]
                        std = row[metric_std_col].iloc[0]
                        se = std / np.sqrt(count) if count > 0 else 0
                        ci = 1.96 * se
                        errors.append(ci)
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            positions = [pos + idx * width for pos in range(len(unique_x))]
            ax.bar(positions, values, width=width, label=algo,
                  yerr=errors if show_ci else None, capsize=3, alpha=0.8)
        
        ax.set_xticks([pos + (len(algorithms) - 1) * width / 2 for pos in range(len(unique_x))])
        ax.set_xticklabels([str(x) for x in unique_x], rotation=15, ha='right')
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        if metadata.n_configs > 0:
            ax.text(0.02, 0.98, f"n={metadata.n_configs} configs", 
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


class BoxPlotStrategy(PlotStrategy):
    """Box plot showing distribution across configurations"""
    
    def requires_x_axis(self) -> bool:
        return False
    
    def plot(self, ax, data, metadata, metric, x_param, show_ci, show_significance):
        metric_mean_col = f"{metric}_mean"
        
        # Group by algorithm (and optionally prior)
        if 'use_prior_comparison' in data.columns and data['use_prior_comparison'].iloc[0]:
            group_keys = ['algorithm', 'use_prior']
        else:
            group_keys = ['algorithm']
        
        grouped = data.groupby(group_keys)
        
        plot_data = []
        labels = []
        positions = []
        pos = 1
        
        for key, group in grouped:
            vals = group[metric_mean_col].dropna().values
            if len(vals) == 0:
                continue
            plot_data.append(vals)
            
            if isinstance(key, tuple):
                algo, use_prior = key
                labels.append(f"{algo}\n(prior={use_prior})")
            else:
                labels.append(str(key))
            positions.append(pos)
            pos += 1
        
        if not plot_data:
            return
        
        bp = ax.boxplot(plot_data, positions=positions, labels=labels,
                       showmeans=True, meanline=False, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.6),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(marker='D', markerfacecolor='green', 
                                     markeredgecolor='green', markersize=6))
        
        # Add significance stars if requested
        if show_significance and len(plot_data) >= 2:
            self._add_significance_stars(ax, plot_data, positions)
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Algorithm' + (' + Prior' if len(group_keys) > 1 else ''), 
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', rotation=0)
        
        # Add distribution info
        ax.text(0.02, 0.98, f"n={metadata.n_configs} configs\nDistribution across configurations", 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _add_significance_stars(self, ax, data_groups, positions):
        """Add significance indicators between groups"""
        # Compare first group with others
        baseline = data_groups[0]
        y_max = max([max(d) for d in data_groups])
        y_range = y_max - min([min(d) for d in data_groups])
        
        for i in range(1, len(data_groups)):
            _, p_value = stats.mannwhitneyu(baseline, data_groups[i], alternative='two-sided')
            
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            else:
                continue
            
            # Draw significance bar
            y_pos = y_max + 0.05 * y_range * (i)
            ax.plot([positions[0], positions[i]], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text((positions[0] + positions[i]) / 2, y_pos, stars, 
                   ha='center', va='bottom', fontsize=10)


class ViolinPlotStrategy(PlotStrategy):
    """Violin plot showing full distribution"""
    
    def requires_x_axis(self) -> bool:
        return False
    
    def plot(self, ax, data, metadata, metric, x_param, show_ci, show_significance):
        metric_mean_col = f"{metric}_mean"
        
        if 'use_prior_comparison' in data.columns and data['use_prior_comparison'].iloc[0]:
            group_keys = ['algorithm', 'use_prior']
        else:
            group_keys = ['algorithm']
        
        grouped = data.groupby(group_keys)
        
        plot_data = []
        labels = []
        positions = []
        pos = 1
        
        for key, group in grouped:
            vals = group[metric_mean_col].dropna().values
            if len(vals) == 0:
                continue
            plot_data.append(vals)
            
            if isinstance(key, tuple):
                algo, use_prior = key
                labels.append(f"{algo}\n(prior={use_prior})")
            else:
                labels.append(str(key))
            positions.append(pos)
            pos += 1
        
        if not plot_data:
            return
        
        parts = ax.violinplot(plot_data, positions=positions, showmeans=True, 
                             showextrema=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.6)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Algorithm' + (' + Prior' if len(group_keys) > 1 else ''), 
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        ax.text(0.02, 0.98, f"n={metadata.n_configs} configs\nFull distribution", 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


class InteractiveViz:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Performance Comparison Tool - Enhanced")
        self.root.geometry("1400x900")
        
        self.last_filtered_df = pd.DataFrame()
        self.last_metadata = AggregationMetadata()
        self.plot_strategies = {
            'Line': LinePlotStrategy(),
            'Bar': BarPlotStrategy(),
            'Box': BoxPlotStrategy(),
            'Violin': ViolinPlotStrategy()
        }
        
        # Preset configurations
        self.presets = {
            "Sample Size Sensitivity": {
                "x_axis": "num_samples",
                "plot_type": "Line",
                "filters": {}
            },
            "Noise Robustness": {
                "x_axis": "noise_intensity_level",
                "plot_type": "Line",
                "facet_col": "noise_type"
            },
            "Algorithm Comparison": {
                "x_axis": "algorithm",
                "plot_type": "Box",
                "filters": {}
            },
            "Prior Impact": {
                "x_axis": "use_prior",
                "plot_type": "Box",
                "compare_prior": True
            }
        }
        
        self._setup_ui()
        self.update_status()
    
    def _setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Status bar at top
        self.status_bar = ttk.Label(main_container, text="Ready", 
                                    relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        self.status_bar.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        # Paned window for controls and plot
        self.paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.paned.grid(row=1, column=0, sticky="nsew")
        main_container.rowconfigure(1, weight=1)
        main_container.columnconfigure(0, weight=1)
        
        # Left panel: Controls
        self.controls = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.controls, weight=0)
        
        # Make controls scrollable
        canvas = tk.Canvas(self.controls, width=350)
        scrollbar = ttk.Scrollbar(self.controls, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Right panel: Plot
        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        
        # Build control sections in scrollable frame
        self._build_controls(scrollable_frame)
        
        # Setup matplotlib
        self._setup_plot_area()
    
    def _build_controls(self, parent):
        row = 0
        
        # ===== PRESET SECTION =====
        preset_frame = ttk.LabelFrame(parent, text="Quick Start", padding=(10, 5))
        preset_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(preset_frame, text="Load Preset:").grid(row=0, column=0, sticky="w", pady=2)
        self.preset_var = tk.StringVar()
        preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                      values=['Custom'] + list(self.presets.keys()),
                                      state="readonly", width=25)
        preset_dropdown.grid(row=0, column=1, sticky="ew", pady=2)
        preset_dropdown.current(0)
        preset_dropdown.bind('<<ComboboxSelected>>', self._load_preset)
        preset_frame.columnconfigure(1, weight=1)
        
        # ===== ANALYSIS MODE =====
        mode_frame = ttk.LabelFrame(parent, text="Analysis Configuration", padding=(10, 5))
        mode_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(mode_frame, text="Metric (Y-Axis):").grid(row=0, column=0, sticky="w", pady=2)
        self.metric_var = tk.StringVar()
        ttk.Combobox(mode_frame, textvariable=self.metric_var, values=metric_cols,
                    state="readonly", width=23).grid(row=0, column=1, sticky="ew", pady=2)
        self.metric_var.set(metric_cols[0])
        
        ttk.Label(mode_frame, text="Plot Type:").grid(row=1, column=0, sticky="w", pady=2)
        self.plot_type_var = tk.StringVar()
        plot_dropdown = ttk.Combobox(mode_frame, textvariable=self.plot_type_var,
                                    values=list(self.plot_strategies.keys()),
                                    state="readonly", width=23)
        plot_dropdown.grid(row=1, column=1, sticky="ew", pady=2)
        plot_dropdown.current(0)
        plot_dropdown.bind('<<ComboboxSelected>>', self._on_plot_type_change)
        
        ttk.Label(mode_frame, text="X-Axis Parameter:").grid(row=2, column=0, sticky="w", pady=2)
        self.x_var = tk.StringVar()
        x_params = ['num_samples', 'num_nodes', 'num_edges', 'root_distribution_type',
                    'root_variation_level', 'root_mean_bias', 'noise_type',
                    'noise_intensity_level', 'algorithm', 'use_prior']
        self.x_dropdown = ttk.Combobox(mode_frame, textvariable=self.x_var,
                                      values=x_params, state="readonly", width=23)
        self.x_dropdown.grid(row=2, column=1, sticky="ew", pady=2)
        self.x_var.set(x_params[0])
        
        mode_frame.columnconfigure(1, weight=1)
        
        # ===== DATA FILTERS =====
        filter_frame = ttk.LabelFrame(parent, text="Data Filters", padding=(10, 5))
        filter_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        filter_row = 0
        
        # Algorithm
        self.algo_var = self._add_filter_dropdown(filter_frame, "Algorithm:",
                                                  sorted(df_agg['algorithm'].unique()),
                                                  filter_row)
        filter_row += 1
        
        # Prior
        self.prior_var = self._add_filter_dropdown(filter_frame, "Use Prior:",
                                                   ['True', 'False'], filter_row)
        filter_row += 1
        
        # Configuration filters
        self.eq_type_var = self._add_filter_dropdown(filter_frame, "Equation Type:",
                                                     sorted(df_agg['equation_type'].unique()),
                                                     filter_row)
        filter_row += 1
        
        self.var_type_var = self._add_filter_dropdown(filter_frame, "Variable Type:",
                                                      sorted(df_agg['var_type_tag'].unique()),
                                                      filter_row)
        filter_row += 1
        
        # Collapsible advanced filters
        self.show_advanced = tk.BooleanVar(value=False)
        advanced_toggle = ttk.Checkbutton(filter_frame, text="Show Advanced Filters",
                                         variable=self.show_advanced,
                                         command=self._toggle_advanced)
        advanced_toggle.grid(row=filter_row, column=0, columnspan=2, sticky="w", pady=5)
        filter_row += 1
        
        self.advanced_frame = ttk.Frame(filter_frame)
        self.advanced_frame.grid(row=filter_row, column=0, columnspan=2, sticky="ew")
        self.advanced_frame.grid_remove()  # Hidden by default
        filter_row += 1
        
        adv_row = 0
        self.root_dist_var = self._add_filter_dropdown(self.advanced_frame, "Root Distribution:",
                                                       sorted(df_agg['root_distribution_type'].unique()),
                                                       adv_row)
        adv_row += 1
        
        self.root_var_level = self._add_filter_dropdown(self.advanced_frame, "Root Variation:",
                                                        sorted(df_agg['root_variation_level'].unique()),
                                                        adv_row)
        adv_row += 1
        
        self.root_mean_bias = self._add_filter_dropdown(self.advanced_frame, "Root Mean Bias:",
                                                        sorted(df_agg['root_mean_bias'].unique()),
                                                        adv_row)
        adv_row += 1
        
        self.noise_type_var = self._add_filter_dropdown(self.advanced_frame, "Noise Type:",
                                                        sorted(df_agg['noise_type'].unique()),
                                                        adv_row)
        adv_row += 1
        
        self.noise_intensity_var = self._add_filter_dropdown(self.advanced_frame, "Noise Intensity:",
                                                             sorted(df_agg['noise_intensity_level'].unique()),
                                                             adv_row)
        
        filter_frame.columnconfigure(1, weight=1)
        
        # ===== VISUALIZATION OPTIONS =====
        viz_frame = ttk.LabelFrame(parent, text="Visualization Options", padding=(10, 5))
        viz_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        viz_row = 0
        
        # Faceting
        facet_choices = ['None', 'pattern', 'root_distribution_type', 'root_variation_level',
                        'root_mean_bias', 'noise_type', 'noise_intensity_level', 'use_prior']
        
        self.facet_row_var = self._add_filter_dropdown(viz_frame, "Facet Rows:",
                                                       facet_choices, viz_row,
                                                       include_all=False)
        viz_row += 1
        
        self.facet_col_var = self._add_filter_dropdown(viz_frame, "Facet Columns:",
                                                       facet_choices, viz_row,
                                                       include_all=False)
        viz_row += 1
        
        # Options
        self.show_ci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show Confidence Intervals (95%)",
                       variable=self.show_ci_var).grid(row=viz_row, column=0, 
                                                       columnspan=2, sticky="w", pady=2)
        viz_row += 1
        
        self.show_sig_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Show Significance Tests (*p<0.05)",
                       variable=self.show_sig_var).grid(row=viz_row, column=0,
                                                        columnspan=2, sticky="w", pady=2)
        viz_row += 1
        
        self.compare_prior_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Compare Prior vs No-Prior",
                       variable=self.compare_prior_var).grid(row=viz_row, column=0,
                                                             columnspan=2, sticky="w", pady=2)
        viz_row += 1
        
        viz_frame.columnconfigure(1, weight=1)
        
        # ===== PUBLICATION SETTINGS =====
        pub_frame = ttk.LabelFrame(parent, text="Publication Settings", padding=(10, 5))
        pub_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        pub_row = 0
        
        ttk.Label(pub_frame, text="Figure Size (inches):").grid(row=pub_row, column=0, 
                                                                sticky="w", pady=2)
        size_frame = ttk.Frame(pub_frame)
        size_frame.grid(row=pub_row, column=1, sticky="ew", pady=2)
        
        self.fig_width_var = tk.StringVar(value="10")
        self.fig_height_var = tk.StringVar(value="6")
        ttk.Entry(size_frame, textvariable=self.fig_width_var, width=5).pack(side="left")
        ttk.Label(size_frame, text=" × ").pack(side="left")
        ttk.Entry(size_frame, textvariable=self.fig_height_var, width=5).pack(side="left")
        pub_row += 1
        
        ttk.Label(pub_frame, text="Font Scale:").grid(row=pub_row, column=0, sticky="w", pady=2)
        self.font_scale_var = tk.StringVar(value="1.0")
        ttk.Entry(pub_frame, textvariable=self.font_scale_var, width=10).grid(
            row=pub_row, column=1, sticky="w", pady=2)
        pub_row += 1
        
        ttk.Label(pub_frame, text="DPI (export):").grid(row=pub_row, column=0, sticky="w", pady=2)
        self.dpi_var = tk.StringVar(value="300")
        ttk.Combobox(pub_frame, textvariable=self.dpi_var, values=['150', '300', '600'],
                    state="readonly", width=8).grid(row=pub_row, column=1, sticky="w", pady=2)
        pub_row += 1
        
        pub_frame.columnconfigure(1, weight=1)
        
        # ===== ACTION BUTTONS =====
        button_frame = ttk.Frame(parent, padding=(10, 5))
        button_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Button(button_frame, text="Generate Plot", 
                  command=self.plot_graph).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Export Data (CSV)", 
                  command=self.export_filtered_data).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Save Plot (High-Res)", 
                  command=self.save_plot).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Export Settings (JSON)",
                  command=self.export_settings).pack(fill="x", pady=2)
        
        # Add filter change callbacks
        for var in [self.algo_var, self.prior_var, self.eq_type_var, self.var_type_var,
                   self.root_dist_var, self.root_var_level, self.root_mean_bias,
                   self.noise_type_var, self.noise_intensity_var]:
            var.trace_add('write', lambda *args: self.update_status())
    
    def _add_filter_dropdown(self, parent, label, values, row, include_all=True):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 5))
        var = tk.StringVar()
        opts = ['All'] + list(values) if include_all else list(values)
        dropdown = ttk.Combobox(parent, textvariable=var, values=opts,
                               state="readonly", width=20)
        dropdown.grid(row=row, column=1, sticky="ew", pady=2)
        dropdown.current(0)
        parent.columnconfigure(1, weight=1)
        return var
    
    def _toggle_advanced(self):
        if self.show_advanced.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
    
    def _on_plot_type_change(self, event=None):
        plot_type = self.plot_type_var.get()
        strategy = self.plot_strategies.get(plot_type)
        
        if strategy and not strategy.requires_x_axis():
            self.x_dropdown.config(state="disabled")
        else:
            self.x_dropdown.config(state="readonly")
        
        self.update_status()
    
    def _load_preset(self, event=None):
        preset_name = self.preset_var.get()
        if preset_name == 'Custom':
            return
        
        preset = self.presets.get(preset_name)
        if not preset:
            return
        
        # Apply preset settings
        if 'x_axis' in preset:
            self.x_var.set(preset['x_axis'])
        if 'plot_type' in preset:
            self.plot_type_var.set(preset['plot_type'])
            self._on_plot_type_change()
        if 'facet_col' in preset:
            self.facet_col_var.set(preset['facet_col'])
        if 'facet_row' in preset:
            self.facet_row_var.set(preset['facet_row'])
        if 'compare_prior' in preset:
            self.compare_prior_var.set(preset['compare_prior'])
        
        # Apply filters if specified
        filters = preset.get('filters', {})
        for key, value in filters.items():
            if hasattr(self, f"{key}_var"):
                getattr(self, f"{key}_var").set(value)
        
        self.update_status()
        messagebox.showinfo("Preset Loaded", f"Applied preset: {preset_name}")
    
    def _setup_plot_area(self):
        toolbar_container = ttk.Frame(self.plot_frame)
        toolbar_container.grid(row=0, column=0, sticky="ew")
        
        self.figure = plt.Figure(figsize=(10, 6), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
    
    def update_status(self):
        """Update status bar with current filter state"""
        try:
            df_filtered = self._apply_filters(df_agg)
            n_configs = len(df_filtered)
            n_algos = df_filtered['algorithm'].nunique() if not df_filtered.empty else 0
            
            if n_configs == 0:
                self.status_bar.config(
                    text="⚠️ WARNING: No data matches current filters",
                    foreground="red"
                )
            elif n_configs < 5:
                self.status_bar.config(
                    text=f"⚠️ Only {n_configs} configurations | {n_algos} algorithms (Low sample size)",
                    foreground="orange"
                )
            else:
                # Calculate aggregation info
                active_filters = sum(1 for var in [self.algo_var, self.prior_var, 
                                                   self.eq_type_var, self.var_type_var,
                                                   self.root_dist_var, self.root_var_level,
                                                   self.root_mean_bias, self.noise_type_var,
                                                   self.noise_intensity_var] 
                                   if var.get() != 'All')
                
                self.status_bar.config(
                    text=f"✓ {n_configs} configurations | {n_algos} algorithms | {active_filters} active filters",
                    foreground="green"
                )
        except Exception as e:
            self.status_bar.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )
    
    def _apply_filters(self, df):
        """Apply all active filters to dataframe"""
        df_filtered = df.copy()
        
        filter_map = {
            'algorithm': self.algo_var,
            'use_prior': self.prior_var,
            'equation_type': self.eq_type_var,
            'var_type_tag': self.var_type_var,
            'root_distribution_type': self.root_dist_var,
            'root_variation_level': self.root_var_level,
            'root_mean_bias': self.root_mean_bias,
            'noise_type': self.noise_type_var,
            'noise_intensity_level': self.noise_intensity_var
        }
        
        for col, var in filter_map.items():
            value = var.get()
            if value != 'All':
                if col == 'use_prior':
                    df_filtered = df_filtered[df_filtered[col] == (value == 'True')]
                else:
                    df_filtered = df_filtered[df_filtered[col] == value]
        
        return df_filtered
    
    def _aggregate_data(self, df, x_param, metric):
        """Aggregate data and track metadata"""
        metadata = AggregationMetadata()
        metadata.n_configs = len(df)
        
        # Determine what we're aggregating over
        active_filters = {}
        for col, var in [('algorithm', self.algo_var), ('use_prior', self.prior_var),
                        ('equation_type', self.eq_type_var), ('var_type_tag', self.var_type_var),
                        ('root_distribution_type', self.root_dist_var),
                        ('root_variation_level', self.root_var_level),
                        ('root_mean_bias', self.root_mean_bias),
                        ('noise_type', self.noise_type_var),
                        ('noise_intensity_level', self.noise_intensity_var)]:
            if var.get() != 'All':
                active_filters[col] = var.get()
        
        metadata.fixed_at = active_filters
        
        # Determine varying dimensions
        plot_type = self.plot_type_var.get()
        strategy = self.plot_strategies.get(plot_type)
        
        if strategy and strategy.requires_x_axis():
            varying = [x_param, 'algorithm']
        else:
            varying = ['algorithm']
            if self.compare_prior_var.get():
                varying.append('use_prior')
        
        metadata.varying_dims = varying
        
        # Determine what we're aggregating over
        all_group_cols = [col for col in group_cols if col in df.columns]
        metadata.aggregated_over = [col for col in all_group_cols 
                                   if col not in varying and col not in active_filters]
        
        # Perform aggregation
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"
        metric_count_col = f"{metric}_count"
        
        if not strategy.requires_x_axis():
            # For box/violin, we don't aggregate - use raw metric_mean values
            agg_df = df.copy()
            if self.compare_prior_var.get():
                agg_df['use_prior_comparison'] = True
        else:
            # For line/bar, aggregate by varying dimensions
            grouped = df.groupby(varying)
            
            # Aggregate with proper error propagation
            agg_result = grouped.agg({
                metric_mean_col: 'mean',
                metric_std_col: lambda x: np.sqrt(np.mean(np.square(x))),
                metric_count_col: 'sum'
            }).reset_index()
            
            agg_df = agg_result
        
        return agg_df, metadata
    
    def plot_graph(self):
        """Main plotting function"""
        metric = self.metric_var.get()
        x_param = self.x_var.get()
        plot_type = self.plot_type_var.get()
        
        # Get plot strategy
        strategy = self.plot_strategies.get(plot_type)
        if not strategy:
            messagebox.showerror("Error", f"Unknown plot type: {plot_type}")
            return
        
        # Apply filters
        df_filtered = self._apply_filters(df_agg)
        
        if df_filtered.empty:
            messagebox.showwarning("No Data", "No data matches the selected filters.")
            return
        
        # Handle faceting
        facet_row = self.facet_row_var.get()
        facet_col = self.facet_col_var.get()
        
        # Override faceting if comparing prior (for non-box plots)
        if self.compare_prior_var.get() and plot_type not in ['Box', 'Violin']:
            facet_col = 'use_prior'
        
        # Determine facet values
        row_values = ['All'] if facet_row == 'None' else sorted(
            df_filtered[facet_row].dropna().unique().tolist()
        )
        col_values = ['All'] if facet_col == 'None' else sorted(
            df_filtered[facet_col].dropna().unique().tolist()
        )
        
        if facet_col == 'use_prior':
            col_values = [False, True]
        
        n_rows = len(row_values)
        n_cols = len(col_values)
        
        # Clear and create subplots
        self.figure.clear()
        
        # Apply publication settings
        try:
            fig_width = float(self.fig_width_var.get())
            fig_height = float(self.fig_height_var.get())
            font_scale = float(self.font_scale_var.get())
        except ValueError:
            fig_width, fig_height, font_scale = 10, 6, 1.0
        
        self.figure.set_size_inches(fig_width, fig_height)
        plt.rcParams.update({'font.size': 10 * font_scale})
        
        axes = self.figure.subplots(n_rows, n_cols, squeeze=False)
        
        show_ci = self.show_ci_var.get()
        show_sig = self.show_sig_var.get()
        
        export_data = []
        any_visible = False
        
        for i, row_val in enumerate(row_values):
            for j, col_val in enumerate(col_values):
                ax = axes[i][j]
                
                # Filter for this facet
                subset = df_filtered.copy()
                
                if facet_row != 'None':
                    subset = subset[subset[facet_row] == row_val]
                
                if facet_col != 'None':
                    if facet_col == 'use_prior':
                        subset = subset[subset['use_prior'] == col_val]
                    else:
                        subset = subset[subset[facet_col] == col_val]
                
                if subset.empty:
                    ax.set_visible(False)
                    continue
                
                # Aggregate data for this facet
                agg_data, metadata = self._aggregate_data(subset, x_param, metric)
                
                if agg_data.empty:
                    ax.set_visible(False)
                    continue
                
                # Plot using strategy
                try:
                    strategy.plot(ax, agg_data, metadata, metric, x_param, 
                                show_ci, show_sig)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", 
                           transform=ax.transAxes, ha='center', va='center')
                    print(f"Plot error: {e}")
                
                # Set facet title
                title_parts = []
                if facet_row != 'None':
                    title_parts.append(f"{facet_row}: {row_val}")
                if facet_col != 'None':
                    title_parts.append(f"{facet_col}: {col_val}")
                
                if title_parts:
                    ax.set_title(" | ".join(title_parts), fontsize=11*font_scale, 
                               fontweight='bold')
                
                any_visible = True
                
                # Store for export
                export_subset = agg_data.copy()
                export_subset['facet_row'] = facet_row if facet_row != 'None' else 'All'
                export_subset['facet_row_value'] = row_val if facet_row != 'None' else 'All'
                export_subset['facet_col'] = facet_col if facet_col != 'None' else 'All'
                export_subset['facet_col_value'] = col_val if facet_col != 'None' else 'All'
                export_subset['metadata'] = metadata.to_string()
                export_data.append(export_subset)
        
        if not any_visible:
            messagebox.showwarning("No Data", "No data to display for selected facets.")
            return
        
        # Add overall title
        suptitle = f"{metric.replace('_', ' ').title()} - {plot_type} Plot"
        self.figure.suptitle(suptitle, fontsize=14*font_scale, fontweight='bold', y=0.98)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Store for export
        if export_data:
            self.last_filtered_df = pd.concat(export_data, ignore_index=True)
            self.last_metadata = metadata
        
        messagebox.showinfo("Success", f"Plot generated with {len(export_data)} facet(s)")
    
    def export_filtered_data(self):
        """Export filtered and aggregated data to CSV"""
        if self.last_filtered_df.empty:
            messagebox.showwarning("No Data", "Generate a plot first to cache data.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
        )
        
        if not path:
            return
        
        try:
            if path.endswith('.xlsx'):
                # Export to Excel with multiple sheets
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    self.last_filtered_df.to_excel(writer, sheet_name='Aggregated Data', 
                                                   index=False)
                    
                    # Add metadata sheet
                    metadata_df = pd.DataFrame({
                        'Setting': ['Metric', 'X-Axis', 'Plot Type', 'Configurations', 
                                   'Aggregated Over', 'Fixed Filters'],
                        'Value': [
                            self.metric_var.get(),
                            self.x_var.get(),
                            self.plot_type_var.get(),
                            str(self.last_metadata.n_configs),
                            ', '.join(self.last_metadata.aggregated_over),
                            str(self.last_metadata.fixed_at)
                        ]
                    })
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            else:
                self.last_filtered_df.to_csv(path, index=False)
            
            messagebox.showinfo("Success", f"Data exported to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def save_plot(self):
        """Save current plot to file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG (High-Res)", "*.png"), ("PDF (Vector)", "*.pdf"),
                      ("SVG (Vector)", "*.svg")]
        )
        
        if not path:
            return
        
        try:
            dpi = int(self.dpi_var.get())
        except ValueError:
            dpi = 300
        
        try:
            self.figure.savefig(path, dpi=dpi, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            messagebox.showinfo("Success", f"Plot saved to {path} (DPI={dpi})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def export_settings(self):
        """Export current settings to JSON for reproducibility"""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if not path:
            return
        
        settings = {
            'analysis': {
                'metric': self.metric_var.get(),
                'x_axis': self.x_var.get(),
                'plot_type': self.plot_type_var.get()
            },
            'filters': {
                'algorithm': self.algo_var.get(),
                'use_prior': self.prior_var.get(),
                'equation_type': self.eq_type_var.get(),
                'var_type': self.var_type_var.get(),
                'root_distribution': self.root_dist_var.get(),
                'root_variation': self.root_var_level.get(),
                'root_mean_bias': self.root_mean_bias.get(),
                'noise_type': self.noise_type_var.get(),
                'noise_intensity': self.noise_intensity_var.get()
            },
            'visualization': {
                'facet_row': self.facet_row_var.get(),
                'facet_col': self.facet_col_var.get(),
                'show_ci': self.show_ci_var.get(),
                'show_significance': self.show_sig_var.get(),
                'compare_prior': self.compare_prior_var.get()
            },
            'publication': {
                'fig_width': self.fig_width_var.get(),
                'fig_height': self.fig_height_var.get(),
                'font_scale': self.font_scale_var.get(),
                'dpi': self.dpi_var.get()
            },
            'metadata': {
                'n_configs': self.last_metadata.n_configs,
                'aggregated_over': self.last_metadata.aggregated_over,
                'fixed_at': self.last_metadata.fixed_at
            }
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", f"Settings exported to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export settings: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveViz(root)
    root.mainloop()