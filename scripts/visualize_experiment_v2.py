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
try:
    df = pd.read_csv('causal_discovery_results2/experiment_results.csv')
except FileNotFoundError:
    # Fallback for development/testing if file not found
    print("Warning: Results file not found. Creating dummy data.")
    df = pd.DataFrame() 

# Global configuration
group_cols = [
    'algorithm', 'use_prior', 'pattern', 'num_nodes', 'num_edges', 'num_samples',
    'equation_type', 'var_type_tag', 'root_distribution_type', 'root_variation_level',
    'root_mean_bias', 'noise_type', 'noise_intensity_level', 'edge_density'
]

metric_cols = ['shd', 'normalized_shd', 'f1_score', 'precision', 'recall', 'execution_time']

class AggregationMetadata:
    def __init__(self):
        self.n_configs = 0
        self.aggregated_over = []
        self.fixed_at = {}
        self.varying_dims = []

class PlotStrategy(ABC):
    @abstractmethod
    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        pass

    @abstractmethod
    def requires_x_axis(self):
        pass

    @abstractmethod
    def requires_y_axis(self):
        pass

class LinePlotStrategy(PlotStrategy):
    def requires_x_axis(self): return True
    def requires_y_axis(self): return False

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        
        algorithms = data['algorithm'].unique()
        
        for algo in algorithms:
            subset = data[data['algorithm'] == algo].sort_values(x_param)
            if subset.empty: continue
            
            x_vals = subset[x_param]
            y_vals = subset[metric_mean]
            
            label = algo
            if 'use_prior' in subset.columns and len(subset['use_prior'].unique()) > 1:
                 # Handle prior distinction if needed
                 pass

            if style == 'Thesis':
                ax.plot(x_vals, y_vals, marker='o', label=label, linewidth=2, markersize=6)
                # Thesis style typically avoids clutter, maybe no error bars unless requested?
                # For now, let's keep it simple.
            else:
                if metric_std in subset.columns:
                    ax.errorbar(x_vals, y_vals, yerr=subset[metric_std], marker='o', label=label, capsize=3)
                else:
                    ax.plot(x_vals, y_vals, marker='o', label=label)
        
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

class BarPlotStrategy(PlotStrategy):
    def requires_x_axis(self): return True
    def requires_y_axis(self): return False

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        
        algorithms = data['algorithm'].unique()
        unique_x = sorted(data[x_param].unique())
        
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            subset = data[data['algorithm'] == algo]
            means = []
            stds = []
            for x in unique_x:
                row = subset[subset[x_param] == x]
                if not row.empty:
                    means.append(row[metric_mean].iloc[0])
                    stds.append(row[metric_std].iloc[0] if metric_std in row.columns else 0)
                else:
                    means.append(0)
                    stds.append(0)
            
            x_pos = [x + i * width for x in range(len(unique_x))]
            ax.bar(x_pos, means, width, label=algo, yerr=stds, capsize=3)
            
        ax.set_xticks([r + width * (len(algorithms) - 1) / 2 for r in range(len(unique_x))])
        ax.set_xticklabels(unique_x)
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()

class BoxPlotStrategy(PlotStrategy):
    def requires_x_axis(self): return False
    def requires_y_axis(self): return False

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        # For box plot, we need raw data or distribution data. 
        # Since we are working with aggregated data (mostly), this is tricky.
        # But wait, _aggregate_data returns 'df' which might be raw if not aggregated?
        # No, _aggregate_data aggregates.
        # However, for BoxPlot, we want to show distribution over the *aggregated_over* dimensions.
        # In phase1, BoxPlot used 'metric_mean' of the filtered subset (which represents means of different configs).
        
        metric_mean = f"{metric}_mean"
        
        data_to_plot = []
        labels = []
        
        algorithms = data['algorithm'].unique()
        for algo in algorithms:
            subset = data[data['algorithm'] == algo]
            vals = subset[metric_mean].dropna()
            if not vals.empty:
                data_to_plot.append(vals)
                labels.append(algo)
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels, showmeans=True)
            
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

class ViolinPlotStrategy(PlotStrategy):
    def requires_x_axis(self): return False
    def requires_y_axis(self): return False

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        metric_mean = f"{metric}_mean"
        
        data_to_plot = []
        labels = []
        
        algorithms = data['algorithm'].unique()
        for algo in algorithms:
            subset = data[data['algorithm'] == algo]
            vals = subset[metric_mean].dropna()
            if not vals.empty:
                data_to_plot.append(vals)
                labels.append(algo)
        
        if data_to_plot:
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            # Customizing violin plot is harder with labels, need to set xticks manually
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

class HeatmapPlotStrategy(PlotStrategy):
    def requires_x_axis(self): return True
    def requires_y_axis(self): return True

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        # Pivot data: x_param vs y_param, value = metric_mean
        # We need to aggregate over algorithm? Or facet by algorithm?
        # Usually heatmap is for one algorithm or average of all.
        # If multiple algorithms are present, we might need to average them or warn.
        # For now, let's average over algorithms if not filtered.
        
        metric_mean = f"{metric}_mean"
        
        pivot = data.pivot_table(index=y_param, columns=x_param, values=metric_mean, aggfunc='mean')
        
        im = ax.imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_title(f"{metric} Heatmap")

class ScatterPlotStrategy(PlotStrategy):
    def requires_x_axis(self): return True
    def requires_y_axis(self): return True

    def plot(self, ax, data, metadata, metric, x_param, y_param, style):
        # Scatter plot: X vs Y.
        # X and Y can be parameters or metrics.
        
        x_col = f"{x_param}_mean" if x_param in metric_cols else x_param
        y_col = f"{y_param}_mean" if y_param in metric_cols else y_param
        
        algorithms = data['algorithm'].unique()
        
        for algo in algorithms:
            subset = data[data['algorithm'] == algo]
            ax.scatter(subset[x_col], subset[y_col], label=algo, alpha=0.7)
            
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)


class InteractiveViz:
    def __init__(self, root):
        self.root = root
        self.root.title("Causal Discovery Experiment Visualizer v2")
        
        self.plot_strategies = {
            'Line': LinePlotStrategy(),
            'Bar': BarPlotStrategy(),
            'Box': BoxPlotStrategy(),
            'Violin': ViolinPlotStrategy(),
            'Heatmap': HeatmapPlotStrategy(),
            'Scatter': ScatterPlotStrategy()
        }
        
        self.last_filtered_df = pd.DataFrame()
        self.last_metadata = None
        
        self._build_controls()
        
        # Initialize figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_controls(self):
        # Main layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar for controls
        self.controls_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(self.controls_frame, weight=0)
        
        # Plot area
        self.plot_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(self.plot_frame, weight=1)
        
        # --- Control Groups ---
        
        # 1. Plot Settings
        plot_group = ttk.LabelFrame(self.controls_frame, text="Plot Settings", padding=5)
        plot_group.pack(fill=tk.X, pady=5)
        
        self.plot_type_var = tk.StringVar(value='Line')
        ttk.Label(plot_group, text="Plot Type:").grid(row=0, column=0, sticky='w')
        type_cb = ttk.Combobox(plot_group, textvariable=self.plot_type_var, 
                               values=list(self.plot_strategies.keys()), state='readonly')
        type_cb.grid(row=0, column=1, sticky='ew')
        type_cb.bind('<<ComboboxSelected>>', self._on_plot_type_change)
        
        self.style_var = tk.StringVar(value='Default')
        ttk.Label(plot_group, text="Style:").grid(row=1, column=0, sticky='w')
        ttk.Combobox(plot_group, textvariable=self.style_var, 
                     values=['Default', 'Thesis'], state='readonly').grid(row=1, column=1, sticky='ew')
        
        # 2. Analysis Axes
        axis_group = ttk.LabelFrame(self.controls_frame, text="Analysis Axes", padding=5)
        axis_group.pack(fill=tk.X, pady=5)
        
        # Metric (Primary Y)
        self.metric_var = tk.StringVar(value='shd')
        ttk.Label(axis_group, text="Metric (Primary):").grid(row=0, column=0, sticky='w')
        ttk.Combobox(axis_group, textvariable=self.metric_var, 
                     values=metric_cols, state='readonly').grid(row=0, column=1, sticky='ew')
        
        # X-Axis
        self.x_var = tk.StringVar(value='num_samples')
        self.x_label = ttk.Label(axis_group, text="X-Axis:")
        self.x_label.grid(row=1, column=0, sticky='w')
        
        x_params = ['num_samples', 'num_nodes', 'num_edges', 'edge_density', 
                    'root_distribution_type', 'root_variation_level', 'root_mean_bias', 
                    'noise_type', 'noise_intensity_level', 'algorithm', 'use_prior'] + metric_cols
        
        self.x_dropdown = ttk.Combobox(axis_group, textvariable=self.x_var, 
                                       values=x_params, state='readonly')
        self.x_dropdown.grid(row=1, column=1, sticky='ew')
        
        # Y-Axis (Secondary/Scatter)
        self.y_var = tk.StringVar(value='recall')
        self.y_label = ttk.Label(axis_group, text="Y-Axis (Sec):")
        self.y_label.grid(row=2, column=0, sticky='w')
        
        y_params = x_params # Same params available
        self.y_dropdown = ttk.Combobox(axis_group, textvariable=self.y_var, 
                                       values=y_params, state='readonly')
        self.y_dropdown.grid(row=2, column=1, sticky='ew')
        
        # 3. Filters
        filter_group = ttk.LabelFrame(self.controls_frame, text="Filters", padding=5)
        filter_group.pack(fill=tk.X, pady=5)
        
        self.algo_var = self._add_filter_dropdown(filter_group, "Algorithm:", df['algorithm'].unique() if not df.empty else [], 0)
        self.prior_var = self._add_filter_dropdown(filter_group, "Use Prior:", ['True', 'False'], 1)
        self.eq_type_var = self._add_filter_dropdown(filter_group, "Eq Type:", df['equation_type'].unique() if not df.empty else [], 2)
        self.var_type_var = self._add_filter_dropdown(filter_group, "Var Type:", df['var_type_tag'].unique() if not df.empty else [], 3)
        self.root_dist_var = self._add_filter_dropdown(filter_group, "Root Dist:", df['root_distribution_type'].unique() if not df.empty else [], 4)
        self.root_var_level = self._add_filter_dropdown(filter_group, "Root Var:", df['root_variation_level'].unique() if not df.empty else [], 5)
        self.root_mean_bias = self._add_filter_dropdown(filter_group, "Root Bias:", df['root_mean_bias'].unique() if not df.empty else [], 6)
        self.noise_type_var = self._add_filter_dropdown(filter_group, "Noise Type:", df['noise_type'].unique() if not df.empty else [], 7)
        self.noise_intensity_var = self._add_filter_dropdown(filter_group, "Noise Int:", df['noise_intensity_level'].unique() if not df.empty else [], 8)
        
        # 4. Faceting
        facet_group = ttk.LabelFrame(self.controls_frame, text="Faceting", padding=5)
        facet_group.pack(fill=tk.X, pady=5)
        
        facet_opts = ['None', 'pattern', 'root_distribution_type', 'root_variation_level', 
                      'root_mean_bias', 'noise_type', 'noise_intensity_level', 'use_prior', 'edge_density']
        
        self.facet_row_var = tk.StringVar(value='None')
        ttk.Label(facet_group, text="Row:").grid(row=0, column=0, sticky='w')
        ttk.Combobox(facet_group, textvariable=self.facet_row_var, values=facet_opts, state='readonly').grid(row=0, column=1, sticky='ew')
        
        self.facet_col_var = tk.StringVar(value='None')
        ttk.Label(facet_group, text="Col:").grid(row=1, column=0, sticky='w')
        ttk.Combobox(facet_group, textvariable=self.facet_col_var, values=facet_opts, state='readonly').grid(row=1, column=1, sticky='ew')
        
        # 5. Options
        opt_group = ttk.LabelFrame(self.controls_frame, text="Options", padding=5)
        opt_group.pack(fill=tk.X, pady=5)
        
        self.compare_prior_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_group, text="Compare Prior", variable=self.compare_prior_var).pack(anchor='w')
        
        self.show_ci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_group, text="Show CI/Error", variable=self.show_ci_var).pack(anchor='w')
        
        self.show_sig_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_group, text="Show Significance", variable=self.show_sig_var).pack(anchor='w')
        
        # 6. Actions
        action_group = ttk.Frame(self.controls_frame, padding=5)
        action_group.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_group, text="Generate Plot", command=self.plot_graph).pack(fill=tk.X, pady=2)
        ttk.Button(action_group, text="Export Data", command=self.export_filtered_data).pack(fill=tk.X, pady=2)
        ttk.Button(action_group, text="Save Plot", command=self.save_plot).pack(fill=tk.X, pady=2)
        ttk.Button(action_group, text="Export Settings", command=self.export_settings).pack(fill=tk.X, pady=2)
        
        # Initial State
        self._on_plot_type_change(None)

    def _add_filter_dropdown(self, parent, label, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
        var = tk.StringVar(value='All')
        vals = ['All'] + sorted([str(v) for v in values if pd.notna(v)])
        ttk.Combobox(parent, textvariable=var, values=vals, state='readonly').grid(row=row, column=1, sticky='ew')
        return var

    def _on_plot_type_change(self, event):
        plot_type = self.plot_type_var.get()
        strategy = self.plot_strategies.get(plot_type)
        
        if strategy:
            if strategy.requires_x_axis():
                self.x_dropdown.state(['!disabled'])
                self.x_label.state(['!disabled'])
            else:
                self.x_dropdown.state(['disabled'])
                self.x_label.state(['disabled'])
                
            if strategy.requires_y_axis():
                self.y_dropdown.state(['!disabled'])
                self.y_label.state(['!disabled'])
            else:
                self.y_dropdown.state(['disabled'])
                self.y_label.state(['disabled'])

    def plot_graph(self):
        if df.empty:
            messagebox.showwarning("No Data", "No data loaded.")
            return

        plot_type = self.plot_type_var.get()
        strategy = self.plot_strategies.get(plot_type)
        
        metric = self.metric_var.get()
        x_param = self.x_var.get()
        y_param = self.y_var.get()
        style = self.style_var.get()
        
        # Aggregate data
        agg_data, metadata = self._aggregate_data(df, x_param, metric)
        
        if agg_data.empty:
            messagebox.showwarning("No Data", "No data matches filters.")
            return
            
        self.last_filtered_df = agg_data
        self.last_metadata = metadata
        
        # Clear figure
        self.figure.clear()
        
        # Handle Faceting
        facet_row = self.facet_row_var.get()
        facet_col = self.facet_col_var.get()
        
        if facet_row == 'None' and facet_col == 'None':
            ax = self.figure.add_subplot(111)
            strategy.plot(ax, agg_data, metadata, metric, x_param, y_param, style)
            ax.set_title(f"{metric} by {x_param}")
        else:
            # Faceting logic
            row_vals = ['All'] if facet_row == 'None' else sorted(agg_data[facet_row].unique())
            col_vals = ['All'] if facet_col == 'None' else sorted(agg_data[facet_col].unique())
            
            axes = self.figure.subplots(len(row_vals), len(col_vals), squeeze=False)
            
            for i, r_val in enumerate(row_vals):
                for j, c_val in enumerate(col_vals):
                    ax = axes[i][j]
                    subset = agg_data.copy()
                    if facet_row != 'None': subset = subset[subset[facet_row] == r_val]
                    if facet_col != 'None': subset = subset[subset[facet_col] == c_val]
                    
                    if not subset.empty:
                        strategy.plot(ax, subset, metadata, metric, x_param, y_param, style)
                        ax.set_title(f"{r_val} | {c_val}")
                    else:
                        ax.axis('off')
                        
        self.figure.tight_layout()
        self.canvas.draw()

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
        
        # Filter data
        filtered_df = df.copy()
        for col, val in active_filters.items():
            if col == 'use_prior':
                filtered_df = filtered_df[filtered_df[col] == (val == 'True')]
            else:
                filtered_df = filtered_df[filtered_df[col] == val]
                
        # Determine varying dimensions
        plot_type = self.plot_type_var.get()
        strategy = self.plot_strategies.get(plot_type)
        
        varying = ['algorithm']
        if self.compare_prior_var.get():
            varying.append('use_prior')
            
        y_param = self.y_var.get()
        
        if strategy:
            if strategy.requires_x_axis():
                if x_param in df.columns and x_param not in metric_cols:
                    varying.append(x_param)
            
            if strategy.requires_y_axis():
                 if y_param in df.columns and y_param not in metric_cols:
                    varying.append(y_param)
        
        # Add faceting cols to varying so we don't aggregate them away
        facet_row = self.facet_row_var.get()
        facet_col = self.facet_col_var.get()
        if facet_row != 'None': varying.append(facet_row)
        if facet_col != 'None': varying.append(facet_col)
        
        varying = list(set(varying)) # Unique
        metadata.varying_dims = varying
        
        # Determine what we're aggregating over
        all_group_cols = [col for col in group_cols if col in df.columns]
        metadata.aggregated_over = [col for col in all_group_cols 
                                   if col not in varying and col not in active_filters]
        
        # Perform aggregation
        if not strategy.requires_x_axis() and plot_type != 'Heatmap' and plot_type != 'Scatter':
            # For box/violin, we might want raw data? 
            # But we usually aggregate over seeds/configs to get distribution of means.
            # Let's aggregate by varying dims + 'dataset_name' or similar to keep distribution?
            # Actually, if we want distribution over configurations, we should NOT aggregate everything.
            # But for simplicity, let's just return the filtered_df if plot_type is Box/Violin
            # and let the strategy handle it?
            # Phase 1 logic: BoxPlot used 'metric_mean' from df_agg (which was already aggregated by config).
            # Here 'df' is raw-ish (phase_test_with_meta).
            # Let's aggregate by (varying + unique_config_id)
            pass
            
        # General aggregation
        # Group by varying dimensions
        grouped = filtered_df.groupby(varying)
        
        metrics_to_agg = {metric}
        if x_param in metric_cols: metrics_to_agg.add(x_param)
        if y_param in metric_cols: metrics_to_agg.add(y_param)
            
        agg_dict = {}
        for m in metrics_to_agg:
            if m in df.columns:
                agg_dict[f"{m}_mean"] = (m, 'mean')
                agg_dict[f"{m}_std"] = (m, 'std')
                
        # We need to construct the aggregation dictionary correctly for pandas
        # Named aggregation is cleaner:
        agg_kwargs = {}
        for m in metrics_to_agg:
            if m in df.columns:
                agg_kwargs[f"{m}_mean"] = pd.NamedAgg(column=m, aggfunc='mean')
                agg_kwargs[f"{m}_std"] = pd.NamedAgg(column=m, aggfunc='std')
                
        if not agg_kwargs:
            return filtered_df, metadata # Fallback

        agg_df = grouped.agg(**agg_kwargs).reset_index()
        
        return agg_df, metadata

    def export_filtered_data(self):
        if self.last_filtered_df.empty:
            messagebox.showwarning("No Data", "Generate a plot first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.last_filtered_df.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Saved to {path}")

    def save_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {path}")

    def export_settings(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            settings = {
                'plot_type': self.plot_type_var.get(),
                'metric': self.metric_var.get(),
                'x_axis': self.x_var.get(),
                'filters': {k: v for k, v in self.last_metadata.fixed_at.items()} if self.last_metadata else {}
            }
            with open(path, 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", f"Saved settings to {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveViz(root)
    root.mainloop()