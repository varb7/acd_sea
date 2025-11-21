import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Load the data (Phase 3 with metadata)
# Make sure the CSV is in the working directory or adjust the path accordingly.
df = pd.read_csv('causal_discovery_results2/experiment_results_phase_test_with_meta.csv')

# Aggregate data with means & stds for deeper analysis
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
agg_dict = {m: ['mean', 'std'] for m in metric_cols}
df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
df_agg.columns = [
    '_'.join([c for c in col if c]) if isinstance(col, tuple) else col
    for col in df_agg.columns.values
]


class InteractiveViz:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Performance Comparison")

        self.last_filtered_df = pd.DataFrame()

        # Layout with controls pane + plot pane
        self.paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.controls = ttk.Frame(self.paned, padding=10)
        self.paned.add(self.controls, weight=0)

        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        controls_group = ttk.LabelFrame(self.controls, text="Controls", padding=(10, 8))
        controls_group.grid(row=0, column=0, sticky="new")
        for i in range(2):
            controls_group.columnconfigure(i, weight=1)

        row_idx = 0

        # Metric selection
        ttk.Label(controls_group, text="Select Metric (Y-Axis):").grid(
            row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.metric_var = tk.StringVar()
        metrics = metric_cols
        self.metric_dropdown = ttk.Combobox(
            controls_group, textvariable=self.metric_var, values=metrics, state="readonly"
        )
        self.metric_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.metric_dropdown.current(0)
        row_idx += 1

        # X-axis parameter (used for Line/Bar; Box ignores it and uses Algorithm(+Prior))
        ttk.Label(controls_group, text="Select X-Axis Parameter:").grid(
            row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.x_var = tk.StringVar()
        x_params = [
            'num_samples',
            'num_nodes',
            'root_distribution_type',
            'root_variation_level',
            'root_mean_bias',
            'noise_type',
            'noise_intensity_level',
            'algorithm',
            'use_prior'
        ]
        self.x_dropdown = ttk.Combobox(
            controls_group, textvariable=self.x_var, values=x_params, state="readonly"
        )
        self.x_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.x_dropdown.current(0)
        row_idx += 1

        # Plot type (new: Box)
        ttk.Label(controls_group, text="Select Plot Type:").grid(
            row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.plot_type_var = tk.StringVar()
        plot_types = ['Box', 'Line', 'Bar']  # Box added, and made first/default
        self.plot_type_dropdown = ttk.Combobox(
            controls_group, textvariable=self.plot_type_var, values=plot_types, state="readonly"
        )
        self.plot_type_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.plot_type_dropdown.current(0)
        row_idx += 1

        ttk.Separator(controls_group, orient='horizontal').grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", pady=(6, 6)
        )
        row_idx += 1

        # Configuration-level filters: these define the "configuration" you described.
        self.eq_type_var = self._add_dropdown(
            controls_group, "Equation Type:", sorted(df_agg['equation_type'].dropna().unique()), row_idx
        )
        row_idx += 1
        self.var_type_var = self._add_dropdown(
            controls_group, "Var Type:", sorted(df_agg['var_type_tag'].dropna().unique()), row_idx
        )
        row_idx += 1

        # Prior filter
        ttk.Label(controls_group, text="Use Prior:").grid(
            row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        self.prior_var = tk.StringVar()
        priors = ['All', 'True', 'False']
        self.prior_dropdown = ttk.Combobox(
            controls_group, textvariable=self.prior_var, values=priors, state="readonly"
        )
        self.prior_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.prior_dropdown.current(0)
        row_idx += 1

        # Root and noise properties (config dimensions)
        self.root_dist_var = self._add_dropdown(
            controls_group, "Root Dist. Type:", sorted(df_agg['root_distribution_type'].dropna().unique()), row_idx
        )
        row_idx += 1
        self.root_var_level = self._add_dropdown(
            controls_group, "Root Variation:", sorted(df_agg['root_variation_level'].dropna().unique()), row_idx
        )
        row_idx += 1
        self.root_mean_bias = self._add_dropdown(
            controls_group, "Root Mean Bias:", sorted(df_agg['root_mean_bias'].dropna().unique()), row_idx
        )
        row_idx += 1
        self.noise_type_var = self._add_dropdown(
            controls_group, "Noise Type:", sorted(df_agg['noise_type'].dropna().unique()), row_idx
        )
        row_idx += 1
        self.noise_intensity_var = self._add_dropdown(
            controls_group, "Noise Intensity:", sorted(df_agg['noise_intensity_level'].dropna().unique()), row_idx
        )
        row_idx += 1

        ttk.Separator(controls_group, orient='horizontal').grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", pady=(6, 6)
        )
        row_idx += 1

        # Algorithm filter
        self.algo_var = self._add_dropdown(
            controls_group, "Select Algorithm:", sorted(df_agg['algorithm'].dropna().unique()), row_idx
        )
        row_idx += 1

        # Faceting options
        facet_choices = [
            'None',
            'pattern',
            'root_distribution_type',
            'root_variation_level',
            'root_mean_bias',
            'noise_type',
            'noise_intensity_level',
            'use_prior'
        ]
        self.facet_row_var = self._add_dropdown(
            controls_group, "Facet Rows:", facet_choices[1:], row_idx, include_all=False, extra_values=['None']
        )
        row_idx += 1
        self.facet_col_var = self._add_dropdown(
            controls_group, "Facet Columns:", facet_choices[1:], row_idx, include_all=False, extra_values=['None']
        )
        row_idx += 1

        # Error bars (for Line & Bar; Boxplot inherently shows spread)
        self.show_error_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_group,
            text="Show Error Bars (Line/Bar only)",
            variable=self.show_error_var
        ).grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=(2, 2))
        row_idx += 1

        # Compare prior vs no-prior on the same graph
        self.compare_prior_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls_group,
            text="Compare Prior vs No-Prior (same plot)",
            variable=self.compare_prior_var
        ).grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row_idx += 1

        # Buttons
        button_frame = ttk.Frame(controls_group)
        button_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(button_frame, text="Plot", command=self.plot_graph).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_frame, text="Export Data", command=self.export_filtered_data).grid(
            row=0, column=1, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_frame, text="Save Plot", command=self.save_plot).grid(
            row=0, column=2, sticky="ew"
        )

        # Matplotlib figure & toolbar
        toolbar_container = ttk.Frame(self.plot_frame)
        toolbar_container.grid(row=0, column=0, sticky="ew")
        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.bind("<Configure>", self._on_resize)

    def _add_dropdown(self, parent, label, values, row_idx, include_all=True, extra_values=None):
        ttk.Label(parent, text=label).grid(
            row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w"
        )
        var = tk.StringVar()
        opts = list(values)
        if include_all:
            opts = ['All'] + opts
        if extra_values:
            opts = extra_values + opts
        dropdown = ttk.Combobox(parent, textvariable=var, values=opts, state="readonly")
        dropdown.grid(row=row_idx, column=1, sticky="ew")
        dropdown.current(0)
        return var

    def _on_resize(self, event):
        if event.width <= 1 or event.height <= 1:
            return
        toolbar_height = self.toolbar.winfo_height() if hasattr(self, 'toolbar') else 0
        available_width = max(1, event.width)
        available_height = max(1, event.height - toolbar_height)
        dpi = self.figure.get_dpi()
        self.figure.set_size_inches(
            available_width / dpi, available_height / dpi, forward=True
        )
        self.canvas.draw_idle()

    def _filter_dataframe(self):
        """
        Apply configuration-level filters.

        IMPORTANT: If a filter is set to 'All', that dimension is NOT restricted,
        meaning we aggregate over all its values. This implements your requirement:
        "If these parameters are not selected but the others are then I need the
        average of all datasets which pass the selected configuration."
        """
        df_plot = df_agg.copy()

        if self.eq_type_var.get() != 'All':
            df_plot = df_plot[df_plot['equation_type'] == self.eq_type_var.get()]
        if self.var_type_var.get() != 'All':
            df_plot = df_plot[df_plot['var_type_tag'] == self.var_type_var.get()]
        if self.prior_var.get() != 'All':
            df_plot = df_plot[df_plot['use_prior'] == (self.prior_var.get() == 'True')]
        if self.algo_var.get() != 'All':
            df_plot = df_plot[df_plot['algorithm'] == self.algo_var.get()]
        if self.root_dist_var.get() != 'All':
            df_plot = df_plot[df_plot['root_distribution_type'] == self.root_dist_var.get()]
        if self.root_var_level.get() != 'All':
            df_plot = df_plot[df_plot['root_variation_level'] == self.root_var_level.get()]
        if self.root_mean_bias.get() != 'All':
            df_plot = df_plot[df_plot['root_mean_bias'] == self.root_mean_bias.get()]
        if self.noise_type_var.get() != 'All':
            df_plot = df_plot[df_plot['noise_type'] == self.noise_type_var.get()]
        if self.noise_intensity_var.get() != 'All':
            df_plot = df_plot[df_plot['noise_intensity_level'] == self.noise_intensity_var.get()]

        return df_plot

    def plot_graph(self):
        metric = self.metric_var.get()
        x_param = self.x_var.get()
        plot_type = self.plot_type_var.get()
        facet_row = self.facet_row_var.get()
        facet_col = self.facet_col_var.get()

        # For Line/Bar, user can choose to facet by use_prior
        # For Box, compare_prior is handled INSIDE the plot, not via facets
        if self.compare_prior_var.get() and plot_type != 'Box':
            facet_col = 'use_prior'

        df_plot = self._filter_dataframe()
        self.last_filtered_df = df_plot.copy()

        if df_plot.empty:
            messagebox.showinfo("No Data", "No rows match the selected filters.")
            return

        self.figure.clear()

        row_values = ['All'] if facet_row == 'None' else sorted(
            df_plot[facet_row].dropna().unique().tolist()
        )
        col_values = ['All'] if facet_col == 'None' else sorted(
            df_plot[facet_col].dropna().unique().tolist()
        )
        if facet_col == 'use_prior':
            # Ensure deterministic order of columns for faceting by prior
            col_values = ['False', 'True']

        n_rows = len(row_values)
        n_cols = len(col_values)
        axes = self.figure.subplots(n_rows, n_cols, squeeze=False)

        export_frames = []
        any_visible = False

        for i, row_val in enumerate(row_values):
            for j, col_val in enumerate(col_values):
                ax = axes[i][j]
                subset = df_plot.copy()

                if facet_row != 'None':
                    subset = subset[subset[facet_row] == row_val]
                if facet_col != 'None':
                    # For 'use_prior' faceting, col_val is string 'True'/'False'
                    if facet_col == 'use_prior':
                        subset = subset[subset['use_prior'] == (col_val == 'True')]
                    else:
                        subset = subset[subset[facet_col] == col_val]

                if subset.empty:
                    ax.set_visible(False)
                    continue

                agg_subset = self._plot_subset(ax, subset, x_param, metric, plot_type)
                if agg_subset is None or agg_subset.empty:
                    ax.set_visible(False)
                    continue

                title_parts = []
                if facet_row != 'None':
                    title_parts.append(f"{facet_row}: {row_val}")
                if facet_col != 'None':
                    title_parts.append(f"{facet_col}: {col_val}")
                ax.set_title(" | ".join(title_parts) if title_parts else "All Data")

                any_visible = True

                # Add facet information to exported summary
                agg_subset = agg_subset.assign(
                    facet_row=facet_row if facet_row != 'None' else 'All',
                    facet_row_value=row_val if facet_row != 'None' else 'All',
                    facet_col=facet_col if facet_col != 'None' else 'All',
                    facet_col_value=col_val if facet_col != 'None' else 'All',
                )
                export_frames.append(agg_subset)

        if not any_visible:
            messagebox.showinfo("No Data", "No data to display for the selected facets.")
            return

        self.figure.tight_layout()
        self.canvas.draw()

        if export_frames:
            self.last_filtered_df = pd.concat(export_frames, ignore_index=True)

    def _plot_subset(self, ax, subset, x_param, metric, plot_type):
        """
        Plot a single facet subset.

        For Box:
            - X-axis: Algorithm (optionally split by prior True/False in the same graph)
            - Y-axis: metric_mean (distribution over matching configurations)
            - Uses df_agg rows directly -> no single data-point-only view.
        """
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"

        if plot_type == 'Box':
            # Use the distribution of metric_mean across all configurations
            if metric_mean_col not in subset.columns:
                return None

            # Decide grouping: Algorithm only, or Algorithm + use_prior
            compare_prior = self.compare_prior_var.get()
            if compare_prior:
                group_keys = ['algorithm', 'use_prior']
            else:
                group_keys = ['algorithm']

            # Build boxplot data
            grouped = subset.groupby(group_keys)

            data = []
            labels = []
            for key, group in grouped:
                vals = group[metric_mean_col].dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals)
                if compare_prior:
                    algo, use_prior = key
                    labels.append(f"{algo}\nprior={bool(use_prior)}")
                else:
                    labels.append(str(key))  # key is algorithm

            if not data:
                return None

            bp = ax.boxplot(
                data,
                labels=labels,
                showmeans=True,
                meanline=False
            )

            ax.set_xlabel("Algorithm" + (" + Prior" if compare_prior else ""))
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.3)

            # Build a summary frame (mean of means, std of means, count per group)
            summary = grouped[metric_mean_col].agg(
                mean_of_means='mean',
                std_of_means='std',
                n='count'
            ).reset_index()

            return summary

        # For Line/Bar we keep your original style (aggregated along x_param)
        subset = subset.sort_values(x_param)
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"
        agg_subset = self._aggregate_for_plot(subset, x_param, metric_mean_col, metric_std_col)
        if agg_subset.empty:
            return None

        algorithms = agg_subset['algorithm'].unique()
        show_error = self.show_error_var.get()

        if x_param == 'use_prior':
            agg_subset = agg_subset.assign(use_prior=agg_subset['use_prior'].astype(str))

        if plot_type == 'Line':
            for algo in algorithms:
                df_a = agg_subset[agg_subset['algorithm'] == algo]
                if df_a.empty:
                    continue
                x_vals = df_a[x_param].tolist()
                y_vals = df_a[metric_mean_col].tolist()
                if show_error and metric_std_col in df_a:
                    y_err = df_a[metric_std_col].fillna(0).tolist()
                    ax.errorbar(
                        x_vals, y_vals, yerr=y_err, marker='o',
                        label=algo, capsize=3
                    )
                else:
                    ax.plot(x_vals, y_vals, marker='o', label=algo)
        else:  # Bar
            unique_x = list(dict.fromkeys(agg_subset[x_param].tolist()))
            width = 0.8 / len(algorithms) if len(algorithms) else 0.8
            for idx, algo in enumerate(algorithms):
                df_a = agg_subset[agg_subset['algorithm'] == algo]
                values = []
                errs = []
                for xv in unique_x:
                    row = df_a[df_a[x_param] == xv]
                    if not row.empty:
                        values.append(row[metric_mean_col].iloc[0])
                        errs.append(
                            row[metric_std_col].iloc[0]
                            if show_error and metric_std_col in row
                            else 0
                        )
                    else:
                        values.append(0)
                        errs.append(0)
                positions = [pos + idx * width for pos in range(len(unique_x))]
                ax.bar(
                    positions,
                    values,
                    width=width,
                    label=algo,
                    yerr=errs if show_error else None,
                    capsize=3
                )
            ax.set_xticks(
                [pos + (len(algorithms) - 1) * width / 2 for pos in range(len(unique_x))]
            )
            ax.set_xticklabels(unique_x, rotation=15, ha='right')

        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        if len(algorithms) > 1:
            ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.3)
        return agg_subset

    def _aggregate_for_plot(self, subset, x_param, metric_mean_col, metric_std_col):
        """
        For Line/Bar: aggregate over everything except x_param and algorithm.
        This is where we average over remaining dimensions that are not explicitly
        selected on the x-axis, implementing the 'average over all datasets'
        requirement for those views.
        """
        group_keys = [x_param, 'algorithm']
        if x_param not in subset.columns:
            return pd.DataFrame()
        if metric_std_col not in subset.columns:
            subset = subset.assign(**{metric_std_col: 0.0})
        agg = subset.groupby(group_keys).agg({
            metric_mean_col: 'mean',
            metric_std_col: lambda s: float(
                np.sqrt(np.nanmean(np.square(s)))
            ) if not s.isna().all() else 0.0
        }).reset_index()
        return agg

    def export_filtered_data(self):
        if self.last_filtered_df.empty:
            messagebox.showinfo("No Data", "Run a plot first to cache filtered data.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if not path:
            return
        self.last_filtered_df.to_csv(path, index=False)
        messagebox.showinfo("Export Complete", f"Saved filtered data to {path}")

    def save_plot(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")]
        )
        if not path:
            return
        self.figure.savefig(path, dpi=200)
        messagebox.showinfo("Plot Saved", f"Saved figure to {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveViz(root)
    root.mainloop()
