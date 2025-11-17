import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

# Load the data
df = pd.read_csv('experiment_results.csv')

# Aggregate data: group by key params, mean over seeds
group_cols = ['algorithm', 'use_prior', 'pattern', 'num_nodes', 'num_edges', 'num_samples', 'equation_type', 'var_type_tag']
df_agg = df.groupby(group_cols).agg({
    'shd': 'mean',
    'normalized_shd': 'mean',
    'f1_score': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'execution_time': 'mean'
}).reset_index()

# GUI Class
class InteractiveViz:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Performance Comparison")

        # Paned layout: controls on the left, plot on the right
        self.paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Controls frame
        self.controls = ttk.Frame(self.paned, padding=10)
        self.paned.add(self.controls, weight=0)

        # Plot frame
        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)  # canvas row
        self.plot_frame.columnconfigure(0, weight=1)

        # Controls content (two-column grid inside a labeled frame)
        controls_group = ttk.LabelFrame(self.controls, text="Controls", padding=(10, 8))
        controls_group.grid(row=0, column=0, sticky="new")
        for i in range(2):
            controls_group.columnconfigure(i, weight=1)

        row_idx = 0

        ttk.Label(controls_group, text="Select Metric (Y-Axis):").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.metric_var = tk.StringVar()
        metrics = ['f1_score', 'shd', 'normalized_shd', 'precision', 'recall', 'execution_time']
        self.metric_dropdown = ttk.Combobox(controls_group, textvariable=self.metric_var, values=metrics, state="readonly")
        self.metric_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.metric_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Select X-Axis Parameter:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.x_var = tk.StringVar()
        x_params = ['num_samples', 'num_nodes']
        self.x_dropdown = ttk.Combobox(controls_group, textvariable=self.x_var, values=x_params, state="readonly")
        self.x_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.x_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Select Plot Type:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.plot_type_var = tk.StringVar()
        plot_types = ['Line', 'Bar']
        self.plot_type_dropdown = ttk.Combobox(controls_group, textvariable=self.plot_type_var, values=plot_types, state="readonly")
        self.plot_type_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.plot_type_dropdown.current(0)
        row_idx += 1

        ttk.Separator(controls_group, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        row_idx += 1

        ttk.Label(controls_group, text="Equation Type:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.eq_type_var = tk.StringVar()
        eq_types = ['All'] + sorted(df_agg['equation_type'].unique().tolist())
        self.eq_type_dropdown = ttk.Combobox(controls_group, textvariable=self.eq_type_var, values=eq_types, state="readonly")
        self.eq_type_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.eq_type_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Var Type:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.var_type_var = tk.StringVar()
        var_types = ['All'] + sorted(df_agg['var_type_tag'].unique().tolist())
        self.var_type_dropdown = ttk.Combobox(controls_group, textvariable=self.var_type_var, values=var_types, state="readonly")
        self.var_type_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.var_type_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Use Prior:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.prior_var = tk.StringVar()
        priors = ['All', 'True', 'False']
        self.prior_dropdown = ttk.Combobox(controls_group, textvariable=self.prior_var, values=priors, state="readonly")
        self.prior_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.prior_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Select Algorithm:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.algo_var = tk.StringVar()
        algos = ['All'] + sorted(df_agg['algorithm'].unique().tolist())
        self.algo_dropdown = ttk.Combobox(controls_group, textvariable=self.algo_var, values=algos, state="readonly")
        self.algo_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.algo_dropdown.current(0)
        row_idx += 1

        ttk.Label(controls_group, text="Group By (for single view):").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.group_var = tk.StringVar()
        groups = ['pattern']  # Focusing on pattern for single views
        self.group_dropdown = ttk.Combobox(controls_group, textvariable=self.group_var, values=groups, state="readonly")
        self.group_dropdown.grid(row=row_idx, column=1, sticky="ew")
        self.group_dropdown.current(0)
        self.group_dropdown.bind("<<ComboboxSelected>>", self.update_group_values)
        row_idx += 1

        ttk.Label(controls_group, text="Select Group Value:").grid(row=row_idx, column=0, padx=(0, 8), pady=4, sticky="w")
        self.group_value_var = tk.StringVar()
        self.group_value_dropdown = ttk.Combobox(controls_group, textvariable=self.group_value_var, state="readonly")
        self.group_value_dropdown.grid(row=row_idx, column=1, sticky="ew")
        row_idx += 1

        # Plot Button
        self.plot_button = ttk.Button(controls_group, text="Plot", command=self.plot_graph)
        self.plot_button.grid(row=row_idx, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        row_idx += 1

        # Plot area: toolbar + canvas
        toolbar_container = ttk.Frame(self.plot_frame)
        toolbar_container.grid(row=0, column=0, sticky="ew")
        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container, pack_toolbar=False)
        self.toolbar.update()
        # Place toolbar with ttk for better theming
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")

        # Resize handling: make figure match available area
        self.plot_frame.bind("<Configure>", self._on_resize)

        # Initial update
        self.update_group_values()

    def _on_resize(self, event):
        # Adjust figure size to available space in pixels → inches
        if event.width <= 1 or event.height <= 1:
            return
        # Deduct a small height for the toolbar if present
        toolbar_height = self.toolbar.winfo_height() if hasattr(self, 'toolbar') else 0
        available_width = max(1, event.width)
        available_height = max(1, event.height - toolbar_height)
        dpi = self.figure.get_dpi()
        self.figure.set_size_inches(available_width / dpi, available_height / dpi, forward=True)
        self.canvas.draw_idle()

    def update_group_values(self, event=None):
        group = self.group_var.get()
        if group:
            unique_groups = sorted(df_agg[group].unique().tolist())
            self.group_value_dropdown['values'] = unique_groups
            if unique_groups:
                self.group_value_dropdown.current(0)

    def plot_graph(self):
        metric = self.metric_var.get()
        x_param = self.x_var.get()
        plot_type = self.plot_type_var.get()
        eq_type = self.eq_type_var.get()
        var_type = self.var_type_var.get()
        use_prior = self.prior_var.get()
        algo = self.algo_var.get()
        group_by = self.group_var.get()
        group_value = self.group_value_var.get()

        if not group_value:
            return  # No plot if no group value selected

        # Filter data
        df_plot = df_agg.copy()
        if eq_type != 'All':
            df_plot = df_plot[df_plot['equation_type'] == eq_type]
        if var_type != 'All':
            df_plot = df_plot[df_plot['var_type_tag'] == var_type]
        if use_prior != 'All':
            df_plot = df_plot[df_plot['use_prior'] == (use_prior == 'True')]
        if algo != 'All':
            df_plot = df_plot[df_plot['algorithm'] == algo]

        # Filter to the selected group value
        df_plot = df_plot[df_plot[group_by] == group_value]

        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Sort by x_param
        df_plot = df_plot.sort_values(x_param)

        # Get algorithms to plot
        algorithms = [algo] if algo != 'All' else df_plot['algorithm'].unique()

        if plot_type == 'Line':
            for a in algorithms:
                df_a = df_plot[df_plot['algorithm'] == a]
                ax.plot(df_a[x_param], df_a[metric], marker='o', label=a)
        elif plot_type == 'Bar':
            # For bar, group by x_param and algorithm
            unique_x = sorted(df_plot[x_param].unique())
            width = 0.8 / len(algorithms) if len(algorithms) > 0 else 0.8
            for i, a in enumerate(algorithms):
                df_a = df_plot[df_plot['algorithm'] == a]
                # Align bars to unique_x ordering
                values = []
                for xv in unique_x:
                    row = df_a[df_a[x_param] == xv]
                    values.append(row[metric].iloc[0] if not row.empty else 0)
                positions = [j + i * width for j in range(len(unique_x))]
                ax.bar(positions, values, width=width, label=a)
            ax.set_xticks([j + (len(algorithms) - 1) * width / 2 for j in range(len(unique_x))])
            ax.set_xticklabels(unique_x)

        ax.set_title(f"{group_value} ({group_by})")
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        if len(algorithms) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveViz(root)
    root.mainloop()