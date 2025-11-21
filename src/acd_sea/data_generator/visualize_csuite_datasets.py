"""
Interactive CSuite dataset browser and DAG visualizer.

Launch with:
    python data_generator/visualize_csuite_datasets.py

Features:
- Scan CSuite output directories (default: csuite_grid_datasets_phase3)
- Filter datasets by pattern, phase, equation type, variable type, etc.
- Visualize the stored DAG (adjacency matrix + metadata) using NetworkX
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    # When executed from project root
    from data_generator.experiments.generate_csuite_grid import (
        describe_noise_config,
        describe_root_distribution,
    )
except ImportError:  # pragma: no cover - fallback when run from data_generator dir
    from experiments.generate_csuite_grid import (  # type: ignore
        describe_noise_config,
        describe_root_distribution,
    )


@dataclass
class DatasetEntry:
    name: str
    path: Path
    pattern: str
    phase: str
    num_nodes: Optional[int]
    num_samples: Optional[int]
    equation_type: Optional[str]
    var_type: Optional[str]
    root_dist_type: Optional[str]
    root_dist_tag: Optional[str]
    noise_type: Optional[str]
    noise_tag: Optional[str]
    seed: Optional[int]
    metadata: Dict

    def summary(self) -> str:
        parts = [
            f"{self.pattern or 'unknown'}",
            f"{self.num_nodes or '?'} nodes",
            f"{self.num_samples or '?'} samples",
        ]
        if self.equation_type:
            parts.append(self.equation_type)
        if self.var_type:
            parts.append(self.var_type)
        if self.root_dist_tag:
            parts.append(f"root:{self.root_dist_tag}")
        if self.noise_tag:
            parts.append(f"noise:{self.noise_tag}")
        return " | ".join(parts)


def _safe_load_dict(raw) -> Dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = yaml.safe_load(raw)
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}
    return {}


class CsuiteVisualizer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CSuite Dataset Browser")

        self.dataset_entries: List[DatasetEntry] = []

        self._scan_thread: Optional[threading.Thread] = None

        self._build_layout()
        self._configure_plot_area()

        default_dir = Path("csuite_grid_datasets_phase3")
        self.root_dir_var.set(str(default_dir.resolve()))
        self._schedule_scan()

    # ------------------------------------------------------------------ UI setup
    def _build_layout(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(paned, padding=10)
        controls.columnconfigure(0, weight=1)
        paned.add(controls, weight=0)

        self.plot_frame = ttk.Frame(paned, padding=(8, 0, 8, 8))
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        paned.add(self.plot_frame, weight=1)

        self._build_controls(controls)

    def _build_controls(self, parent: ttk.Frame) -> None:
        row = 0

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, foreground="gray").grid(
            row=row, column=0, sticky="w", pady=(0, 6)
        )
        row += 1

        # Directory selector
        dir_frame = ttk.LabelFrame(parent, text="Dataset Directory", padding=8)
        dir_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        dir_frame.columnconfigure(0, weight=1)
        self.root_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.root_dir_var).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(dir_frame, text="Browse", command=self._browse_dir).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Button(dir_frame, text="Refresh", command=self._schedule_scan).grid(
            row=0, column=2, padx=(6, 0)
        )
        row += 1

        filter_frame = ttk.LabelFrame(parent, text="Filters", padding=8)
        filter_frame.grid(row=row, column=0, sticky="ew")
        for idx in range(2):
            filter_frame.columnconfigure(idx, weight=1)

        # Combobox helper
        self.filter_vars: Dict[str, tk.StringVar] = {}

        def add_filter(label: str, key: str, r: int):
            ttk.Label(filter_frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="All")
            combo = ttk.Combobox(
                filter_frame,
                textvariable=var,
                state="readonly",
                values=("All",),
            )
            combo.grid(row=r, column=1, sticky="ew", pady=2)
            combo.bind("<<ComboboxSelected>>", lambda _evt: self._apply_filters())
            self.filter_vars[key] = var

        add_filter("Pattern", "pattern", 0)
        add_filter("Phase", "phase", 1)
        add_filter("Equation", "equation_type", 2)
        add_filter("Var Type", "var_type", 3)
        add_filter("# Nodes", "num_nodes", 4)
        add_filter("# Samples", "num_samples", 5)
        add_filter("Root Dist", "root_dist_tag", 6)
        add_filter("Noise", "noise_tag", 7)

        ttk.Label(filter_frame, text="Search").grid(row=8, column=0, sticky="w", pady=2)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var)
        search_entry.grid(row=8, column=1, sticky="ew", pady=2)
        search_entry.bind("<KeyRelease>", lambda _evt: self._apply_filters())

        row += 1

        list_frame = ttk.LabelFrame(parent, text="Datasets", padding=8)
        list_frame.grid(row=row, column=0, sticky="nsew", pady=(8, 0))
        parent.rowconfigure(row, weight=1)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        columns = ("summary",)
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=12, selectmode="browse"
        )
        self.tree.heading("summary", text="Dataset Summary")
        self.tree.column("summary", anchor="w", width=420)
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.bind("<<TreeviewSelect>>", lambda _evt: None)

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row + 1, column=0, sticky="ew", pady=(8, 0))
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(btn_frame, text="Visualize DAG", command=self._plot_selected).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(btn_frame, text="Open Folder", command=self._open_selected).grid(
            row=0, column=1, sticky="ew", padx=(0, 4)
        )
        ttk.Button(btn_frame, text="Clear Plot", command=self._clear_plot).grid(
            row=0, column=2, sticky="ew"
        )

    def _configure_plot_area(self) -> None:
        toolbar_container = ttk.Frame(self.plot_frame)
        toolbar_container.grid(row=0, column=0, sticky="ew")

        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, toolbar_container, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------ directory scanning
    def _browse_dir(self) -> None:
        selection = filedialog.askdirectory(initialdir=self.root_dir_var.get() or ".")
        if selection:
            self.root_dir_var.set(selection)
            self._schedule_scan()

    def _schedule_scan(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return
        self._scan_thread = threading.Thread(target=self._scan_datasets, daemon=True)
        self._scan_thread.start()

    def _scan_datasets(self) -> None:
        directory = Path(self.root_dir_var.get()).expanduser().resolve()
        if not directory.exists():
            self._set_status(f"Missing directory: {directory}")
            return

        self._set_status("Scanning datasets...")
        entries: List[DatasetEntry] = []
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("csuite_"):
                continue
            entry = self._build_entry(folder)
            if entry:
                entries.append(entry)

        self.root.after(0, self._on_scan_complete, entries, directory)

    def _on_scan_complete(self, entries: List[DatasetEntry], directory: Path) -> None:
        self.dataset_entries = entries
        self._refresh_filters()
        self._set_status(f"Loaded {len(entries)} datasets from {directory}")

    def _build_entry(self, folder: Path) -> Optional[DatasetEntry]:
        name = folder.name
        meta_path = folder / f"{name}_meta.txt"
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = yaml.safe_load(fh)
            if not isinstance(meta, dict):
                raise ValueError("metadata not dict")
        except Exception as err:
            self._set_status(f"Failed to load {meta_path.name}: {err}")
            return None

        pattern = str(meta.get("pattern", "")).strip()
        num_nodes = meta.get("num_nodes")
        num_samples = meta.get("num_samples")
        equation_type = meta.get("equation_type_override") or meta.get("equation_type")
        var_type = meta.get("var_type_tag")
        seed = meta.get("seed")

        root_dist = {
            "type": meta.get("root_distribution_type"),
            "params": _safe_load_dict(meta.get("root_distribution_params", {})),
        }
        root_tag = describe_root_distribution(root_dist) if root_dist["type"] else None

        noise_cfg = {
            "type": meta.get("noise_type"),
            "params": _safe_load_dict(meta.get("noise_params", {})),
        }
        noise_tag = describe_noise_config(noise_cfg) if noise_cfg["type"] else None

        phase = self._parse_phase(name)

        return DatasetEntry(
            name=name,
            path=folder,
            pattern=pattern,
            phase=phase,
            num_nodes=num_nodes,
            num_samples=num_samples,
            equation_type=equation_type,
            var_type=var_type,
            root_dist_type=root_dist.get("type"),
            root_dist_tag=root_tag,
            noise_type=noise_cfg.get("type"),
            noise_tag=noise_tag,
            seed=seed,
            metadata=meta,
        )

    @staticmethod
    def _parse_phase(name: str) -> str:
        for token in name.split("_"):
            if token.startswith("p") and token[1:].isdigit():
                return token.upper()
        return "UNKNOWN"

    def _refresh_filters(self) -> None:
        def unique_values(attr: str) -> Sequence[str]:
            vals = sorted(
                {
                    str(getattr(entry, attr))
                    for entry in self.dataset_entries
                    if getattr(entry, attr)
                }
            )
            return ["All"] + vals

        for key in self.filter_vars:
            combo_values = unique_values(key)
            # Update combobox values
            children = [
                child
                for child in self.filter_vars.keys()
                if child == key
            ]
            _ = children  # silence lint - values updated below
            combo = self._find_combobox_for_var(self.filter_vars[key])
            if combo:
                combo["values"] = combo_values
            self.filter_vars[key].set("All")

        self._apply_filters()

    def _find_combobox_for_var(self, var: tk.StringVar) -> Optional[ttk.Combobox]:
        # Tkinter doesn't provide reverse lookup; walk widgets to find matching variable
        for widget in self.root.winfo_children():
            combo = self._search_combobox(widget, var)
            if combo:
                return combo
        return None

    def _search_combobox(
        self, widget: tk.Widget, var: tk.StringVar
    ) -> Optional[ttk.Combobox]:
        if isinstance(widget, ttk.Combobox) and widget.cget("textvariable") == str(var):
            return widget
        for child in widget.winfo_children():
            found = self._search_combobox(child, var)
            if found:
                return found
        return None

    def _apply_filters(self) -> None:
        filtered = self.dataset_entries
        for key, var in self.filter_vars.items():
            val = var.get()
            if val and val != "All":
                filtered = [entry for entry in filtered if str(getattr(entry, key)) == val]

        query = self.search_var.get().strip().lower()
        if query:
            filtered = [
                entry
                for entry in filtered
                if query in entry.name.lower() or query in entry.summary().lower()
            ]

        for item in self.tree.get_children():
            self.tree.delete(item)

        for entry in filtered:
            self.tree.insert("", tk.END, iid=entry.name, values=(entry.summary(),))

        if filtered:
            self.tree.selection_set(filtered[0].name)
        self._set_status(f"Displaying {len(filtered)} dataset(s)")

    # ------------------------------------------------------------------ plotting
    def _plot_selected(self) -> None:
        entry = self._get_selected_entry()
        if not entry:
            messagebox.showinfo("Select Dataset", "Please select a dataset to visualize.")
            return

        name = entry.name
        adj_path = entry.path / f"{name}_adj_matrix.csv"
        if not adj_path.exists():
            messagebox.showerror("Missing File", f"Adjacency matrix not found:\n{adj_path}")
            return

        try:
            df_adj = pd.read_csv(adj_path, index_col=0)
            G = nx.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
        except Exception as err:
            messagebox.showerror("Load Error", f"Failed to read adjacency matrix:\n{err}")
            return

        meta = entry.metadata
        root_nodes = set(meta.get("root_nodes") or [])
        station_map = _safe_load_dict(meta.get("station_map", {}))
        station_names = meta.get("station_names") or []
        temporal_order = meta.get("temporal_order") or list(G.nodes)

        pos = self._compute_positions(G, temporal_order, station_map, station_names)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title(
            f"{entry.name}\n"
            f"{entry.pattern} | {entry.phase} | eq:{entry.equation_type or 'n/a'} | "
            f"root:{entry.root_dist_tag or 'n/a'} | noise:{entry.noise_tag or 'n/a'}"
        )

        colors = [
            "#ffcc4d" if node in root_nodes else "#68a0cf" for node in G.nodes()
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=colors,
            node_size=800,
            ax=ax,
            linewidths=1.2,
            edgecolors="#333333",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.5,
            edge_color="#555555",
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="#1a1a1a", ax=ax)

        ax.axis("off")
        ax.margins(0.1)

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#ffcc4d",
                label="Root",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#68a0cf",
                label="Non-root",
                markersize=10,
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper left", frameon=False)

        self.figure.tight_layout()
        self.canvas.draw()
        self._set_status(f"Rendered DAG for {entry.name}")

    def _compute_positions(
        self,
        G: nx.DiGraph,
        temporal_order: Sequence[str],
        station_map: Dict[str, str],
        station_names: Sequence[str],
    ) -> Dict[str, tuple]:
        nodes = list(G.nodes())
        if not nodes:
            return {}

        order_index = {node: idx for idx, node in enumerate(temporal_order) if node in nodes}
        if not order_index:
            order_index = {node: idx for idx, node in enumerate(nodes)}

        if station_map:
            station_levels: Dict[str, int] = {}
            if station_names:
                station_levels = {name: idx for idx, name in enumerate(station_names)}
            else:
                unique_stations = sorted(set(station_map.values()))
                station_levels = {name: idx for idx, name in enumerate(unique_stations)}
            positions = {
                node: (order_index.get(node, 0), -station_levels.get(station_map.get(node, ""), 0))
                for node in nodes
            }
        else:
            positions = {node: (order_index.get(node, 0), 0) for node in nodes}

        # If all y-values are identical, spread slightly
        y_vals = {pos[1] for pos in positions.values()}
        if len(y_vals) <= 1:
            ordered_nodes = sorted(positions, key=lambda n: order_index.get(n, 0))
            positions = {
                node: (order_index.get(node, 0), (idx % 3) - 1)
                for idx, node in enumerate(ordered_nodes)
            }

        return positions

    def _clear_plot(self) -> None:
        self.figure.clear()
        self.canvas.draw()

    def _get_selected_entry(self) -> Optional[DatasetEntry]:
        selection = self.tree.selection()
        if not selection:
            return None
        name = selection[0]
        for entry in self.dataset_entries:
            if entry.name == name:
                return entry
        return None

    def _open_selected(self) -> None:
        entry = self._get_selected_entry()
        if not entry:
            messagebox.showinfo("Select Dataset", "Select a dataset to open.")
            return
        path = entry.path
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as err:
            messagebox.showerror("Open Folder", f"Failed to open folder:\n{err}")

    def _set_status(self, text: str) -> None:
        self.root.after_idle(lambda: self.status_var.set(text))

    def _on_resize(self, event) -> None:
        if event.width <= 1 or event.height <= 1:
            return
        toolbar_height = self.toolbar.winfo_height() if hasattr(self, "toolbar") else 0
        available_width = max(1, event.width)
        available_height = max(1, event.height - toolbar_height)
        dpi = self.figure.get_dpi()
        self.figure.set_size_inches(
            available_width / dpi,
            available_height / dpi,
            forward=True,
        )
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    app = CsuiteVisualizer(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()

