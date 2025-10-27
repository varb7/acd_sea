import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import networkx as nx
from pyvis.network import Network
from IPython import get_ipython
import os
import subprocess
import pkg_resources
import pickle
import yaml
from numpy.random import default_rng

try:
    from scipy.stats import truncnorm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. truncated_normal distribution will not work.")

from causalAssembly.models_dag import ProductionLineGraph
import numpy as np  # SymPy / FCM imports come later if you need them


def softmax_fn(logits):
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)


def generate_nominal_category(X, num_classes=3, seed=42, input_names=None):
    """
    Generate categorical variables from continuous inputs.
    
    Args:
        X: Single input array OR dict of multiple inputs
        num_classes: Number of categories to generate
        seed: Random seed
        input_names: Names of input variables (for multi-input case)
    
    Returns:
        y: Categorical values
        logits: Generated logits
        probs: Probability distributions
    """
    rng = np.random.default_rng(seed)
    
    # Handle both single input and multi-input cases
    if isinstance(X, dict):
        # Multi-input case
        input_names = list(X.keys())
        n = len(X[input_names[0]])
        num_inputs = len(input_names)
        
        # Stack all inputs into a matrix
        X_matrix = np.column_stack([X[name] for name in input_names])
    else:
        # Single input case (backward compatibility)
        X = X.reshape(-1, 1)
        n = X.shape[0]
        num_inputs = 1
        X_matrix = X
        input_names = ['input'] if input_names is None else input_names

    logits = np.zeros((n, num_classes))
    func_types = rng.choice(['linear', 'poly', 'sin', 'exp', 'log', 'sqrt', 'interaction'], size=num_classes)

    for k in range(num_classes):
        # Create a new RNG for each class to ensure different random values
        class_rng = np.random.default_rng(seed + k)
        if func_types[k] == 'linear':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                b = class_rng.uniform(-2, 2)
                logits[:, k] = a * X_matrix[:, 0] + b
            else:
                # Multi-input linear: weighted sum (scaled for reasonable logits)
                weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                bias = class_rng.uniform(-2, 2)
                logits[:, k] = np.sum(X_matrix * weights, axis=1) + bias
                
        elif func_types[k] == 'poly':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                b = class_rng.uniform(-2, 2)
                c = class_rng.uniform(-1, 1)
                logits[:, k] = a * X_matrix[:, 0] ** 2 + b * X_matrix[:, 0] + c
            else:
                # Multi-input polynomial: quadratic terms + interactions (scaled)
                weights = class_rng.uniform(-0.0001, 0.0001, num_inputs)  # Much smaller weights
                interaction_weight = class_rng.uniform(-0.0001, 0.0001)  # Much smaller interaction
                bias = class_rng.uniform(-1, 1)
                
                # Quadratic terms
                quad_terms = np.sum(X_matrix ** 2 * weights, axis=1)
                # Interaction terms (pairwise products)
                interaction_terms = 0
                for i in range(num_inputs):
                    for j in range(i+1, num_inputs):
                        interaction_terms += X_matrix[:, i] * X_matrix[:, j]
                
                logits[:, k] = quad_terms + interaction_weight * interaction_terms + bias
                
        elif func_types[k] == 'sin':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                b = class_rng.uniform(-2, 2)
                logits[:, k] = np.sin(a * X_matrix[:, 0] + b)
            else:
                # Multi-input sin: weighted combination (scaled)
                weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                bias = class_rng.uniform(-2, 2)
                combined_input = np.sum(X_matrix * weights, axis=1) + bias
                logits[:, k] = np.sin(combined_input)
                
        elif func_types[k] == 'exp':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                b = class_rng.uniform(-2, 2)
                logits[:, k] = np.exp(np.clip(a * X_matrix[:, 0] + b, -10, 10))
            else:
                # Multi-input exp: weighted combination (scaled)
                weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                bias = class_rng.uniform(-2, 2)
                combined_input = np.sum(X_matrix * weights, axis=1) + bias
                logits[:, k] = np.exp(np.clip(combined_input, -10, 10))
                
        elif func_types[k] == 'log':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                logits[:, k] = np.log(np.abs(a * X_matrix[:, 0]) + 1)
            else:
                # Multi-input log: weighted combination (scaled)
                weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                combined_input = np.abs(np.sum(X_matrix * weights, axis=1)) + 1
                logits[:, k] = np.log(combined_input)
                
        elif func_types[k] == 'sqrt':
            if num_inputs == 1:
                a = class_rng.uniform(-3, 3)
                logits[:, k] = np.sqrt(np.abs(a * X_matrix[:, 0]) + 1)
            else:
                # Multi-input sqrt: weighted combination (scaled)
                weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                combined_input = np.abs(np.sum(X_matrix * weights, axis=1)) + 1
                logits[:, k] = np.sqrt(combined_input)
                
        elif func_types[k] == 'interaction':
            # Special case for multi-input: create interaction effects
            if num_inputs == 1:
                # Fallback to linear for single input
                a = class_rng.uniform(-3, 3)
                b = class_rng.uniform(-2, 2)
                logits[:, k] = a * X_matrix[:, 0] + b
            else:
                # Create meaningful interactions (scaled)
                main_weights = class_rng.uniform(-0.01, 0.01, num_inputs)  # Much smaller weights
                interaction_weight = class_rng.uniform(-0.0001, 0.0001)  # Much smaller interaction
                bias = class_rng.uniform(-1, 1)
                
                # Main effects
                main_effects = np.sum(X_matrix * main_weights, axis=1)
                
                # Interaction effects (product of all inputs)
                interaction_effect = np.prod(X_matrix, axis=1) * interaction_weight
                
                logits[:, k] = main_effects + interaction_effect + bias

    probs = softmax_fn(logits)
    y = np.array([rng.choice(num_classes, p=probs[i]) for i in range(n)])
    return y, probs, func_types


# If you have this utility function from your original code:
# from scdg.utils import get_generation_type

class CausalDataGenerator:
    """
    CausalDataGenerator: A class to generate synthetic causal datasets from a directed acyclic graph (DAG).

    This class allows you to:
    - Generate random DAGs with a specified number of nodes, root nodes, and edges.
    - Assign distributions to root nodes (either automatically or manually).
    - Assign equations (linear/non-linear/random) to non-root nodes.
    - Add noise to non-root nodes.
    - Generate synthetic datasets consistent with the specified causal structure.

    **High-Level Usage:**
    ---------------------
    # Quick start, one-call pipeline:
    cdg = CausalDataGenerator(num_samples=500, seed=42)
    data = cdg.generate_data_pipeline(
        total_nodes=10,
        root_nodes=3,
        edges=15,
        equation_type='random',
        root_distributions_override={
            'a': {'dist': 'normal', 'mean': 5.0, 'std': 1.0}
        }
    )

    # More controlled approach:
    cdg = CausalDataGenerator(num_samples=500, seed=42)
    G, roots = cdg.generate_random_graph(total_nodes=10, root_nodes=3, edges=15)

    # Manually set distributions for root nodes:
    cdg.set_root_distributions({'a': {'dist': 'normal', 'mean': 0, 'std': 1},
                                'b': {'dist': 'uniform', 'low': 0, 'high': 10}})

    # Optionally set noise for some non-root nodes:
    cdg.set_noise_for_nodes({'d': {'type': 'uniform', 'params': {'low': 0, 'high': 2}}})

    # Assign random equations to non-root nodes:
    cdg.assign_equations_to_graph_nodes(equation_type='random')

    # Generate the data:
    data = cdg.generate_data()
    """

    def __init__(self, num_samples=100, default_noise_type='normal', default_noise_params=None, seed=None):
        self.num_samples = num_samples
        self.default_noise_type = default_noise_type
        self.default_noise_params = default_noise_params or {'mean': 0, 'std': 1}
        self.node_noise_info = {}
        self.root_nodes = set()
        self.G = None
        self.node_functions = None
        self.root_ranges = {}
        self.unobserved_confounders = set()
        self.data = None
        self.node_metadata = {}
        self.seed = seed
        self.rng = default_rng(seed)
        self.add_noise_to_root_nodes = False
        self.nominal_nodes = {}  # e.g., {'d': {'num_classes': 3, 'input': 'a'}}

    def generate_data_pipeline(self, total_nodes, root_nodes, edges,
                               equation_type='random',
                               root_distributions_override=None,
                               default_noise_type=None,
                               default_noise_params=None,
                               node_noise_override=None,
                               overall_root_distribution_type=None,
                               add_noise_to_root_nodes=False,
                               categorical_nodes=None,  # explicit mapping  {"d": {"input": "a", "num_classes": 4}, ...}
                               categorical_percentage=0.0,  # fraction of *non-root* nodes to flip to nominal
                               max_categories=6,  # upper bound when we sample num_classes ∈ [2, max_categories]
                               categorical_root_nodes=None  # list like ["b", "c"]  → root nodes made categorical
                               ):
        """
        High-level method to generate data with a single call.

        Parameters:
        - total_nodes (int): Total number of nodes in the DAG.
        - root_nodes (int): Number of root (exogenous) nodes.
        - edges (int): Total number of edges in the DAG.
        - equation_type (str): 'random', 'linear', or 'non_linear'.
        - root_distributions_override (dict): Optional dict to manually set distributions for root nodes.
        - default_noise_type (str): Override default noise type for all non-root nodes.
        - default_noise_params (dict): Override default noise parameters.
        - node_noise_override (dict): Optional dict to set noise for specific nodes.
        - overall_root_distribution_type (str): Set the same distribution type for all root nodes.
                                               Options: 'uniform', 'normal', 'exponential', 'beta'
        - add_noise_to_root_nodes (bool): Whether to add noise to root nodes in addition to their distributions.

        Returns:
        - pd.DataFrame: Generated dataset.
        """
        # Override default noise settings if provided
        if default_noise_type is not None:
            self.default_noise_type = default_noise_type
        if default_noise_params is not None:
            self.default_noise_params = default_noise_params

        # Store setting for adding noise to root nodes
        self.add_noise_to_root_nodes = add_noise_to_root_nodes

        self.generate_random_graph(total_nodes, root_nodes, edges)

        # If user provided custom distributions, set them
        if root_distributions_override:
            self.set_root_distributions(root_distributions_override)

        # Set custom noise for specific nodes if provided
        if node_noise_override:
            self.set_noise_for_nodes(node_noise_override)

        # Assign equations (this also assigns random distributions if not already set)
        self.assign_equations_to_graph_nodes(equation_type=equation_type)

        # Apply overall root distribution type if specified
        if overall_root_distribution_type:
            self._set_overall_root_distribution_type(overall_root_distribution_type)

        # If still no root distributions, assign them randomly
        if not self.root_ranges:
            self.root_ranges = self._assign_random_distributions_to_root_nodes(list(self.root_nodes))

        # ------------------------------------------------------------------
        #  CATEGORICAL / NOMINAL CONFIGURATION
        # ------------------------------------------------------------------
        if categorical_nodes:
            # user gave an explicit dict → just register it
            self.set_nominal_nodes(categorical_nodes)

        # Root nodes that should themselves be categorical
        if categorical_root_nodes:
            if isinstance(categorical_root_nodes, str) and categorical_root_nodes.lower() == "all":
                target_roots = list(self.root_nodes)  # every root
            else:
                target_roots = list(categorical_root_nodes)  # user-specified subset

            self._set_categorical_roots(target_roots, max_categories)

        # Randomly sample a percentage of *non-root* nodes to become categorical
        if categorical_percentage > 0:
            self._sample_random_nominals(
                categorical_percentage=categorical_percentage,
                max_categories=max_categories
            )
        # ------------------------------------------------------------------

        return self.generate_data()

    def generate_random_graph(self, total_nodes, root_nodes, edges):
        """
        Generates a random DAG with given number of nodes, root nodes, and edges.

        Parameters:
        - total_nodes (int): The total number of nodes in the graph.
        - root_nodes (int): The number of root nodes (no incoming edges).
        - edges (int): The total number of edges.

        Returns:
        - (nx.DiGraph, list): The generated DAG and a list of root nodes.
        """
        G, roots = self._generate_graph_impl(total_nodes, root_nodes, edges)
        self.G = G
        self.root_nodes = set(roots)
        return G, roots

    def set_root_distributions(self, root_dist):
        """
        Manually sets distributions for root nodes.

        Parameters:
        - root_dist (dict): Mapping from root node name to its distribution config.

        Example:
        --------
        cdg.set_root_distributions({'a': {'dist': 'normal', 'mean': 0, 'std': 1}})
        """
        missing_roots = set(root_dist.keys()) - self.root_nodes
        if missing_roots:
            raise ValueError(f"Nodes {missing_roots} are not root nodes. Only root nodes can have distributions.")
        self.root_ranges.update(root_dist)

    def set_noise_for_nodes(self, node_noise_info):
        """
        Assigns noise types and parameters to specified non-root nodes.

        Parameters:
        - node_noise_info (dict): e.g. {'d': {'type': 'uniform', 'params': {'low':0,'high':2}}}
        """
        root_nodes_with_noise = self.root_nodes.intersection(node_noise_info.keys())
        if root_nodes_with_noise:
            raise ValueError(f"Cannot assign noise to root nodes: {root_nodes_with_noise}.")

        self.node_noise_info.update(node_noise_info)

    def assign_equations_to_graph_nodes(self, equation_type='random'):
        """
        Assigns equations to non-root nodes.

        Parameters:
        - equation_type (str): 'random', 'linear', or 'non_linear'.
        """
        if self.G is None:
            raise ValueError("Graph not defined. Call generate_random_graph first.")

        # If no root distributions yet, assign random ones now
        if not self.root_ranges:
            self.root_ranges = self._assign_random_distributions_to_root_nodes(list(self.root_nodes))

        self.node_functions = self._assign_equations_impl(self.G, equation_type=equation_type)

    def generate_data(self):
        """
        Generates data using the defined graph, root distributions, equations, and noise.

        Returns:
        - pd.DataFrame: The generated dataset.
        """
        if self.G is None or self.node_functions is None or not self.root_ranges:
            raise ValueError("Cannot generate data. Ensure the graph, equations, and root distributions are defined.")

        self.data = self._define_functions_and_generate_data_impl(self.G, self.node_functions, self.root_ranges)
        return self.data

    def remove_node_and_simulate_unobserved_confounder(self, node_to_remove):
        """
        Simulates an unobserved confounder by removing a node from the observed data and graph.

        Parameters:
        - node_to_remove (str): Name of the node to treat as an unobserved confounder.
        """
        if self.G is None:
            raise ValueError("Graph not defined. Generate the graph first.")

        if node_to_remove not in self.G.nodes:
            print(f"Node '{node_to_remove}' does not exist in the graph.")
            return

        if len(self.G.nodes) <= 1:
            raise ValueError("Cannot remove the last node from the graph.")

        if node_to_remove in self.unobserved_confounders:
            print(f"Node '{node_to_remove}' is already an unobserved confounder.")
            return

        if not hasattr(self, 'unobserved_confounder_metadata'):
            self.unobserved_confounder_metadata = {}

        self.unobserved_confounder_metadata[node_to_remove] = {
            'edges_in': list(self.G.in_edges(node_to_remove)),
            'edges_out': list(self.G.out_edges(node_to_remove)),
            'is_root': node_to_remove in self.root_nodes,
            'noise_info': self.node_noise_info.get(node_to_remove, None),
            'root_range': self.root_ranges.get(node_to_remove, None)
        }

        if node_to_remove in self.root_nodes:
            self.root_nodes.remove(node_to_remove)
            if node_to_remove in self.root_ranges:
                del self.root_ranges[node_to_remove]

        if node_to_remove in self.node_noise_info:
            del self.node_noise_info[node_to_remove]

        self.G.remove_node(node_to_remove)

        if self.data is not None and node_to_remove in self.data.columns:
            self.data.drop(columns=[node_to_remove], inplace=True)

        self.unobserved_confounders.add(node_to_remove)

    def plot_graph(self):
        """
        Plots the current DAG with root distributions and noise info.
        """
        if self.G is None:
            raise ValueError("Graph is not defined.")

        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, with_labels=False, node_size=1500, node_color='lightblue', arrowsize=20)

        node_labels = {}
        for node in self.G.nodes():
            label = f"{node}"
            if node in self.root_nodes:
                dist_info = self.root_ranges.get(node, {})
                if dist_info:
                    dist_type = dist_info.get('dist', '')
                    params = ', '.join(f"{k}={v:.2f}" for k, v in dist_info.items() if k != 'dist')
                    label += f"\n{dist_type}({params})"
            else:
                noise_info = self.node_noise_info.get(node, {'type': self.default_noise_type,
                                                             'params': self.default_noise_params})
                noise_type = noise_info['type']
                noise_params = noise_info['params']
                params = ', '.join(f"{k}={v}" for k, v in noise_params.items())
                label += f"\nNoise: {noise_type}({params})"
            node_labels[node] = label

        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8)

        if self.node_functions is not None:
            edge_labels = {}
            for (parents, child), expr in self.node_functions.items():
                for parent in parents:
                    if self.G.has_edge(parent, child):
                        edge_labels[(parent, child)] = expr
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title("Causal Graph", fontsize=16)
        plt.axis('off')
        plt.show()

    def plot_interactive_graph(self, output_file="graph_with_highlight.html"):
        """
        Visualizes the causal graph using pyvis for interactive inspection.
        """
        if self.G is None:
            raise ValueError("Graph is not defined. Please generate the graph first.")

        G_vis = self.G.copy()

        for node in self.unobserved_confounders:
            G_vis.add_node(node)
            metadata = self.unobserved_confounder_metadata.get(node, {})
            for src, tgt in metadata.get('edges_in', []):
                G_vis.add_edge(src, node)
            for src, tgt in metadata.get('edges_out', []):
                G_vis.add_edge(node, tgt)

        net = Network(notebook=self.is_notebook(), directed=True)

        # Add nodes
        for node in G_vis.nodes:
            if node in self.unobserved_confounders:
                net.add_node(node, label=str(node), color="white", shape="circle",
                             title=f"Unobserved Confounder {node}", borderWidth=2, borderWidthSelected=2,
                             opacity=0.5)
            elif node in self.root_nodes:
                dist_info = self.root_ranges.get(node, {})
                dist_type = dist_info.get('dist', '')
                params = ', '.join(f"{k}={v:.2f}" for k, v in dist_info.items() if k != 'dist')
                title = f"Root Node {node}<br>{dist_type}({params})"
                net.add_node(node, label=str(node), color="red", title=title)
            else:
                noise_info = self.node_noise_info.get(node, {'type': self.default_noise_type,
                                                             'params': self.default_noise_params})
                noise_type = noise_info['type']
                noise_params = noise_info['params']
                params = ', '.join(f"{k}={v}" for k, v in noise_params.items())
                title = f"Node {node}<br>Noise: {noise_type}({params})"
                net.add_node(node, label=str(node), color="grey", title=title)

        # Add edges
        for edge in G_vis.edges:
            src, tgt = edge
            if src in self.unobserved_confounders or tgt in self.unobserved_confounders:
                net.add_edge(src, tgt, color='grey', dashes=True)
            else:
                net.add_edge(src, tgt, color='black')

        net.set_options("""
        var options = {
          "edges": {
            "color": {
              "inherit": false
            }
          },
          "interaction": {
            "hover": true,
            "selectConnectedEdges": false
          },
          "physics": {
            "enabled": true
          }
        }
        """)

        net.save_graph(output_file)
        print(f"Graph has been saved to {output_file}. Injecting custom JavaScript...")
        self.inject_custom_js(output_file)

    def inject_custom_js(self, output_file):
        """
        Injects custom JavaScript into the saved HTML file for graph interactions.
        """
        # Adjust as needed, depending on your JS file location and packaging
        js_path = pkg_resources.resource_filename('scdg', 'static/js/graph_interactions.js')

        if not os.path.exists(js_path):
            raise FileNotFoundError(f"JavaScript file not found at {js_path}.")

        script_tag = f'<script src="{js_path}"></script>'

        with open(output_file, 'r') as file:
            content = file.read()

        content = content.replace('</body>', f'{script_tag}\n</body>')

        with open(output_file, 'w') as file:
            file.write(content)

        print(f"JavaScript file has been linked in {output_file}")

        # Attempt to open the HTML file in the default web browser
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_file)
            elif 'Darwin' in str(os.uname()):
                subprocess.call(['open', output_file])
            else:
                subprocess.call(['xdg-open', output_file])
        except Exception as e:
            print(f"Could not open the file automatically. Please open {output_file} manually.")

    def save_to_pickle(self, filename_prefix="rg"):
        """
        Saves the object to a pickle file with a name derived from graph properties.
        """
        # This depends on your original implementation of get_generation_type
        # If not available, you can skip or implement a stub.
        connection_type = "unknown"
        if self.G:
            # If you have a method to detect generation type:
            # connection_type = get_generation_type(self).replace(" ", "").lower()
            pass

        total_nodes = len(self.G.nodes) if self.G else 0
        root_nodes = len(self.root_nodes)
        edges = len(self.G.edges) if self.G else 0
        filename = f"{filename_prefix}_{connection_type}_{total_nodes}n_{root_nodes}rn_{edges}e.pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

        print(f"Object saved as '{filename}'.")

    def load_from_yaml(self, yaml_file):
        """
        Loads graph structure, root node distributions, node functions, and noise from a YAML file.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        nodes = config.get('nodes', [])
        edges = config.get('edges', [])
        if not nodes or not edges:
            raise ValueError("YAML file must contain 'nodes' and 'edges' sections.")

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        self.G = G
        self.root_nodes = {node for node, deg in G.in_degree() if deg == 0}

        root_ranges = config.get('root_distributions', {})
        missing_roots = self.root_nodes - set(root_ranges.keys())
        if missing_roots:
            raise ValueError(f"Root nodes {missing_roots} are missing distributions in the YAML file.")
        self.root_ranges = root_ranges

        node_functions = config.get('node_functions', {})
        if not node_functions:
            raise ValueError("YAML must contain 'node_functions' section.")
        formatted_node_functions = {}
        for key, expr in node_functions.items():
            parents_child = key.strip().split('->')
            if len(parents_child) != 2:
                raise ValueError(f"Invalid edge function key: '{key}'. Format: 'parent1,parent2->child'.")
            parents = tuple(p.strip() for p in parents_child[0].split(','))
            child = parents_child[1].strip()
            formatted_node_functions[(parents, child)] = expr
        self.node_functions = formatted_node_functions

        node_noise_info = config.get('noise', {})
        self.set_noise_for_nodes(node_noise_info)
        print("Configuration loaded from YAML file.")

    @staticmethod
    def load_from_pickle(filename):
        """
        Loads a CausalDataGenerator object from a pickle file.
        """
        with open(filename, "rb") as file:
            obj = pickle.load(file)
        print(f"Object loaded from '{filename}'.")
        return obj

    def is_notebook(self):
        """
        Check if running inside a Jupyter notebook for pyvis rendering.
        """
        try:
            shell = get_ipython().__class__.__name__
            if 'ZMQInteractiveShell' in shell:
                return True
            else:
                return False
        except NameError:
            return False

    def convert_to_manufacturing(self,

                                 strategy: str = "dirichlet",
                                 alpha: float = 0.7,
                                 k_min: int = 3,
                                 k_max: int = 8,
                                 seed: int | None = None,
                                 return_mapper: bool = False):
        """
        Wrap the current DAG (self.G) into a causalAssembly ProductionLineGraph.

        Parameters
        ----------
        strategy : {"dirichlet"}
            Currently only the Dirichlet topo-slice strategy is implemented.
            Future: plug other mappers here.
        alpha : float
            Dirichlet concentration (1 ≈ equal sizes, <1 ⇒ more unequal).
        k_min, k_max : int
            Min / max number of stations to draw.
        seed : int | None
            RNG seed for reproducibility.
        return_mapper : bool
            If True, also return the dict {station → [nodes]}.

        Returns
        -------
        ProductionLineGraph
            The manufacturing DAG with renamed nodes (e.g. "Station2_b").
        """
        if self.G is None:
            raise ValueError("Graph not defined. Generate or load a DAG first.")

        # 1. ensure node labels are strings
        g_str = nx.relabel_nodes(self.G, str)

        # 2. choose station-split strategy
        if strategy == "dirichlet":
            cell_mapper = self._dirichlet_station_mapper(
                g_str, alpha=alpha, k_min=k_min, k_max=k_max, seed=seed
            )
        else:
            raise NotImplementedError(f"Unknown strategy '{strategy}'")

        # 3. build the manufacturing graph
        pline = ProductionLineGraph.from_nx(g_str, cell_mapper)

        # 4. store for later use (optional but convenient)
        self.pline = pline

        return (pline, cell_mapper) if return_mapper else pline

    def set_nominal_nodes(self, nominal_node_info):
        """
        Define some non-root nodes as nominal (categorical) with softmax-based class sampling.

        Parameters:
        - nominal_node_info (dict): e.g., {'d': {'input': 'a', 'num_classes': 3}}
                                 or {'d': {'input': ['a', 'b'], 'num_classes': 3}}

        Each entry specifies:
            - 'input': parent node(s) to base logits on (single string or list of strings)
            - 'num_classes': number of nominal classes to generate
        """
        for node, info in nominal_node_info.items():
            if node in self.root_nodes:
                raise ValueError(f"Nominal node '{node}' cannot be a root node.")
            if 'input' not in info or 'num_classes' not in info:
                raise ValueError(f"Each nominal node must specify 'input' and 'num_classes'.")
            
            # Convert single input to list for consistency
            if isinstance(info['input'], str):
                info['input'] = [info['input']]
            
            self.nominal_nodes[node] = info

    # -------------------- INTERNAL HELPER METHODS --------------------

    def _generate_graph_impl(self, total_nodes, root_nodes, edges):
        # Place your original random graph generation logic here:
        # We use the user's original logi=
        # lkc as posted, slightly refactored:

        if root_nodes >= total_nodes:
            raise ValueError("Number of root nodes must be less than total nodes.")

        G = nx.DiGraph()
        node_names = self._generate_node_names(total_nodes)
        roots = node_names[:root_nodes]
        non_roots = node_names[root_nodes:]

        # Minimum edges: at least connect roots to form a weakly connected structure
        non_root_nodes = total_nodes - root_nodes
        min_required_edges = root_nodes + (non_root_nodes - 1)
        if edges < min_required_edges:
            raise ValueError(f"At least {min_required_edges} edges required.")

        # Randomly generate edges
        mandatory_edges = [(self.rng.choice(roots), nr) for nr in non_roots]
        G.add_edges_from(mandatory_edges)
        remaining_edges = edges - len(mandatory_edges)

        # Possible edges from roots to non-roots (excluding mandatory)
        available_root_to_non_root_edges = [(r, n) for r in roots for n in non_roots if (r, n) not in G.edges]
        self.rng.shuffle(available_root_to_non_root_edges)
        for edge in available_root_to_non_root_edges[:remaining_edges]:
            G.add_edge(*edge)
        remaining_edges -= min(remaining_edges, len(available_root_to_non_root_edges))

        if remaining_edges > 0:
            available_non_root_to_non_root_edges = [(n1, n2) for i, n1 in enumerate(non_roots) for n2 in
                                                    non_roots[i + 1:] if (n1, n2) not in G.edges]
            self.rng.shuffle(available_non_root_to_non_root_edges)
            for edge in available_non_root_to_non_root_edges[:remaining_edges]:
                G.add_edge(*edge)

        return G, roots

    def _generate_node_names(self, total_nodes):
        # Same node naming logic as before
        alphabet = [chr(i) for i in range(97, 123)]
        node_names = []
        for i in range(total_nodes):
            if i < 26:
                node_names.append(alphabet[i])
            else:
                prefix = alphabet[i // 26 - 1]
                suffix = str(i % 26 + 1)
                node_names.append(prefix + suffix)
        return node_names

    def _assign_random_distributions_to_root_nodes(self, root_nodes):
        # Assign random distributions (original logic)
        distribution_types = ['uniform', 'normal', 'exponential', 'beta', 'truncated_normal', 'lognormal',
                              'categorical', 'categorical_non_uniform']
        root_ranges = {}

        for node in root_nodes:
            dist_type = self.rng.choice(distribution_types)
            if dist_type == 'uniform':
                low = self.rng.uniform(0, 10)
                high = self.rng.uniform(low, low + 10)
                root_ranges[node] = {'dist': 'uniform', 'low': low, 'high': high}
            elif dist_type == 'normal':
                mean = self.rng.uniform(0, 5)      # Reduce from [0,10] to [0,5]
                std = self.rng.uniform(0.1, 2.0)   # Reduce from [0.1,5.0] to [0.1,2.0]
                root_ranges[node] = {'dist': 'normal', 'mean': mean, 'std': std}
            elif dist_type == 'exponential':
                scale = self.rng.uniform(1, 10)
                root_ranges[node] = {'dist': 'exponential', 'scale': scale}
            elif dist_type == 'beta':
                a = self.rng.uniform(0.5, 5.0)
                b = self.rng.uniform(0.5, 5.0)
                root_ranges[node] = {'dist': 'beta', 'a': a, 'b': b}
            elif dist_type == 'truncated_normal':
                mean = self.rng.uniform(0, 10)
                std = self.rng.uniform(0.1, 5.0)
                low = mean - 2 * std  # Truncate at 2 std devs
                high = mean + 2 * std
                root_ranges[node] = {'dist': 'truncated_normal', 'mean': mean, 'std': std, 'low': low, 'high': high}
            elif dist_type == 'lognormal':
                mean = self.rng.uniform(0, 2)
                sigma = self.rng.uniform(0.1, 1.0)
                root_ranges[node] = {'dist': 'lognormal', 'mean': mean, 'sigma': sigma}
            elif dist_type == 'categorical':
                num_classes = self.rng.integers(2, 6)  # 2-5 categories
                root_ranges[node] = {'dist': 'categorical', 'num_classes': num_classes}
            elif dist_type == 'categorical_non_uniform':
                num_classes = self.rng.integers(2, 6)  # 2-5 categories
                # Generate skewed probabilities
                skew_factor = self.rng.uniform(0.1, 0.9)
                probabilities = self._generate_skewed_probabilities(num_classes, skew_factor)
                root_ranges[node] = {
                    'dist': 'categorical_non_uniform',
                    'num_classes': num_classes,
                    'probabilities': probabilities
                }

        return root_ranges

    def _assign_equations_impl(self, G, equation_type='random'):
        # Logic from assign_equations_to_graph_nodes from original code:

        roots = [node for node in G.nodes if G.in_degree(node) == 0]
        non_roots = [node for node in G.nodes if G.in_degree(node) > 0]
        single_parent_nodes = {child: list(G.predecessors(child))[0] for child in non_roots if G.in_degree(child) == 1}
        multi_parent_nodes = {child: list(G.predecessors(child)) for child in non_roots if G.in_degree(child) > 1}

        # Depending on equation_type
        if equation_type == 'random':
            single_parent_equations = {}
            for child, parent in single_parent_nodes.items():
                non_linear = self.rng.choice([True, False])
                eq = self._assign_random_equations_to_single_parent_nodes({child: parent}, non_linear=non_linear)
                single_parent_equations.update(eq)

            multi_parent_equations = {}
            for child, parents in multi_parent_nodes.items():
                non_linear = self.rng.choice([True, False])
                eq = self._assign_random_equations_to_multiparent_nodes({child: parents}, non_linear=non_linear)
                multi_parent_equations.update(eq)

        elif equation_type == 'linear':
            single_parent_equations = self._assign_random_equations_to_single_parent_nodes(single_parent_nodes,
                                                                                           non_linear=False)
            multi_parent_equations = self._assign_random_equations_to_multiparent_nodes(multi_parent_nodes,
                                                                                        non_linear=False)
        elif equation_type == 'non_linear':
            single_parent_equations = self._assign_random_equations_to_single_parent_nodes(single_parent_nodes,
                                                                                           non_linear=True)
            multi_parent_equations = self._assign_random_equations_to_multiparent_nodes(multi_parent_nodes,
                                                                                        non_linear=True)
        else:
            raise ValueError("equation_type must be 'random', 'linear', or 'non_linear'")

        node_functions = {}
        node_functions.update(single_parent_equations)
        node_functions.update(multi_parent_equations)
        return node_functions

    def _assign_random_equations_to_single_parent_nodes(self, single_parent_nodes, non_linear=False):
        # Same logic as original
        possible_functions_linear = [
            "{coeff} * {0} + {const}",
            "{coeff} * {0}",
            "{0} + {const}",
            "{0} - {const}",
            "{const} * {0}",
        ]

        possible_functions_non_linear = [
            "np.sin({coeff} * {0}) + {const}",
            "np.cos({coeff} * {0}) + {const}",
            "np.exp(np.clip({coeff} * {0}, -5, 5)) + {const}",  # Add clipping
            "np.log(np.abs({coeff} * {0}) + 1) + {const}",
            "np.sqrt(np.abs({coeff} * {0})) + {const}",
            "np.sin({coeff} * {0}) + {const}",  # Replace tan with sin
        ]

        node_functions = {}
        for child, parent in single_parent_nodes.items():
            if non_linear:
                possible_functions = possible_functions_non_linear
            else:
                possible_functions = possible_functions_linear
            function_template = self.rng.choice(possible_functions)
            coeff = round(self.rng.uniform(-2, 2), 2)
            while coeff == 0:
                coeff = round(self.rng.uniform(-2, 2), 2)
            const = round(self.rng.uniform(-5, 5), 2)
            power = round(self.rng.uniform(1.0, 2.0), 2)
            # Fix: Use the parent as a variable name directly in the expression
            # First format with placeholders, then replace {0} with actual parent name
            expression = function_template.replace('{0}', parent)
            expression = expression.format(coeff=coeff, const=const, power=power)
            node_functions[(parent,), child] = expression
        return node_functions

    def _assign_random_equations_to_multiparent_nodes(self, multi_parent_nodes, non_linear=False):
        # Same logic as original
        node_functions = {}
        for child, parents in multi_parent_nodes.items():
            if len(parents) < 2:
                raise ValueError(f"Node {child} has fewer than two parents.")
            terms = []
            for parent in parents:
                coeff = round(self.rng.uniform(-2, 2), 2)
                while coeff == 0:
                    coeff = round(self.rng.uniform(-2, 2), 2)
                const = round(self.rng.uniform(-5, 5), 2)
                power = round(self.rng.uniform(1.0, 2.0), 2)
                if non_linear:
                    function_template = self.rng.choice([
                        "{coeff} * np.abs({parent}) ** {power}",
                        "np.sin({coeff} * {parent})",
                        "np.cos({coeff} * {parent})",
                        "np.log(np.abs({coeff} * {parent}) + 1)",
                        "np.sqrt(np.abs({coeff} * {parent}))",
                    ])
                else:
                    function_template = "{coeff} * {parent}"
                term = function_template.format(parent=parent, coeff=coeff, power=power)
                terms.append(term)
            expression = terms[0]
            for term in terms[1:]:
                op = self.rng.choice([' + ', ' - '])
                expression = f"({expression}){op}({term})"
            const = round(self.rng.uniform(-5, 5), 2)
            expression = f"({expression}) + {const}"
            node_functions[tuple(parents), child] = expression
        return node_functions

    def _create_function_from_expression(self, expr, variables, node_name):
        allowed_locals = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "math_e": np.e,
            "pi": np.pi,
            "abs": np.abs,
            "log1p": np.log1p
        }

        def func(**kwargs):
            try:
                # Clip input values to prevent extreme computations
                clipped_kwargs = {}
                for var in variables:
                    clipped_kwargs[var] = np.clip(kwargs[var], -10, 10)

                local_scope = {var: clipped_kwargs[var] for var in variables}

                with np.errstate(all='raise'):
                    result = eval(expr, {"__builtins__": None}, {**allowed_locals, **local_scope})

                # Clip result to prevent extreme values
                result = np.clip(result, -1e6, 1e6)

                # Check for NaN/Inf and handle gracefully
                if np.isnan(result).any() or np.isinf(result).any():
                    print(f"Warning: NaN/Inf detected in node '{node_name}' with expression '{expr}'")
                    # Replace NaN/Inf with bounded random values
                    nan_mask = np.isnan(result) | np.isinf(result)
                    result[nan_mask] = self.rng.uniform(-5, 5, size=np.sum(nan_mask))

                # Add noise
                noise_info = self.node_noise_info.get(node_name, {'type': self.default_noise_type,
                                                          'params': self.default_noise_params})
                noise_type = noise_info['type']
                noise_params = noise_info['params']

                if noise_type == 'normal':
                    noise = self.rng.normal(loc=noise_params.get('mean', 0),
                                    scale=noise_params.get('std', 1),
                                    size=self.num_samples)
                elif noise_type == 'uniform':
                    noise = self.rng.uniform(low=noise_params.get('low', -1),
                                     high=noise_params.get('high', 1),
                                     size=self.num_samples)
                elif noise_type == 'exponential':
                    noise = self.rng.exponential(scale=noise_params.get('scale', 1),
                                         size=self.num_samples)
                elif noise_type == 'beta':
                    a = noise_params.get('a', 0.5)
                    b = noise_params.get('b', 0.5)
                    noise = self.rng.beta(a, b, size=self.num_samples)
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")

                combined = result + noise

                # Final bounds check
                combined = np.clip(combined, -1e6, 1e6)

                return combined

            except FloatingPointError as e:
                print(f"Floating point error in '{expr}': {e}")
                return self.rng.uniform(-5, 5, size=self.num_samples)  # Return bounded random values
            except Exception as e:
                print(f"Error evaluating '{expr}' with variables {variables}: {e}")
                return self.rng.uniform(-5, 5, size=self.num_samples)  # Return bounded random values

        return func

    def _define_functions_and_generate_data_impl(self, G, node_functions, root_ranges):
        # Generate data from root distributions
        data = {}
        for node, range_info in root_ranges.items():
            if range_info["dist"] == "uniform":
                data[node] = self.rng.uniform(range_info["low"], range_info["high"], self.num_samples)
            elif range_info["dist"] == "normal":
                data[node] = self.rng.normal(range_info["mean"], range_info["std"], self.num_samples)
            elif range_info["dist"] == "exponential":
                data[node] = self.rng.exponential(scale=range_info["scale"], size=self.num_samples)
            elif range_info["dist"] == "beta":
                data[node] = self.rng.beta(range_info["a"], range_info["b"], self.num_samples)
            elif range_info["dist"] == "truncated_normal":
                if not SCIPY_AVAILABLE:
                    raise ImportError("scipy is required for truncated_normal distribution. Please install scipy.")

                # Use scipy.stats.truncnorm for truncated normal distribution
                a = (range_info["low"] - range_info["mean"]) / range_info["std"]
                b = (range_info["high"] - range_info["mean"]) / range_info["std"]

                # Validate parameters
                if range_info["std"] <= 0:
                    raise ValueError(f"Standard deviation must be positive for truncated_normal distribution")
                if range_info["low"] >= range_info["high"]:
                    raise ValueError(f"Low bound must be less than high bound for truncated_normal distribution")

                data[node] = truncnorm.rvs(a, b, loc=range_info["mean"], scale=range_info["std"],
                                           size=self.num_samples, random_state=self.rng)
            elif range_info["dist"] == "lognormal":
                data[node] = self.rng.lognormal(range_info["mean"], range_info["sigma"], self.num_samples)
            elif range_info["dist"] == "categorical":  # Uniform categorical
                k = int(range_info["num_classes"])
                data[node] = self.rng.integers(0, k, self.num_samples)
            elif range_info["dist"] == "categorical_non_uniform":  # Non-uniform categorical
                k = int(range_info["num_classes"])
                probabilities = range_info.get("probabilities", None)
                if probabilities is None:
                    # Fallback to uniform if no probabilities provided
                    data[node] = self.rng.integers(0, k, self.num_samples)
                else:
                    # Use provided probabilities for non-uniform sampling
                    data[node] = self.rng.choice(k, size=self.num_samples, p=probabilities)
            else:
                raise ValueError(f"Invalid distribution type for root node '{node}': {range_info['dist']}")

            # Add noise to root nodes if requested
            if hasattr(self, 'add_noise_to_root_nodes') and self.add_noise_to_root_nodes:
                noise_info = self.node_noise_info.get(node, {'type': self.default_noise_type,
                                                             'params': self.default_noise_params})
                noise_type = noise_info['type']
                noise_params = noise_info['params']

                if noise_type == 'normal':
                    noise = self.rng.normal(loc=noise_params.get('mean', 0),
                                            scale=noise_params.get('std', 1),
                                            size=self.num_samples)
                elif noise_type == 'uniform':
                    noise = self.rng.uniform(low=noise_params.get('low', -1),
                                             high=noise_params.get('high', 1),
                                             size=self.num_samples)
                elif noise_type == 'exponential':
                    noise = self.rng.exponential(scale=noise_params.get('scale', 1),
                                                 size=self.num_samples)
                elif noise_type == 'beta':
                    a = noise_params.get('a', 0.5)
                    b = noise_params.get('b', 0.5)
                    noise = self.rng.beta(a, b, size=self.num_samples)
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")

                data[node] += noise

        # Rest of the method remains the same
        # Create callable functions for each child node
        functions = {}
        for (parents, child) in node_functions:
            expr = node_functions[(parents, child)]
            # Validate variables - improved extraction for complex expressions
            allowable_functions = {'np', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'math_e', 'pi', 'abs', 'log1p', 'clip', 'select', 'where'}
            parameter_names = {'default'}  # Add parameter names to exclude
            expr_vars = set()
            # Extract variables more carefully, handling np.select and np.where
            import re
            # Find all variable names that are not function names or parameters
            # Extract variables that look like identifiers (letters/underscores or standalone digits)
            # Exclude numeric constants by checking if they're part of mathematical operations
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            all_matches = re.findall(var_pattern, expr)
            
            for match in all_matches:
                if match not in allowable_functions and match not in parameter_names:
                    expr_vars.add(match)
            
            # Also check for numeric variables (like "0", "1") but only if they appear as standalone
            # Check if the numeric part exists independently (not part of a number like "3.59")
            standalone_numbers = re.findall(r'(?<!\d)\b([0-9]+)\b(?!\d)', expr)
            for num_str in standalone_numbers:
                # Only add if it's a valid variable name (single digit without decimal context)
                if num_str in [str(p) for p in parents]:
                    expr_vars.add(num_str)
            if set(parents) - expr_vars:
                raise ValueError(f"Expression '{expr}' for '{child}' does not use all parent variables.")
            if expr_vars - set(parents):
                raise ValueError(f"Expression '{expr}' for '{child}' has variables not in parents.")
            functions[child] = self._create_function_from_expression(expr, list(parents), node_name=child)

        # Evaluate node data in topological order
        for node in nx.topological_sort(G):
            if node not in root_ranges:  # non-root
                if node in self.nominal_nodes:
                    # Nominal node – use generate_nominal_category
                    info = self.nominal_nodes[node]
                    input_nodes = info['input']  # Now a list
                    num_classes = info['num_classes']
                    
                    # Handle both single and multiple inputs
                    if len(input_nodes) == 1:
                        # Single input (backward compatibility)
                        X = data[input_nodes[0]]
                        y, _, _ = generate_nominal_category(X, num_classes=num_classes, seed=self.seed)
                    else:
                        # Multiple inputs
                        X_dict = {input_node: data[input_node] for input_node in input_nodes}
                        y, _, _ = generate_nominal_category(X_dict, num_classes=num_classes, seed=self.seed)
                    
                    data[node] = y
                else:
                    args = {u: data[u] for u in G.predecessors(node)}
                    data[node] = functions[node](**args)

        return pd.DataFrame(data)

    def _set_overall_root_distribution_type(self, distribution_type):
        """
        Sets the same distribution type for all root nodes while preserving
        any existing settings or creating new random parameters.

        Parameters:
        - distribution_type (str): One of 'uniform', 'normal', 'exponential', 'beta', 'truncated_normal', 'lognormal', 'categorical', 'categorical_non_uniform'
        """
        if distribution_type not in ['uniform', 'normal', 'exponential', 'beta', 'truncated_normal', 'lognormal',
                                     'categorical', 'categorical_non_uniform']:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

        for node in self.root_nodes:
            # If the node already has distribution settings, preserve parameters if possible
            if node in self.root_ranges:
                old_settings = self.root_ranges[node]
                # Create new settings with same distribution type but preserve parameters when possible
                if distribution_type == 'uniform':
                    self.root_ranges[node] = {
                        'dist': 'uniform',
                        'low': old_settings.get('low', self.rng.uniform(0, 10)),
                        'high': old_settings.get('high', self.rng.uniform(10, 20))
                    }
                elif distribution_type == 'normal':
                    self.root_ranges[node] = {
                        'dist': 'normal',
                        'mean': old_settings.get('mean', self.rng.uniform(0, 10)),
                        'std': old_settings.get('std', self.rng.uniform(0.1, 5.0))
                    }
                elif distribution_type == 'exponential':
                    self.root_ranges[node] = {
                        'dist': 'exponential',
                        'scale': old_settings.get('scale', self.rng.uniform(1, 10))
                    }
                elif distribution_type == 'beta':
                    self.root_ranges[node] = {
                        'dist': 'beta',
                        'a': old_settings.get('a', self.rng.uniform(0.5, 5.0)),
                        'b': old_settings.get('b', self.rng.uniform(0.5, 5.0))
                    }
                elif distribution_type == 'truncated_normal':
                    mean = old_settings.get('mean', self.rng.uniform(0, 10))
                    std = old_settings.get('std', self.rng.uniform(0.1, 5.0))
                    low = old_settings.get('low', mean - 2 * std)
                    high = old_settings.get('high', mean + 2 * std)

                    # Ensure valid parameters
                    if low >= high:
                        low = mean - 1.5 * std
                        high = mean + 1.5 * std

                    self.root_ranges[node] = {
                        'dist': 'truncated_normal',
                        'mean': mean,
                        'std': std,
                        'low': low,
                        'high': high
                    }
                elif distribution_type == 'lognormal':
                    self.root_ranges[node] = {
                        'dist': 'lognormal',
                        'mean': old_settings.get('mean', self.rng.uniform(0, 2)),
                        'sigma': old_settings.get('sigma', self.rng.uniform(0.1, 1.0))
                    }
                elif distribution_type == 'categorical':
                    self.root_ranges[node] = {
                        'dist': 'categorical',
                        'num_classes': old_settings.get('num_classes', self.rng.integers(2, 6))
                    }
                elif distribution_type == 'categorical_non_uniform':
                    num_classes = old_settings.get('num_classes', self.rng.integers(2, 6))
                    skew_factor = old_settings.get('skew_factor', self.rng.uniform(0.1, 0.9))
                    probabilities = self._generate_skewed_probabilities(num_classes, skew_factor)
                    self.root_ranges[node] = {
                        'dist': 'categorical_non_uniform',
                        'num_classes': num_classes,
                        'probabilities': probabilities
                    }
            else:
                # For nodes without existing settings, create new ones based on type
                if distribution_type == 'uniform':
                    low = self.rng.uniform(0, 10)
                    high = self.rng.uniform(low, low + 10)
                    self.root_ranges[node] = {'dist': 'uniform', 'low': low, 'high': high}
                elif distribution_type == 'normal':
                    mean = self.rng.uniform(0, 10)
                    std = self.rng.uniform(0.1, 5.0)
                    self.root_ranges[node] = {'dist': 'normal', 'mean': mean, 'std': std}
                elif distribution_type == 'exponential':
                    scale = self.rng.uniform(1, 10)
                    self.root_ranges[node] = {'dist': 'exponential', 'scale': scale}
                elif distribution_type == 'beta':
                    a = self.rng.uniform(0.5, 5.0)
                    b = self.rng.uniform(0.5, 5.0)
                    self.root_ranges[node] = {'dist': 'beta', 'a': a, 'b': b}
                elif distribution_type == 'truncated_normal':
                    mean = self.rng.uniform(0, 10)
                    std = self.rng.uniform(0.1, 5.0)
                    low = mean - 2 * std
                    high = mean + 2 * std

                    # Ensure valid parameters
                    if low >= high:
                        low = mean - 1.5 * std
                        high = mean + 1.5 * std

                    self.root_ranges[node] = {'dist': 'truncated_normal', 'mean': mean, 'std': std, 'low': low,
                                              'high': high}
                elif distribution_type == 'lognormal':
                    mean = self.rng.uniform(0, 2)
                    sigma = self.rng.uniform(0.1, 1.0)
                    self.root_ranges[node] = {'dist': 'lognormal', 'mean': mean, 'sigma': sigma}
                elif distribution_type == 'categorical':
                    num_classes = self.rng.integers(2, 6)
                    self.root_ranges[node] = {'dist': 'categorical', 'num_classes': num_classes}
                elif distribution_type == 'categorical_non_uniform':
                    num_classes = self.rng.integers(2, 6)
                    skew_factor = self.rng.uniform(0.1, 0.9)
                    probabilities = self._generate_skewed_probabilities(num_classes, skew_factor)
                    self.root_ranges[node] = {
                        'dist': 'categorical_non_uniform',
                        'num_classes': num_classes,
                        'probabilities': probabilities
                    }

    def _dirichlet_station_mapper(self,
                                  g: nx.DiGraph,
                                  alpha: float = 0.7,
                                  k_min: int = 3,
                                  k_max: int = 8,
                                  seed: int | None = None) -> dict[str, list[str]]:
        """
        Split the DAG nodes (kept in topological order) into K stations.
        K is drawn uniformly in [k_min, k_max]. alpha controls how unequal the sizes are.
        """
        rng = np.random.default_rng(seed)
        topo_nodes = list(nx.topological_sort(g))
        N = len(topo_nodes)

        k = int(rng.integers(k_min, min(k_max, N) + 1))
        shares = rng.dirichlet([alpha] * k)
        sizes = np.maximum(1, np.round(shares * N)).astype(int)
        sizes[-1] = N - sizes[:-1].sum()  # fix rounding drift

        mapper, idx = {}, 0
        for i, sz in enumerate(sizes, 1):
            mapper[f"Station{i}"] = [str(n) for n in topo_nodes[idx: idx + sz]]
            idx += sz
        return mapper

    def _set_categorical_roots(self, root_list, max_categories, probabilities=None, skew_factor=None):
        """
        Turn specified root nodes into categorical variables.
        
        Args:
            root_list: List of root nodes to make categorical
            max_categories: Maximum number of categories
            probabilities: Dict mapping node names to probability lists (optional)
            skew_factor: Float to generate skewed probabilities (optional)
        """
        if not hasattr(self, "categorical_root_nodes"):
            self.categorical_root_nodes = {}  # node → k

        for node in root_list:
            if node not in self.root_nodes:
                raise ValueError(f"'{node}' is not a root node, cannot set as categorical.")
            
            k = int(self.rng.integers(2, max_categories + 1))
            self.categorical_root_nodes[node] = k
            
            # Determine distribution type and probabilities
            if probabilities and node in probabilities:
                # Use provided probabilities
                probs = probabilities[node]
                if len(probs) != k:
                    raise ValueError(f"Probabilities for '{node}' must have length {k}, got {len(probs)}")
                if not np.isclose(sum(probs), 1.0, atol=1e-6):
                    raise ValueError(f"Probabilities for '{node}' must sum to 1.0, got {sum(probs)}")
                
                self.root_ranges[node] = {
                    "dist": "categorical_non_uniform", 
                    "num_classes": k,
                    "probabilities": probs
                }
            elif skew_factor is not None:
                # Generate skewed probabilities
                probs = self._generate_skewed_probabilities(k, skew_factor)
                self.root_ranges[node] = {
                    "dist": "categorical_non_uniform", 
                    "num_classes": k,
                    "probabilities": probs
                }
            else:
                # Uniform categorical (default behavior)
                self.root_ranges[node] = {"dist": "categorical", "num_classes": k}

    def _sample_random_nominals(self, categorical_percentage, max_categories):
        """
        Randomly pick a share of non-root nodes and convert them
        to nominal.  Parents / num_classes are chosen at random.
        """
        non_roots = [n for n in self.G.nodes if n not in self.root_nodes]
        choose_n = max(1, int(round(categorical_percentage * len(non_roots))))
        pick = list(self.rng.choice(non_roots, size=choose_n, replace=False))

        for node in pick:
            parents = list(self.G.predecessors(node))
            if not parents:  # safety: shouldn't happen for non-root
                continue
            parent_input = str(self.rng.choice(parents))
            k = int(self.rng.integers(2, max_categories + 1))
            self.nominal_nodes[node] = {"input": parent_input, "num_classes": k}

    def _generate_skewed_probabilities(self, num_classes, skew_factor):
        """
        Generate skewed probabilities for non-uniform categorical distribution.

        Args:
            num_classes (int): Number of categories
            skew_factor (float): Controls skewness (0.1 = very skewed, 0.9 = less skewed)

        Returns:
            list: List of probabilities that sum to 1
        """
        # Create probabilities that favor lower classes
        base_probs = np.array([(1 - skew_factor) ** i for i in range(num_classes)])
        # Normalize to sum to 1
        return (base_probs / base_probs.sum()).tolist()

    def _get_adaptive_coefficient_bounds(self, graph_size):
        """Adapt coefficient ranges based on graph size to prevent cascading explosions"""
        if graph_size <= 10:
            return {'coeff_range': (-2, 2), 'const_range': (-5, 5), 'power_range': (1.0, 2.0)}
        elif graph_size <= 25:
            return {'coeff_range': (-1.5, 1.5), 'const_range': (-3, 3), 'power_range': (1.0, 1.8)}
        else:
            return {'coeff_range': (-1, 1), 'const_range': (-2, 2), 'power_range': (1.0, 1.5)}

    def _is_categorical_node(self, node):
        """
        Check if a node is categorical.
        
        Args:
            node: Node name to check
            
        Returns:
            bool: True if node is categorical, False otherwise
        """
        # Check if node is in nominal_nodes (non-root categorical)
        if node in self.nominal_nodes:
            return True
            
        # Check if node is in categorical_root_nodes (root categorical)
        if hasattr(self, 'categorical_root_nodes') and node in self.categorical_root_nodes:
            return True
            
        # Check if node has categorical distribution in root_ranges
        if (node in self.root_ranges and 
            self.root_ranges[node].get('dist') in ['categorical', 'categorical_non_uniform']):
            return True
            
        return False

    def _detect_categorical_parents(self, parents):
        """
        Detect which parents are categorical.
        
        Args:
            parents: List of parent node names
            
        Returns:
            list: List of categorical parent names
        """
        categorical_parents = []
        for parent in parents:
            if self._is_categorical_node(parent):
                categorical_parents.append(parent)
        return categorical_parents

    def categorical_effect(self, categorical_var, effects_dict, default=0):
        """
        Create a piecewise function for categorical effects.
        
        This is a helper function for the dictionary interface that allows users
        to easily define categorical relationships without writing complex np.select syntax.
        
        Args:
            categorical_var: Name of the categorical variable
            effects_dict: Dict mapping category values to effect expressions
            default: Default value for unhandled categories
            
        Returns:
            str: String expression using np.select
            
        Example:
            categorical_effect('treatment', {
                0: '2.5 * age',
                1: '3.1 * age + 5.2',
                2: '1.8 * age + 8.7'
            })
            Returns: 'np.select([treatment == 0, treatment == 1, treatment == 2], 
                     [2.5 * age, 3.1 * age + 5.2, 1.8 * age + 8.7], default=0)'
        """
        if not effects_dict:
            raise ValueError("effects_dict cannot be empty")
            
        # Sort categories for consistent ordering
        categories = sorted(effects_dict.keys())
        
        # Build conditions and choices
        conditions = [f"{categorical_var} == {cat}" for cat in categories]
        choices = [effects_dict[cat] for cat in categories]
        
        # Create np.select expression
        conditions_str = "[" + ", ".join(conditions) + "]"
        choices_str = "[" + ", ".join(choices) + "]"
        
        return f"np.select({conditions_str}, {choices_str}, default={default})"

    def categorical_where(self, categorical_var, true_effect, false_effect, condition_value=0):
        """
        Create a binary categorical effect using np.where.
        
        This is a simpler helper for binary categorical variables.
        
        Args:
            categorical_var: Name of the categorical variable
            true_effect: Effect when categorical_var equals condition_value
            false_effect: Effect when categorical_var doesn't equal condition_value
            condition_value: Value to check against (default: 0)
            
        Returns:
            str: String expression using np.where
            
        Example:
            categorical_where('treatment', '2.5 * age', '3.1 * age + 5.2', 0)
            Returns: 'np.where(treatment == 0, 2.5 * age, 3.1 * age + 5.2)'
        """
        return f"np.where({categorical_var} == {condition_value}, {true_effect}, {false_effect})"

    def set_skewed_categorical_roots(self, root_list, max_categories, skew_factor=0.7):
        """
        Convenience method to create skewed categorical root distributions.
        
        Args:
            root_list: List of root nodes to make categorical
            max_categories: Maximum number of categories
            skew_factor: Controls skewness (0.1 = very skewed, 1.0 = uniform)
            
        Example:
            cdg.set_skewed_categorical_roots(['education'], max_categories=4, skew_factor=0.3)
            # Creates education with probabilities like [0.1, 0.2, 0.3, 0.4]
        """
        self._set_categorical_roots(root_list, max_categories, skew_factor=skew_factor)

    def set_custom_categorical_roots(self, root_probabilities):
        """
        Set root categorical variables with custom probability distributions.
        
        Args:
            root_probabilities: Dict mapping node names to probability lists
            
        Example:
            cdg.set_custom_categorical_roots({
                'education': [0.1, 0.5, 0.3, 0.1],  # 4 categories
                'severity': [0.6, 0.25, 0.1, 0.05]   # 4 categories
            })
        """
        for node, probs in root_probabilities.items():
            if node not in self.root_nodes:
                raise ValueError(f"'{node}' is not a root node, cannot set as categorical.")
            
            k = len(probs)
            if not np.isclose(sum(probs), 1.0, atol=1e-6):
                raise ValueError(f"Probabilities for '{node}' must sum to 1.0, got {sum(probs)}")
            
            if not hasattr(self, "categorical_root_nodes"):
                self.categorical_root_nodes = {}
            
            self.categorical_root_nodes[node] = k
            self.root_ranges[node] = {
                "dist": "categorical_non_uniform", 
                "num_classes": k,
                "probabilities": probs
            }
