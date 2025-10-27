"""
CSuite-style DAG Generator

This module generates DAGs using configurations inspired by Microsoft's CSuite
benchmark datasets. It supports 2-5 node graphs with various structural patterns
including chains, colliders, backdoors, and mixed confounding scenarios.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from scdg import CausalDataGenerator

from .utils import save_dataset, get_equation_type
from .temporal_utils import assign_mock_stations


class CSuiteConfig:
    """Configuration class for CSuite-style datasets."""
    
    def __init__(self, 
                 pattern: str,
                 num_nodes: int,
                 num_samples: int = 1000,
                 seed: int = 42,
                 **kwargs):
        """
        Initialize CSuite configuration.
        
        Args:
            pattern: CSuite pattern type ('chain', 'collider', 'backdoor', 'mixed_confounding', 'weak_arrow', 'large_backdoor')
            num_nodes: Number of nodes (2-5)
            num_samples: Number of samples to generate
            seed: Random seed
            **kwargs: Additional pattern-specific parameters
        """
        self.pattern = pattern
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.seed = seed
        
        # Validate node count
        if not 2 <= num_nodes <= 5:
            raise ValueError("num_nodes must be between 2 and 5")
        
        # Pattern-specific parameters
        self.kwargs = kwargs
        
        # Generate the specific configuration
        self._generate_config()
    
    def _generate_config(self):
        """Generate specific configuration based on pattern."""
        if self.pattern == "chain":
            self._config_chain()
        elif self.pattern == "collider":
            self._config_collider()
        elif self.pattern == "backdoor":
            self._config_backdoor()
        elif self.pattern == "mixed_confounding":
            self._config_mixed_confounding()
        elif self.pattern == "weak_arrow":
            self._config_weak_arrow()
        elif self.pattern == "large_backdoor":
            self._config_large_backdoor()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
    
    def _config_chain(self):
        """Chain pattern: X0 -> X1 -> X2 -> ... -> Xn"""
        self.graph_structure = {
            'edges': [(str(i), str(i+1)) for i in range(self.num_nodes - 1)],
            'root_nodes': ["0"],
            'leaf_nodes': [str(self.num_nodes - 1)]
        }
        self.variable_types = {str(i): 'continuous' for i in range(self.num_nodes)}
        self.equation_type = 'linear'
        
    def _config_collider(self):
        """Collider pattern: X0 -> X1 <- X2 (with additional nodes if num_nodes > 3)"""
        if self.num_nodes < 3:
            raise ValueError("Collider pattern requires at least 3 nodes")
        
        edges = [("0", "1"), ("2", "1")]  # Basic collider
        if self.num_nodes > 3:
            # Add additional nodes as children of the collider
            for i in range(3, self.num_nodes):
                edges.append(("1", str(i)))
        
        self.graph_structure = {
            'edges': edges,
            'root_nodes': ["0", "2"] if self.num_nodes == 3 else ["0", "2"] + [str(i) for i in range(3, self.num_nodes)],
            'leaf_nodes': ["1"] if self.num_nodes == 3 else [str(i) for i in range(3, self.num_nodes)]
        }
        self.variable_types = {str(i): 'continuous' for i in range(self.num_nodes)}
        self.equation_type = 'linear'
    
    def _config_backdoor(self):
        """Backdoor pattern: X0 -> X1 <- X2 -> X1 (confounding)"""
        if self.num_nodes < 3:
            raise ValueError("Backdoor pattern requires at least 3 nodes")
        
        edges = [("0", "1"), ("2", "1")]  # Basic backdoor
        if self.num_nodes > 3:
            # Add confounders
            for i in range(3, self.num_nodes):
                edges.extend([(str(i), "0"), (str(i), "2")])  # Confound both treatment and confounder
        
        self.graph_structure = {
            'edges': edges,
            'root_nodes': ["2"] if self.num_nodes == 3 else ["2"] + [str(i) for i in range(3, self.num_nodes)],
            'leaf_nodes': ["1"]
        }
        self.variable_types = {str(i): 'continuous' for i in range(self.num_nodes)}
        self.equation_type = 'linear'
    
    def _config_mixed_confounding(self):
        """Mixed confounding pattern with discrete and continuous variables."""
        if self.num_nodes < 4:
            raise ValueError("Mixed confounding pattern requires at least 4 nodes")
        
        # Treatment (X0), Outcome (X1), Confounders (X2+)
        edges = [("0", "1")]  # Treatment -> Outcome
        for i in range(2, self.num_nodes):
            edges.extend([(str(i), "0"), (str(i), "1")])  # Confounders affect both treatment and outcome
        
        self.graph_structure = {
            'edges': edges,
            'root_nodes': [str(i) for i in range(2, self.num_nodes)],
            'leaf_nodes': ["1"]
        }
        
        # Mix of discrete and continuous variables
        self.variable_types = {}
        for i in range(self.num_nodes):
            node_str = str(i)
            if i in [0, 1]:  # Treatment and outcome
                self.variable_types[node_str] = 'discrete' if i == 0 else 'continuous'
            else:  # Confounders
                self.variable_types[node_str] = 'discrete' if i % 2 == 0 else 'continuous'
        
        self.equation_type = 'non_linear'
    
    def _config_weak_arrow(self):
        """Weak arrow pattern with weak causal effects."""
        if self.num_nodes < 3:
            raise ValueError("Weak arrow pattern requires at least 3 nodes")
        
        # Create a chain with weak effects
        edges = [(str(i), str(i+1)) for i in range(self.num_nodes - 1)]
        if self.num_nodes > 3:
            # Add some weak cross-connections
            edges.append(("0", str(self.num_nodes - 1)))  # Weak direct effect
        
        self.graph_structure = {
            'edges': edges,
            'root_nodes': ["0"],
            'leaf_nodes': [str(self.num_nodes - 1)]
        }
        self.variable_types = {str(i): 'continuous' for i in range(self.num_nodes)}
        self.equation_type = 'linear'
        self.weak_effects = True  # Flag for weak causal effects
    
    def _config_large_backdoor(self):
        """Large backdoor pattern with multiple confounders."""
        if self.num_nodes < 4:
            raise ValueError("Large backdoor pattern requires at least 4 nodes")
        
        # Treatment (X0), Outcome (X1), Multiple confounders (X2+)
        edges = [("0", "1")]
        for i in range(2, self.num_nodes):
            edges.extend([(str(i), "0"), (str(i), "1")])  # Each confounder affects both treatment and outcome
        
        self.graph_structure = {
            'edges': edges,
            'root_nodes': [str(i) for i in range(2, self.num_nodes)],
            'leaf_nodes': ["1"]
        }
        self.variable_types = {str(i): 'continuous' for i in range(self.num_nodes)}
        self.equation_type = 'linear'


class CSuiteDAGGenerator:
    """DAG generator using CSuite-style configurations."""
    
    def __init__(self, config: CSuiteConfig):
        """Initialize the generator with a CSuite configuration."""
        self.config = config
        self.random_state = np.random.RandomState(config.seed)
        random.seed(config.seed)
    
    def generate_dag(self) -> Tuple[nx.DiGraph, Dict]:
        """Generate a DAG based on the CSuite configuration."""
        G = nx.DiGraph()
        
        # Add nodes (already using string IDs from config)
        for i in range(self.config.num_nodes):
            G.add_node(str(i))
        
        # Add edges (already using string IDs from config)
        for source, target in self.config.graph_structure['edges']:
            G.add_edge(source, target)
        
        # Generate metadata
        metadata = {
            'pattern': self.config.pattern,
            'num_nodes': self.config.num_nodes,
            'num_edges': len(self.config.graph_structure['edges']),
            'root_nodes': self.config.graph_structure['root_nodes'],
            'leaf_nodes': self.config.graph_structure['leaf_nodes'],
            'variable_types': self.config.variable_types,
            'equation_type': self.config.equation_type,
            'seed': self.config.seed
        }
        
        return G, metadata
    
    def generate_data(self, G: nx.DiGraph) -> Tuple[pd.DataFrame, Dict]:
        """Generate data using SCDG based on the DAG structure."""
        # Create SCDG instance
        cdg = CausalDataGenerator(
            num_samples=self.config.num_samples,
            seed=self.config.seed
        )
        
        # Set the graph structure (with string node IDs)
        cdg.G = G.copy()
        cdg.root_nodes = set(self.config.graph_structure['root_nodes'])
        
        # Prepare distributions for different variable types
        root_distributions = {}
        for node in self.config.graph_structure['root_nodes']:
            var_type = self.config.variable_types[node]
            if var_type == 'discrete':
                root_distributions[node] = {
                    'dist': 'categorical',
                    'num_classes': 3 if node == "0" else 2  # Treatment has 3 classes, others have 2
                }
            else:  # continuous
                root_distributions[node] = {
                    'dist': 'normal',
                    'mean': 0.0,
                    'std': 1.0
                }
        
        # Use direct SCDG API to preserve graph structure
        # Step 1: Set root distributions
        cdg.set_root_distributions(root_distributions)
        
        # Step 2: Assign equations to non-root nodes
        cdg.assign_equations_to_graph_nodes(equation_type=self.config.equation_type)
        
        # Step 3: Generate data using the preset graph (now preserves CSuite structure!)
        df = cdg.generate_data()
        
        # Apply station-wise temporal ordering
        topo_nodes = list(nx.topological_sort(G))
        raw_assignment = assign_mock_stations(topo_nodes, num_stations=3, graph=G)
        station_map = {node: raw_assignment[node].split('_')[0] for node in raw_assignment}
        ordered_stations = sorted(set(station_map.values()), key=lambda s: int(s.replace("Station", "")))
        station_to_nodes = {s: [] for s in ordered_stations}
        for n in topo_nodes:
            station_to_nodes[station_map[n]].append(n)
        station_blocks = [station_to_nodes[s] for s in ordered_stations]
        temporal_order = [n for block in station_blocks for n in block]
        
        # Reorder dataframe columns to match temporal order
        df = df[temporal_order]
        
        # Update metadata with station information
        metadata = {
            'temporal_order': temporal_order,
            'station_blocks': station_blocks,
            'station_names': ordered_stations,
            'station_map': station_map,
            'pattern': self.config.pattern,
            'num_nodes': self.config.num_nodes,
            'num_edges': len(self.config.graph_structure['edges']),
            'root_nodes': self.config.graph_structure['root_nodes'],
            'leaf_nodes': self.config.graph_structure['leaf_nodes'],
            'variable_types': self.config.variable_types,
            'equation_type': self.config.equation_type,
            'seed': self.config.seed,
            'num_samples': self.config.num_samples
        }
        
        return df, metadata
    
    def generate_dataset(self, output_dir: str, base_name: str) -> Dict:
        """Generate a complete dataset and save it."""
        # Generate DAG
        G, dag_metadata = self.generate_dag()
        
        # Generate data
        df, data_metadata = self.generate_data(G)
        
        # Create adjacency matrix
        temporal_order = data_metadata['temporal_order']
        adj_matrix = nx.to_numpy_array(G, nodelist=temporal_order, dtype=int)
        
        # Combine metadata
        metadata = {**dag_metadata, **data_metadata}
        
        # Save dataset
        dataset_dir = os.path.join(output_dir, base_name)
        save_dataset(df, adj_matrix, metadata, dataset_dir, base_name)
        
        return {
            'dataframe': df,
            'adjacency_matrix': adj_matrix,
            'graph': G,
            'metadata': metadata,
            'dataset_dir': dataset_dir
        }


def generate_csuite_meta_dataset(
    patterns: List[str] = None,
    num_nodes_range: Tuple[int, int] = (2, 5),
    num_samples: int = 1000,
    output_dir: str = "csuite_benchmark/datasets",
    seed: int = 42
) -> None:
    """
    Generate a meta dataset using CSuite-style configurations.
    
    Args:
        patterns: List of CSuite patterns to use. If None, uses all available patterns.
        num_nodes_range: Range of number of nodes (min, max)
        num_samples: Number of samples per dataset
        output_dir: Output directory
        seed: Random seed
    """
    if patterns is None:
        patterns = ['chain', 'collider', 'backdoor', 'mixed_confounding', 'weak_arrow', 'large_backdoor']
    
    # Create pattern-specific directories
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_count = 0
    
    for pattern in patterns:
        logging.info(f"Generating datasets for pattern: {pattern}")
        
        # Create pattern-specific directory
        pattern_dir = os.path.join(output_dir, pattern)
        os.makedirs(pattern_dir, exist_ok=True)
        
        # Generate datasets for different node counts
        for num_nodes in range(num_nodes_range[0], num_nodes_range[1] + 1):
            try:
                # Skip patterns that don't support the node count
                if pattern in ['collider', 'backdoor'] and num_nodes < 3:
                    continue
                if pattern in ['mixed_confounding', 'large_backdoor'] and num_nodes < 4:
                    continue
                
                # Create configuration
                config = CSuiteConfig(
                    pattern=pattern,
                    num_nodes=num_nodes,
                    num_samples=num_samples,
                    seed=seed + dataset_count
                )
                
                # Generate dataset
                generator = CSuiteDAGGenerator(config)
                base_name = f"csuite_{pattern}_{num_nodes}nodes_{dataset_count:03d}"
                
                result = generator.generate_dataset(pattern_dir, base_name)
                
                logging.info(f"Generated: {base_name}")
                logging.debug(f"Pattern: {pattern}, Nodes: {num_nodes}, Samples: {num_samples}")
                logging.debug(f"Root nodes: {result['metadata']['root_nodes']}")
                logging.debug(f"Variable types: {result['metadata']['variable_types']}")
                
                dataset_count += 1
                
            except Exception as e:
                logging.error(f"Error generating {pattern} with {num_nodes} nodes: {e}")
                continue
    
    logging.info(f"Generated {dataset_count} CSuite-style datasets successfully")


def test_csuite_generator():
    """Test function to verify CSuite generator works correctly."""
    logging.info("Testing CSuite DAG Generator...")
    
    # Test different patterns
    patterns_to_test = ['chain', 'collider', 'backdoor']
    
    for pattern in patterns_to_test:
        logging.info(f"Testing pattern: {pattern}")
        
        try:
            config = CSuiteConfig(
                pattern=pattern,
                num_nodes=4,
                num_samples=500,
                seed=123
            )
            
            generator = CSuiteDAGGenerator(config)
            G, metadata = generator.generate_dag()
            df, data_metadata = generator.generate_data(G)
            
            logging.debug(f"Graph nodes: {list(G.nodes())}")
            logging.debug(f"Graph edges: {list(G.edges())}")
            logging.debug(f"Data shape: {df.shape}")
            logging.debug(f"Root nodes: {metadata['root_nodes']}")
            logging.debug(f"Variable types: {metadata['variable_types']}")
            
        except Exception as e:
            logging.error(f"Error testing {pattern}: {e}")
    
    logging.info("CSuite generator test completed")


if __name__ == "__main__":
    # Test the generator
    test_csuite_generator()
    
    # Generate a small meta dataset
    generate_csuite_meta_dataset(
        patterns=['chain', 'collider'],
        num_nodes_range=(3, 4),
        num_samples=1000,
        output_dir="test_csuite_datasets",
        seed=42
    )
