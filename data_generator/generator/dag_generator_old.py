import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scdg import CausalDataGenerator

from .utils import save_dataset, get_equation_type
from .structural_patterns import add_structural_variations
from .temporal_utils import generate_temporal_order_from_stations

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif 'Tensor' in str(type(obj)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def save_graph_visualization(G, dataset_dir, base_name, metadata):
    """Save graph visualization as PNG"""
    try:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Get temporal order for node positioning
        temporal_order = metadata.get('temporal_order', list(G.nodes()))
        
        # Create hierarchical layout based on temporal order
        pos = {}
        station_groups = {}
        
        for i, node in enumerate(temporal_order):
            station = node.split('_')[0] if '_' in node else 'Unknown'
            station_groups.setdefault(station, []).append(node)
            
        # Position nodes by station
        y_offset = 0
        for station in sorted(station_groups.keys()):
            nodes = station_groups[station]
            x_positions = np.linspace(0, 1, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y_offset)
            y_offset -= 1
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='red', 
                              arrows=True, arrowsize=20, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Set title and labels
        plt.title(f"Causal Graph: {base_name}\n"
                 f"Nodes: {metadata['num_nodes']}, "
                 f"Root Nodes: {metadata['root_nodes']} ({metadata.get('root_percentage', 0):.1f}%), "
                 f"Edges: {metadata['edges']}", 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel("Temporal Order", fontsize=12)
        plt.ylabel("Station Groups", fontsize=12)
        
        # Set axis limits
        plt.xlim(-0.1, 1.1)
        plt.ylim(y_offset - 0.5, 0.5)
        
        # Add grid for better visualization
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        graph_path = os.path.join(dataset_dir, f"{base_name}_graph.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close the figure to free memory
        
        print(f"  📊 Graph visualization saved: {graph_path}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not save graph visualization: {e}")
        plt.close()  # Ensure figure is closed even if error occurs

def generate_meta_dataset(total_datasets=10, output_dir="causal_meta_dataset"):
    for i in range(total_datasets):
        try:
            # Define your graph parameters (num_nodes, root_nodes, edges, etc.)
            num_nodes = random.randint(3, 5)
            
            # Calculate root nodes based on 10-30% constraint with fallback
            min_root_percentage = 0.10  # 10%
            max_root_percentage = 0.30  # 30%
            
            min_root_nodes = max(1, int(num_nodes * min_root_percentage))
            max_root_nodes = max(min_root_nodes, int(num_nodes * max_root_percentage))
            
            # Ensure we have at least one root node and at most num_nodes - 1
            max_root_nodes = min(max_root_nodes, num_nodes - 1)
            
            # Generate root nodes within the calculated range
            root_nodes = random.randint(min_root_nodes, max_root_nodes)
            
            edges = random.randint(num_nodes * 2, num_nodes * 3)
            equation_type = get_equation_type(random.choice(["low"]))

            cdg = CausalDataGenerator(num_samples=500, seed=42)
            G, roots = cdg.generate_random_graph(num_nodes, root_nodes, edges)

            # Add structure
            G = add_structural_variations(G, pattern="chain")
            cdg.G = G

            # Use causalAssembly's convert_to_manufacturing for proper station assignment
            pline, cell_mapper = cdg.convert_to_manufacturing(
                strategy="dirichlet",
                alpha=0.7,
                k_min=3,
                k_max=3,  # Fixed to 3 stations for consistency
                seed=42,
                return_mapper=True
            )
            
            # Extract station mapping from the manufacturing graph
            station_map = {}
            for station, nodes in cell_mapper.items():
                for node in nodes:
                    station_map[node] = f"{station}_{node}"
            
            # Update the graph with the manufacturing version
            G = pline.to_nx()
            cdg.G = G

            # Generate data
            df = cdg.generate_data_pipeline(
                total_nodes=num_nodes,
                root_nodes=root_nodes,
                edges=edges,
                equation_type=equation_type
            )
            df.columns = list(G.nodes())

            # Create temporal order
            temporal_order = generate_temporal_order_from_stations(station_map)
            adj_matrix = nx.to_numpy_array(G, nodelist=temporal_order, dtype=int)

            # Build metadata
            root_percentage = (root_nodes / num_nodes) * 100
            metadata = {
                "temporal_order": temporal_order,
                "num_nodes": num_nodes,
                "root_nodes": root_nodes,
                "root_percentage": root_percentage,
                "edges": edges,
                "equation_type": equation_type
            }

            base_name = f"dataset_{i}"
            dataset_dir = os.path.join(output_dir, base_name)

            save_dataset(df, adj_matrix, metadata, dataset_dir, base_name)
            
            # Save graph visualization
            save_graph_visualization(G, dataset_dir, base_name, metadata)
            
            # Calculate and display root node percentage
            root_percentage = (root_nodes / num_nodes) * 100
            print(f"[{i+1}/{total_datasets}] Saved {base_name} - Root nodes: {root_nodes}/{num_nodes} ({root_percentage:.1f}%)")

        except Exception as e:
            print(f"Skipping dataset {i+1} due to error: {e}")
