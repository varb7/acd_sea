import random
import networkx as nx
from typing import List, Dict

def assign_mock_stations(nodes: List[int], num_stations: int = 3, graph: nx.DiGraph = None, seed: int = None) -> Dict[int, str]:
    """
    Assign stations to nodes using causalAssembly's dirichlet station mapper approach.
    
    This creates a proper temporal hierarchy based on topological ordering,
    similar to the convert_to_manufacturing method in CausalDataGenerator.
    
    Args:
        nodes: List of node identifiers
        num_stations: Number of stations to create
        graph: NetworkX DiGraph for topological ordering
        seed: Random seed for reproducibility (each dataset should use its own seed)
    
    Returns:
        Dictionary mapping node to "StationX_node" format
    """
    if graph is not None:
        # Use the same approach as causalAssembly's _dirichlet_station_mapper
        import numpy as np
        
        # Get nodes in topological order (this is key!)
        topo_nodes = list(nx.topological_sort(graph))
        N = len(topo_nodes)
        
        # Use dirichlet distribution to create station sizes
        # This ensures stations are created based on causal flow
        alpha = 0.7  # concentration parameter (0.7 gives moderate variation)
        k = min(num_stations, N)  # number of stations
        
        # Generate station sizes using dirichlet distribution
        # Use provided seed for dataset-specific variation
        rng = np.random.default_rng(seed)
        shares = rng.dirichlet([alpha] * k)
        sizes = np.maximum(1, np.round(shares * N)).astype(int)
        sizes[-1] = N - sizes[:-1].sum()  # fix rounding drift
        
        # Create station assignment
        station_assignment = {}
        idx = 0
        for i, sz in enumerate(sizes, 1):
            station_name = f"Station{i}"
            for j in range(sz):
                if idx < len(topo_nodes):
                    node = topo_nodes[idx]
                    station_assignment[node] = f"{station_name}_{node}"
                    idx += 1
        
        return station_assignment
    
    else:
        # Fallback: distribute nodes evenly across stations
        sorted_nodes = sorted(nodes)
        station_assignment = {}
        stations = [f"Station{i+1}" for i in range(num_stations)]
        
        for i, node in enumerate(sorted_nodes):
            station_index = min(i, num_stations - 1)
            station = stations[station_index]
            station_assignment[node] = f"{station}_{node}"
        
        return station_assignment

def relabel_graph_with_stations(G: nx.DiGraph, station_assignment: Dict[int, str]) -> nx.DiGraph:
    return nx.relabel_nodes(G, station_assignment)

def generate_temporal_order_from_stations(station_assignment: Dict[str, str]) -> List[str]:
    """
    Generate temporal order from station assignment.
    Works with both our custom assignment and causalAssembly's convert_to_manufacturing output.
    """
    station_to_nodes = {}
    for node_label in station_assignment.values():
        station = node_label.split('_')[0]
        station_to_nodes.setdefault(station, []).append(node_label)

    # Sort stations by their number (Station1, Station2, Station3, etc.)
    ordered_stations = sorted(station_to_nodes.keys(), key=lambda s: int(s.replace("Station", "")))
    temporal_order = []
    for station in ordered_stations:
        temporal_order.extend(station_to_nodes[station])
    return temporal_order
