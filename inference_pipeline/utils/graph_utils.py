import networkx as nx
import numpy as np
from typing import List

def prune_temporal_violations(graph: nx.DiGraph, temporal_order: List[str]) -> nx.DiGraph:
    """
    Remove edges from nodes that violate the station-based temporal order.
    """
    order_index = {node: i for i, node in enumerate(temporal_order)}
    pruned_graph = graph.copy()
    
    for u, v in list(pruned_graph.edges()):
        if u not in order_index or v not in order_index:
            pruned_graph.remove_edge(u, v)  # skip unknowns
        elif order_index[u] >= order_index[v]:
            pruned_graph.remove_edge(u, v)
    return pruned_graph


def build_true_graph(adj_matrix, column_names):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.relabel_nodes(G, dict(enumerate(column_names)))
