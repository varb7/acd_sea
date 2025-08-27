import networkx as nx
import random
from itertools import product

def add_structural_variations(G: nx.DiGraph, pattern: str) -> nx.DiGraph:
    nodes = list(G.nodes())
    if len(nodes) < 3:
        return G

    num_patterns = max(1, min(5, len(nodes) // 5))
    if pattern == "default":
        return G

    for _ in range(num_patterns):
        try:
            selected = random.sample(nodes, 3)
            for u, v in product(selected, selected):
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

            if pattern == "chain":
                G.add_edge(selected[0], selected[1])
                G.add_edge(selected[1], selected[2])
            elif pattern == "confounder":
                G.add_edge(selected[0], selected[1])
                G.add_edge(selected[0], selected[2])
            elif pattern == "collider":
                G.add_edge(selected[0], selected[1])
                G.add_edge(selected[2], selected[1])
            elif pattern == "mediator":
                G.add_edge(selected[0], selected[1])
                G.add_edge(selected[1], selected[2])
                G.add_edge(selected[0], selected[2])
        except:
            continue

    return G
