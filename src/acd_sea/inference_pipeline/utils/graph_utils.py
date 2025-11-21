import networkx as nx
import numpy as np

def build_true_graph(adj_matrix, column_names):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.relabel_nodes(G, dict(enumerate(column_names)))
