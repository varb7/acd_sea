"""
Test script to verify root node fix for collider, backdoor, and large_backdoor patterns.
"""
import numpy as np
import pickle
from pathlib import Path
import networkx as nx

def test_pattern_root_computation():
    """Test that we correctly identify root nodes for problematic patterns."""
    
    # Test collider pattern (n=5)
    # Expected: edges (0->1), (2->1), (1->3), (1->4)
    # True roots: [0, 2] (nodes 'a' and 'c')
    # Pattern template incorrectly says: [0, 2, 3, 4] (nodes 'a', 'c', 'd', 'e')
    
    print("=" * 70)
    print("Testing Pattern Root Node Computation")
    print("=" * 70)
    
    # Simulate collider pattern
    G_collider = nx.DiGraph()
    nodes = ['a', 'c', 'b', 'd', 'e']
    G_collider.add_nodes_from(nodes)
    G_collider.add_edges_from([('a', 'b'), ('c', 'b'), ('b', 'd'), ('b', 'e')])
    
    # Create adjacency matrix in temporal order
    adj_collider = nx.to_numpy_array(G_collider, nodelist=nodes, dtype=int)
    
    print("\nCollider Pattern (n=5):")
    print(f"Nodes (temporal order): {nodes}")
    print(f"Edges: {list(G_collider.edges())}")
    print(f"True roots (zero indegree): {[n for n in nodes if G_collider.in_degree(n) == 0]}")
    print(f"Adjacency matrix:\n{adj_collider}")
    
    # Compute roots from adjacency
    incoming = np.any(adj_collider != 0, axis=0)
    computed_roots = [nodes[i] for i, has_parent in enumerate(incoming) if not has_parent]
    print(f"Computed roots from adj: {computed_roots}")
    
    # Test backdoor pattern (n=5)
    # Expected: edges (0->1), (2->1), (3->0), (3->2), (4->0), (4->2)
    # True roots: [3, 4] (nodes 'd' and 'e')
    # Pattern template incorrectly says: [2, 3, 4] (nodes 'c', 'd', 'e')
    
    print("\n" + "-" * 70)
    print("Backdoor Pattern (n=5):")
    nodes_bd = ['d', 'e', 'a', 'c', 'b']
    G_backdoor = nx.DiGraph()
    G_backdoor.add_nodes_from(nodes_bd)
    G_backdoor.add_edges_from([('a', 'b'), ('c', 'b'), ('d', 'a'), ('d', 'c'), ('e', 'a'), ('e', 'c')])
    
    adj_backdoor = nx.to_numpy_array(G_backdoor, nodelist=nodes_bd, dtype=int)
    
    print(f"Nodes (temporal order): {nodes_bd}")
    print(f"Edges: {list(G_backdoor.edges())}")
    print(f"True roots (zero indegree): {[n for n in nodes_bd if G_backdoor.in_degree(n) == 0]}")
    print(f"Adjacency matrix:\n{adj_backdoor}")
    
    incoming_bd = np.any(adj_backdoor != 0, axis=0)
    computed_roots_bd = [nodes_bd[i] for i, has_parent in enumerate(incoming_bd) if not has_parent]
    print(f"Computed roots from adj: {computed_roots_bd}")
    
    # Test large_backdoor pattern (n=5)
    # Expected: edges (0->1), (2->0), (2->1), (3->0), (3->1), (4->0), (4->1)
    # True roots: [2, 3, 4] (nodes 'c', 'd', 'e')
    # Pattern template says: [2, 3, 4] - this one is CORRECT!
    
    print("\n" + "-" * 70)
    print("Large Backdoor Pattern (n=5):")
    nodes_lbd = ['c', 'd', 'e', 'a', 'b']
    G_large_backdoor = nx.DiGraph()
    G_large_backdoor.add_nodes_from(nodes_lbd)
    G_large_backdoor.add_edges_from([('a', 'b'), ('c', 'a'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('e', 'a'), ('e', 'b')])
    
    adj_large_backdoor = nx.to_numpy_array(G_large_backdoor, nodelist=nodes_lbd, dtype=int)
    
    print(f"Nodes (temporal order): {nodes_lbd}")
    print(f"Edges: {list(G_large_backdoor.edges())}")
    print(f"True roots (zero indegree): {[n for n in nodes_lbd if G_large_backdoor.in_degree(n) == 0]}")
    print(f"Adjacency matrix:\n{adj_large_backdoor}")
    
    incoming_lbd = np.any(adj_large_backdoor != 0, axis=0)
    computed_roots_lbd = [nodes_lbd[i] for i, has_parent in enumerate(incoming_lbd) if not has_parent]
    print(f"Computed roots from adj: {computed_roots_lbd}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Collider: Expected ['a', 'c'], Got {computed_roots} - {'✓' if set(computed_roots) == {'a', 'c'} else '✗'}")
    print(f"Backdoor: Expected ['d', 'e'], Got {computed_roots_bd} - {'✓' if set(computed_roots_bd) == {'d', 'e'} else '✗'}")
    print(f"Large Backdoor: Expected ['c', 'd', 'e'], Got {computed_roots_lbd} - {'✓' if set(computed_roots_lbd) == {'c', 'd', 'e'} else '✗'}")


def test_real_dataset():
    """Test with a real dataset from the phase3 directory."""
    print("\n" + "=" * 70)
    print("Testing with Real Dataset")
    print("=" * 70)
    
    dataset_path = Path("csuite_grid_datasets_phase3/csuite_collider_5n_p3_non_linear_mixed_exponential_highvariance_normal_highnoise_500_0042")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Load metadata
    meta_files = list(dataset_path.glob("*_meta.pkl"))
    if not meta_files:
        print(f"No metadata file found in {dataset_path}")
        return
    
    with open(meta_files[0], 'rb') as f:
        meta = pickle.load(f)
    
    # Load adjacency
    adj = np.load(dataset_path / "graph.npy")
    temporal_order = [str(n) for n in meta.get('temporal_order', [])]
    
    print(f"\nDataset: {dataset_path.name}")
    print(f"Pattern: {meta.get('pattern')}")
    print(f"Stored root_nodes: {meta.get('root_nodes')}")
    print(f"Temporal order: {temporal_order}")
    print(f"Adjacency shape: {adj.shape}")
    
    # Compute roots from adjacency
    incoming = np.any(adj != 0, axis=0)
    computed_roots = [temporal_order[i] for i, has_parent in enumerate(incoming) if not has_parent]
    
    print(f"Computed roots from adj: {computed_roots}")
    print(f"Match: {'✓' if set(computed_roots) == set([str(n) for n in meta.get('root_nodes', [])]) else '✗ MISMATCH!'}")
    
    # Show adjacency matrix
    print(f"\nAdjacency matrix (first few rows/cols):")
    print(adj[:5, :5])


if __name__ == "__main__":
    test_pattern_root_computation()
    test_real_dataset()

