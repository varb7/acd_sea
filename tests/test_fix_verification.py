"""
End-to-end test to verify the root node fix works in the actual pipeline.
"""
import sys
from pathlib import Path
import numpy as np
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inference_pipeline.run_experiments import (
    _normalize_columns,
    _compute_root_nodes_from_adj,
    _sync_root_nodes_with_graph,
    load_metadata
)

def test_fix_with_real_dataset():
    """Test the fix with a real problematic dataset."""
    print("=" * 70)
    print("Testing Root Node Fix with Real Dataset")
    print("=" * 70)
    
    # Use a known problematic collider dataset
    dataset_path = Path("csuite_grid_datasets_phase3/csuite_collider_5n_p3_non_linear_mixed_exponential_highvariance_normal_highnoise_500_0042")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Trying alternative path...")
        # Try to find any collider dataset
        phase3_dir = Path("csuite_grid_datasets_phase3")
        if phase3_dir.exists():
            collider_dirs = [d for d in phase3_dir.iterdir() if d.is_dir() and "collider" in d.name]
            if collider_dirs:
                dataset_path = collider_dirs[0]
                print(f"Using: {dataset_path}")
            else:
                print("No collider datasets found")
                return
        else:
            print("Phase3 directory not found")
            return
    
    # Load metadata
    meta = load_metadata(dataset_path)
    print(f"\nDataset: {dataset_path.name}")
    print(f"Pattern: {meta.get('pattern')}")
    print(f"Original stored root_nodes: {meta.get('root_nodes')}")
    
    # Load adjacency
    adj = np.load(dataset_path / "graph.npy")
    print(f"Adjacency shape: {adj.shape}")
    
    # Get temporal order and normalize
    temporal_order_raw = meta.get('temporal_order', [])
    cols = _normalize_columns(temporal_order_raw)
    print(f"Temporal order (normalized): {cols}")
    
    # Show adjacency matrix
    print(f"\nAdjacency matrix:")
    print(adj)
    
    # Compute roots from adjacency
    computed_roots = _compute_root_nodes_from_adj(adj, cols)
    print(f"\nComputed roots from adjacency: {computed_roots}")
    
    # Apply the fix
    meta_fixed = _sync_root_nodes_with_graph(meta, adj, cols, dataset_path.name)
    
    print(f"\nAfter fix:")
    print(f"  root_nodes: {meta_fixed.get('root_nodes')}")
    print(f"  _root_nodes_original: {meta_fixed.get('_root_nodes_original', 'N/A')}")
    
    # Verify the fix worked
    if set(meta_fixed.get('root_nodes', [])) == set(computed_roots):
        print("\n✓ Fix verified: root_nodes now matches computed roots")
    else:
        print(f"\n✗ Fix failed: root_nodes={meta_fixed.get('root_nodes')}, expected={computed_roots}")
    
    # Test that PriorKnowledgeFormatter would get the correct roots
    print("\n" + "-" * 70)
    print("Testing PriorKnowledgeFormatter would get correct roots:")
    from inference_pipeline.utils.prior_knowledge import PriorKnowledgeFormatter
    
    formatter = PriorKnowledgeFormatter(meta_fixed)
    print(f"  Formatter.root_nodes: {formatter.root_nodes}")
    
    # Check what forbidden edges would be generated
    forbidden = formatter.extract_root_forbidden_edges()
    print(f"  Root forbidden edges count: {len(forbidden)}")
    print(f"  Sample forbidden edges: {forbidden[:5] if len(forbidden) > 0 else 'None'}")
    
    # Verify no forbidden edges target nodes that actually have parents
    # For collider: 'd' and 'e' have parent 'b', so they should NOT be in root_nodes
    # and thus should NOT have forbidden incoming edges
    nodes_with_parents = ['d', 'e']  # These have parent 'b' in collider pattern
    problematic_forbidden = [edge for edge in forbidden if edge[1] in nodes_with_parents]
    
    if problematic_forbidden:
        print(f"\n✗ PROBLEM: Found {len(problematic_forbidden)} forbidden edges targeting nodes with parents:")
        print(f"  {problematic_forbidden[:5]}")
    else:
        print(f"\n✓ No forbidden edges targeting nodes with parents (d, e)")


if __name__ == "__main__":
    test_fix_with_real_dataset()

