# Root Node Metadata Fix Summary

## Problem

When running experiments with prior knowledge enabled (`--with-prior true`), results were **worse** for collider, backdoor, and large_backdoor patterns. This was counterintuitive since prior knowledge should help, not hurt.

## Root Cause

The metadata stored in `*_meta.pkl` files contained **incorrect root node lists** for these patterns:

1. **Collider pattern (n=5)**: 
   - Stored: `['a', 'c', 'd', 'e']` 
   - Actual: `['a', 'c']` (nodes 'd' and 'e' have parent 'b')

2. **Backdoor pattern (n=5)**:
   - Stored: `['c', 'd', 'e']`
   - Actual: `['d', 'e']` (node 'c' has parents 'd' and 'e')

3. **Large Backdoor pattern (n=5)**:
   - Stored: `['c', 'd', 'e']`
   - Actual: `['c', 'd', 'e']` ✓ (This one was correct!)

### Why This Happened

The pattern template functions in `data_generator/generator/csuite2.py` (e.g., `pattern_collider`, `pattern_backdoor`) defined `root_nodes` based on the pattern structure, not the actual graph. For example:

```python
def pattern_collider(n: int) -> Dict:
    ...
    return {
        "root_nodes": [0, 2] if n == 3 else [0, 2] + list(range(3, n)),  # WRONG!
        ...
    }
```

This incorrectly included nodes `3, 4, ...` (which become 'd', 'e', ...) as roots, even though they have parent 'b' (the collider node).

The `build_graph_from_pattern` function was later fixed to compute roots correctly from the graph (`computed_roots = [node for node in G.nodes if G.in_degree(node) == 0]`), but **existing datasets** still have the old incorrect metadata.

### Impact

When prior knowledge is enabled:
1. `PriorKnowledgeFormatter` reads `root_nodes` from metadata
2. It generates forbidden edges: "no incoming edges to root nodes"
3. For collider: this forbids edges `b→d` and `b→e`, which are **actual edges** in the true graph
4. Algorithms are forced to break these edges, causing worse SHD/F1 scores

## Solution

Added a fix in `inference_pipeline/run_experiments.py` that:

1. **Recomputes root nodes** from the actual adjacency matrix (`graph.npy`)
2. **Compares** with stored metadata
3. **Updates** metadata in-memory if they differ (preserves original for debugging)
4. **Normalizes** string types to handle `np.str_` objects

### Key Functions

- `_compute_root_nodes_from_adj()`: Computes true roots from adjacency matrix
- `_sync_root_nodes_with_graph()`: Updates metadata if roots don't match
- `_normalize_columns()`: Ensures consistent string types

### Code Location

The fix is applied in `run_on_dataset()` before prior knowledge is generated:

```python
meta = _sync_root_nodes_with_graph(meta, true_adj, cols, dataset_dir.name)
prior = format_prior_knowledge_for_algorithm(meta, algo)  # Now uses corrected roots
```

## Verification

Run the test script to verify the fix:

```bash
python test_fix_verification.py
```

Expected output:
- ✓ Root metadata mismatch detected
- ✓ root_nodes updated to correct values
- ✓ No forbidden edges targeting nodes with parents

## Testing

To test with actual experiments:

1. Run a small subset with prior knowledge:
   ```bash
   python run_csuite_pipeline.py run --index <small_index.csv> --with-prior true
   ```

2. Compare results with and without prior knowledge
3. For collider/backdoor patterns, SHD should be **better or equal** with priors, not worse

## Notes

- The fix is **in-memory only** - it doesn't modify the stored `*_meta.pkl` files
- Original root nodes are preserved in `_root_nodes_original` for debugging
- The fix works for both old (incorrect) and new (correct) datasets
- Future dataset generation already uses correct root computation


