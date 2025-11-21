# Tetrad RFCI Changes Report

## Summary

**Status: ✅ File has NOT been modified since commit `28c83a3`**

The current version of `inference_pipeline/tetrad_rfci.py` is **identical** to the version from the last commit that modified it.

## Last Modification

**Commit**: `28c83a3b928c69a789d40536a1b56b002e10422e`  
**Author**: varb7 <varun.bhoj@fau.de>  
**Date**: Mon Oct 27 10:05:46 2025 +0100  
**Message**: "Clean up inference pipeline and integrate prior knowledge"

## Changes Made in Commit 28c83a3

This commit added **prior knowledge support** to `tetrad_rfci.py`. The changes were:

### 1. Modified `_run_rfci()` method (lines ~128-133)
**Before:**
```python
def _run_rfci(self, indep_test):
    rfci = self.search.Rfci(indep_test)
    rfci.setDepth(self.depth)
    return rfci.search()
```

**After:**
```python
def _run_rfci(self, indep_test, knowledge=None):
    rfci = self.search.Rfci(indep_test)
    rfci.setDepth(self.depth)
    if knowledge is not None:
        rfci.setKnowledge(knowledge)
    return rfci.search()
```

**Change**: Added `knowledge` parameter and applied it to RFCI if provided.

### 2. Modified `run()` method signature and implementation (lines ~174-199)
**Before:**
```python
def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None) -> np.ndarray:
    # ... validation code ...
    tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
    indep = self._create_independence_test(tetrad_data, cats, cont)
    pag = self._run_rfci(indep)
    return self._pag_to_adjacency_matrix(pag, columns)
```

**After:**
```python
def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None, prior: Optional[Dict[str, Any]] = None) -> np.ndarray:
    # ... validation code ...
    
    # Build knowledge object if prior knowledge provided
    knowledge = None
    if prior is not None:
        try:
            from utils.tetrad_prior_knowledge import build_tetrad_knowledge
            knowledge = build_tetrad_knowledge(prior, columns)
        except Exception as e:
            print(f"[WARNING] Could not build knowledge for RFCI: {e}")

    tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
    indep = self._create_independence_test(tetrad_data, cats, cont)
    pag = self._run_rfci(indep, knowledge=knowledge)
    return self._pag_to_adjacency_matrix(pag, columns)
```

**Change**: 
- Added `prior` parameter to method signature
- Added prior knowledge building logic using `build_tetrad_knowledge()`
- Passed knowledge to `_run_rfci()`

### 3. Modified `run_rfci()` convenience function (lines ~219-230)
**Before:**
```python
def run_rfci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    count_partial: bool = False,
    include_undirected: bool = True,
) -> np.ndarray:
    rfci = TetradRFCI(alpha=alpha, depth=depth, count_partial=count_partial, include_undirected=include_undirected)
    return rfci.run(data, columns)
```

**After:**
```python
def run_rfci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    count_partial: bool = False,
    include_undirected: bool = True,
    prior: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    rfci = TetradRFCI(alpha=alpha, depth=depth, count_partial=count_partial, include_undirected=include_undirected)
    return rfci.run(data, columns, prior=prior)
```

**Change**: Added `prior` parameter to convenience function and passed it through.

## Current State

### File Statistics
- **Total lines**: 257 (same as commit 28c83a3)
- **Status**: No modifications since Oct 27, 2025
- **Comparison**: `git diff 28c83a3 HEAD -- inference_pipeline/tetrad_rfci.py` shows **no differences**

### Key Features (Current Implementation)

1. **Prior Knowledge Support**: ✅ Fully integrated
   - Accepts `prior` parameter in `run()` method
   - Builds Tetrad knowledge object using `build_tetrad_knowledge()`
   - Applies knowledge to RFCI algorithm via `setKnowledge()`

2. **Data Type Handling**: ✅ Robust
   - Handles mixed categorical/continuous data
   - Uses appropriate independence tests:
     - `IndTestConditionalGaussianLrt` for mixed data
     - `IndTestChiSquare` for discrete data
     - `IndTestFisherZ` for continuous data

3. **PAG to Adjacency Conversion**: ✅ 
   - Extracts directed edges from Partial Ancestral Graph
   - Handles TAIL→ARROW endpoints (directed edges)
   - Optional handling of CIRCLE→ARROW (partial edges) via `count_partial`
   - Optional inclusion of undirected edges via `include_undirected`

## Conclusion

The file `tetrad_rfci.py` is **stable and unchanged** since the prior knowledge integration in October 2025. The current implementation includes all the prior knowledge functionality and matches exactly what was committed in `28c83a3`.

