#!/usr/bin/env python3
"""
PyTetrad CPC (Conservative PC) wrapper.
Returns CPDAG; supports optional inclusion of undirected skeleton edges.
"""

import os, glob
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files

# Import shared CI test selector
from acd_sea.utils.tetrad_ci_tests import TetradCITestSelector

class TetradCPC:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.05)
        self.depth = kwargs.get("depth", -1)
        self.include_undirected = kwargs.get("include_undirected", True)
        
        # Create CI test selector
        ci_test_params = {
            k: v for k, v in kwargs.items() 
            if k in ["linear_gap_threshold", "gaussian_p_threshold", "max_pairs_for_diag", "max_parents_for_diag"]
        }
        self.ci_selector = TetradCITestSelector(
            alpha=self.alpha,
            **ci_test_params
        )
        
        self._ensure_jvm(); self._import_tetrad_modules()

    def _ensure_jvm(self):
        if jpype.isJVMStarted():
            return
        jars = []
        try:
            jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
        except Exception:
            pass
        if os.getenv("TETRAD_JAR"):
            jars.append(os.getenv("TETRAD_JAR"))
        jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
        jars = [j for j in jars if j and os.path.exists(j)]
        if not jars:
            raise RuntimeError("No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, or drop a jar in ./resources/")
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        import edu.cmu.tetrad.search as search
        import edu.cmu.tetrad.search.test as test
        import edu.cmu.tetrad.graph as graph
        import pytetrad.tools.translate as ptt
        self.search, self.test, self.graph, self.ptt = search, test, graph, ptt

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        return self.ci_selector.detect_data_types(df)

    def _convert(self, df: pd.DataFrame):
        df = df.copy(); cats, cont = self._detect_data_types(df)
        
        # Run diagnostics
        diagnostics = self.ci_selector.assess_global_diagnostics(df, cats, cont)
        self.ci_selector.set_diagnostics(diagnostics)
        
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"): tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _indep(self, tetrad_data, cats, cont):
        return self.ci_selector.create_independence_test(
            tetrad_data, cats, cont, self.test, self.ptt
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics including selected CI test."""
        return self.ci_selector.get_diagnostics()

    def _graph_to_adjacency(self, g, columns):
        """
        Convert CPDAG to FCI-compatible adjacency with values {-1, 0, 1, 2}.
        
        Values:
            -1: Backward edge (a ← b)
             0: No edge
             1: Undirected edge (a — b)
             2: Forward edge (a → b)
        
        Note: CPDAGs only have TAIL-TAIL (undirected) and TAIL-ARROW (directed) edges.
        """
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint

        for i, a in enumerate(columns):
            na = g.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = g.getNode(b)
                e = g.getEdge(na, nb)
                if e is None:
                    continue

                # Map endpoints relative to (na, nb)
                if e.getNode1() == na:
                    ea = e.getEndpoint1()
                    eb = e.getEndpoint2()
                else:
                    ea = e.getEndpoint2()
                    eb = e.getEndpoint1()

                # Convert CPDAG endpoints to FCI-compatible values
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 2      # a -> b (directed)
                elif ea == Endpoint.ARROW and eb == Endpoint.TAIL:
                    adj[i, j] = -1     # a <- b (backward)
                elif ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                    adj[i, j] = 1      # a - b (undirected)

        return adj

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None, 
            prior: Optional[dict] = None) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if columns is None: raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy(); columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty: raise ValueError("Input data cannot be empty.")
        
        # Build knowledge object if prior knowledge provided
        knowledge = None
        if prior is not None:
            try:
                from utils.tetrad_prior_knowledge import build_tetrad_knowledge
                knowledge = build_tetrad_knowledge(prior, columns)
            except Exception as e:
                print(f"[WARNING] Could not build knowledge for CPC: {e}")
        
        tetrad_data, cats, cont = self._convert(df)
        
        # Get diagnostics to decide which implementation to use
        diagnostics = self.ci_selector.get_diagnostics()
        regime = diagnostics.get('regime', 'unknown')
        
        # Default: Use Tetrad CPC with parametric test
        self.ci_selector._algorithm_impl = "tetrad"
        indep = self._indep(tetrad_data, cats, cont)
        alg = self.search.Cpc(indep)
        if hasattr(alg, "setDepth"): alg.setDepth(self.depth)
        if knowledge is not None: alg.setKnowledge(knowledge)
        cpdag = alg.search()
        return self._graph_to_adjacency(cpdag, columns)

def run_cpc(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.05,
    depth: int = -1,
    include_undirected: bool = True,
    prior: Optional[dict] = None,
    **kwargs,
) -> np.ndarray:
    cpc = TetradCPC(alpha=alpha, depth=depth, include_undirected=include_undirected, **kwargs)
    return cpc.run(data, columns, prior=prior)

