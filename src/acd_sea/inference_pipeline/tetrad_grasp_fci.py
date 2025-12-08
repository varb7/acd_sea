#!/usr/bin/env python3
"""
GRaSP-FCI (Greedy Relaxations of the Sparsest Permutation + FCI) implementation using PyTetrad.

GRaSP-FCI uses GRaSP for initial search followed by FCI orientation rules.
Relaxes faithfulness assumption and tends to produce more accurate PAGs than GFCI.
"""

import os
import glob
from typing import Optional, Union

import numpy as np
import pandas as pd
import jpype
import jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype

# Import shared CI test selector
from acd_sea.utils.tetrad_ci_tests import TetradCITestSelector

class TetradGraspFCI:
    """
    GRaSP-FCI: Uses GRaSP (Greedy Relaxations of the Sparsest Permutation) 
    in place of FGES for the initial step in the GFCI algorithm.
    
    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for independence tests in FCI orientation phase
    depth : int, default=-1
        Maximum size of conditioning sets. -1 means unlimited
    include_undirected : bool, default=True
        Whether to include undirected edges in the adjacency matrix
    penalty_discount : float, default=2.0
        Penalty discount parameter for BIC score
    num_starts : int, default=1
        Number of random restarts for GRaSP
    use_raskutti_uhler : bool, default=False
        Whether to use Raskutti-Uhler modification
    use_data_order : bool, default=True
        Whether to use data order for initial permutation
    verbose : bool, default=False
        Whether to print verbose output
    complete_rule_set : bool, default=True
        Whether to use Zhang's complete rule set (True) or R1-R4 only (False)
    max_path_length : int, default=-1
        Maximum length of discriminating paths. -1 means unlimited
    """
    
    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.05)
        self.depth = kwargs.get("depth", -1)
        self.include_undirected = kwargs.get("include_undirected", True)
        self.penalty_discount = kwargs.get("penalty_discount", 2.0)
        self.num_starts = kwargs.get("num_starts", 1)
        self.use_raskutti_uhler = kwargs.get("use_raskutti_uhler", False)
        self.use_data_order = kwargs.get("use_data_order", True)
        self.ordered = kwargs.get("ordered", False)
        self.verbose = kwargs.get("verbose", False)
        self.complete_rule_set = kwargs.get("complete_rule_set", True)
        self.max_path_length = kwargs.get("max_path_length", -1)
        
        # Create CI test selector
        ci_test_params = {
            k: v for k, v in kwargs.items() 
            if k in ["linear_gap_threshold", "gaussian_p_threshold", "max_pairs_for_diag", "max_parents_for_diag"]
        }
        self.ci_selector = TetradCITestSelector(
            alpha=self.alpha,
            **ci_test_params
        )
        
        self._ensure_jvm()
        self._import_tetrad_modules()

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
            raise RuntimeError(
                "No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, "
                "or drop a jar in ./resources/"
            )
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        import edu.cmu.tetrad.search as search
        import edu.cmu.tetrad.search.test as test
        import edu.cmu.tetrad.search.score as score
        import edu.cmu.tetrad.graph as graph
        import pytetrad.tools.translate as ptt
        
        self.search = search
        self.test = test
        self.score_module = score
        self.graph = graph
        self.ptt = ptt

    def _detect_data_types(self, df: pd.DataFrame):
        """Detect categorical and continuous columns."""
        return self.ci_selector.detect_data_types(df)

    def _handle_perfect_correlations(self, df: pd.DataFrame, cont_cols: list, threshold: float = 0.9999):
        """
        Remove columns that are perfectly correlated to avoid singular correlation matrices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        cont_cols : list
            List of continuous column names
        threshold : float
            Correlation threshold above which columns are considered perfectly correlated
            
        Returns:
        --------
        tuple: (cleaned_df, removed_columns)
        """
        if len(cont_cols) < 2:
            return df, []
            
        df_cont = df[cont_cols]
        corr_matrix = df_cont.corr().abs()
        
        # Find pairs with correlation above threshold
        removed_cols = []
        remaining_cols = list(cont_cols)
        
        for i in range(len(cont_cols)):
            if cont_cols[i] not in remaining_cols:
                continue
                
            for j in range(i + 1, len(cont_cols)):
                if cont_cols[j] not in remaining_cols:
                    continue
                    
                if corr_matrix.iloc[i, j] > threshold:
                    # Remove the second column in the pair
                    col_to_remove = cont_cols[j]
                    if col_to_remove in remaining_cols:
                        remaining_cols.remove(col_to_remove)
                        removed_cols.append(col_to_remove)
                        print(f"[INFO] Removing column '{col_to_remove}' (corr={corr_matrix.iloc[i, j]:.6f} with '{cont_cols[i]}')")
        
        if removed_cols:
            df_cleaned = df.drop(columns=removed_cols)
            return df_cleaned, removed_cols
        else:
            return df, []

    def _convert_to_tetrad_format(self, df: pd.DataFrame):
        """Convert pandas DataFrame to Tetrad dataset."""
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        
        # Handle perfect correlations that cause singular matrices
        df, removed_cols = self._handle_perfect_correlations(df, cont)
        if removed_cols:
            print(f"[WARNING] Removed {len(removed_cols)} perfectly correlated columns: {removed_cols}")
            # Update column type lists after removal
            cats = [c for c in cats if c in df.columns]
            cont = [c for c in cont if c in df.columns]
        
        # Run diagnostics
        diagnostics = self.ci_selector.assess_global_diagnostics(df, cats, cont)
        self.ci_selector.set_diagnostics(diagnostics)
        
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _get_independence_test(self, tetrad_data, cats, cont):
        """Create appropriate independence test based on data types."""
        return self.ci_selector.create_independence_test(
            tetrad_data, cats, cont, self.test, self.ptt
        )

    def _get_score(self, tetrad_data, cats, cont):
        """Create appropriate score based on data types."""
        if cats and cont:
            score = self.score_module.ConditionalGaussianBicScore(
                tetrad_data, self.penalty_discount, True
            )
        elif cats and not cont:
            score = self.score_module.BdeuScore(tetrad_data)
            score.setSamplePrior(1.0)
            score.setStructurePrior(1.0)
        else:
            score = self.score_module.SemBicScore(tetrad_data, self.penalty_discount, True)
        return score

    def _pag_to_adjacency(self, pag, columns):
        """
        Convert PAG to FCI-compatible adjacency with values {-1, 0, 1, 2}.
        
        Values:
            -1: Backward edge (a ← b)
             0: No edge
             1: Undirected edge (a — b)
             2: Forward edge (a → b)
        """
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint

        for i, a in enumerate(columns):
            na = pag.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = pag.getNode(b)
                e = pag.getEdge(na, nb)
                if e is None:
                    continue

                # Map endpoints relative to (na, nb)
                if e.getNode1() == na:
                    ea = e.getEndpoint1()
                    eb = e.getEndpoint2()
                else:
                    ea = e.getEndpoint2()
                    eb = e.getEndpoint1()

                # Convert PAG endpoints to FCI-compatible values
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 2      # a -> b (definite directed)
                elif ea == Endpoint.ARROW and eb == Endpoint.TAIL:
                    adj[i, j] = -1     # a <- b (definite backward)
                elif ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                    adj[i, j] = 1      # a - b (undirected/skeleton)
                elif ea == Endpoint.CIRCLE and eb == Endpoint.ARROW:
                    adj[i, j] = 2      # a o-> b (partial forward, treat as directed)
                elif ea == Endpoint.ARROW and eb == Endpoint.CIRCLE:
                    adj[i, j] = -1     # a <-o b (partial backward)
                elif ea == Endpoint.CIRCLE and eb == Endpoint.CIRCLE:
                    adj[i, j] = 1      # a o-o b (fully uncertain, treat as undirected)
                elif ea == Endpoint.ARROW and eb == Endpoint.ARROW:
                    adj[i, j] = 1      # a <-> b (bidirected, treat as undirected)

        return adj

    def run(self, data: Union[pd.DataFrame, np.ndarray], 
            columns: Optional[list] = None,
            prior: Optional[dict] = None) -> np.ndarray:
        """Run GRaSP-FCI algorithm."""
        if isinstance(data, np.ndarray):
            if columns is None:
                raise ValueError("Column names must be provided for numpy array input.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        
        if df.empty:
            raise ValueError("Input data cannot be empty.")
        
        # Build knowledge object if prior knowledge provided
        knowledge = None
        if prior is not None:
            try:
                from utils.tetrad_prior_knowledge import build_tetrad_knowledge
                knowledge = build_tetrad_knowledge(prior, columns)
            except Exception as e:
                print(f"[WARNING] Could not build knowledge for GRaSP-FCI: {e}")
        
        # Convert data and create test/score
        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        
        # Get diagnostics to decide which implementation to use
        diagnostics = self.ci_selector.get_diagnostics()
        regime = diagnostics.get('regime', 'unknown')
        
        # Default: Use Tetrad Grasp-FCI with parametric test
        self.ci_selector._algorithm_impl = "tetrad"
        indep_test = self._get_independence_test(tetrad_data, cats, cont)
        score = self._get_score(tetrad_data, cats, cont)
        
        # Create GRaSP-FCI algorithm
        alg = self.search.GraspFci(indep_test, score)
        
        # Set parameters
        if hasattr(alg, "setDepth"):
            alg.setDepth(self.depth)
        if hasattr(alg, "setNumStarts"):
            alg.setNumStarts(self.num_starts)
        if hasattr(alg, "setUseRaskuttiUhler"):
            alg.setUseRaskuttiUhler(self.use_raskutti_uhler)
        if hasattr(alg, "setUseDataOrder"):
            alg.setUseDataOrder(self.use_data_order)
        if hasattr(alg, "setOrdered"):
            alg.setOrdered(self.ordered)
        if hasattr(alg, "setVerbose"):
            alg.setVerbose(self.verbose)
        if hasattr(alg, "setCompleteRuleSetUsed"):
            alg.setCompleteRuleSetUsed(self.complete_rule_set)
        if hasattr(alg, "setMaxPathLength"):
            alg.setMaxPathLength(self.max_path_length)
        if knowledge is not None and hasattr(alg, "setKnowledge"):
            alg.setKnowledge(knowledge)
        
        # Run search
        pag = alg.search()
        
        return self._pag_to_adjacency(pag, columns)

def run_grasp_fci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.05,
    depth: int = -1,
    include_undirected: bool = True,
    prior: Optional[dict] = None,
    **kwargs,
) -> np.ndarray:
    """Convenience function for GRaSP-FCI."""
    grasp_fci = TetradGraspFCI(
        alpha=alpha,
        depth=depth,
        include_undirected=include_undirected,
        penalty_discount=penalty_discount,
        num_starts=num_starts,
        use_raskutti_uhler=use_raskutti_uhler,
        use_data_order=use_data_order,
    )
    return grasp_fci.run(data, columns, prior=prior)

