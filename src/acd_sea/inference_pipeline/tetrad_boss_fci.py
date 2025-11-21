#!/usr/bin/env python3
"""
BOSS-FCI (Best Order Score Search + FCI) implementation using PyTetrad.

BOSS-FCI uses BOSS for initial search followed by FCI orientation rules.
Good for latent confounders and more accurate than standard GFCI for some cases.
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


class TetradBossFCI:
    """
    BOSS-FCI: Uses BOSS (Best Order Score Search) in place of FGES 
    for the initial step in the GFCI algorithm.
    
    Parameters
    ----------
    alpha : float, default=0.01
        Significance level for independence tests in FCI orientation phase
    depth : int, default=-1
        Maximum size of conditioning sets. -1 means unlimited
    include_undirected : bool, default=True
        Whether to include undirected edges in the adjacency matrix
    penalty_discount : float, default=2.0
        Penalty discount parameter for BIC score
    num_starts : int, default=1
        Number of random restarts for BOSS search
    use_bes : bool, default=True
        Whether to use BES (Backward Equivalence Search) in BOSS
    verbose : bool, default=False
        Whether to print verbose output
    complete_rule_set : bool, default=True
        Whether to use Zhang's complete rule set (True) or R1-R4 only (False)
    max_path_length : int, default=-1
        Maximum length of discriminating paths. -1 means unlimited
    """
    
    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.01)
        self.depth = kwargs.get("depth", -1)
        self.include_undirected = kwargs.get("include_undirected", True)
        self.penalty_discount = kwargs.get("penalty_discount", 2.0)
        self.num_starts = kwargs.get("num_starts", 1)
        self.use_bes = kwargs.get("use_bes", True)
        self.verbose = kwargs.get("verbose", False)
        self.complete_rule_set = kwargs.get("complete_rule_set", True)
        self.max_path_length = kwargs.get("max_path_length", -1)
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
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]):
                cats.append(c)
            elif is_float_dtype(df[c]):
                cont.append(c)
            else:
                try:
                    df[c] = df[c].astype("int64")
                    cats.append(c)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    cont.append(c)
        return cats, cont

    def _convert_to_tetrad_format(self, df: pd.DataFrame):
        """Convert pandas DataFrame to Tetrad dataset."""
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _get_independence_test(self, tetrad_data, cats, cont):
        """Create appropriate independence test based on data types."""
        if cats and cont:
            return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont:
            return self.test.IndTestChiSquare(tetrad_data, self.alpha)
        else:
            return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    def _get_score(self, tetrad_data, cats, cont):
        """Create appropriate score based on data types."""
        if cats and cont:
            # Mixed data: use Conditional Gaussian BIC score
            score = self.score_module.ConditionalGaussianBicScore(
                tetrad_data, self.penalty_discount, True
            )
        elif cats and not cont:
            # Discrete data only: use BDeu score
            score = self.score_module.BdeuScore(tetrad_data)
            score.setSamplePrior(1.0)
            score.setStructurePrior(1.0)
        else:
            # Continuous data only: use SEM BIC score (corrected signature)
            score = self.score_module.SemBicScore(tetrad_data, self.penalty_discount, True)
        return score

    def _pag_to_adjacency(self, pag, columns):
        """Convert PAG to adjacency matrix."""
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
                
                if e.getNode1() == na:
                    ea, eb = e.getEndpoint1(), e.getEndpoint2()
                else:
                    ea, eb = e.getEndpoint2(), e.getEndpoint1()
                
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                elif self.include_undirected and (ea == Endpoint.TAIL and eb == Endpoint.TAIL):
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        return adj

    def run(self, data: Union[pd.DataFrame, np.ndarray], 
            columns: Optional[list] = None,
            prior: Optional[dict] = None) -> np.ndarray:
        """Run BOSS-FCI algorithm."""
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
                print(f"[WARNING] Could not build knowledge for BOSS-FCI: {e}")
        
        # Convert data and create test/score
        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        indep_test = self._get_independence_test(tetrad_data, cats, cont)
        score = self._get_score(tetrad_data, cats, cont)
        
        # Create BOSS-FCI algorithm
        alg = self.search.BossFci(indep_test, score)
        
        # Set parameters
        if hasattr(alg, "setDepth"):
            alg.setDepth(self.depth)
        if hasattr(alg, "setNumStarts"):
            alg.setNumStarts(self.num_starts)
        if hasattr(alg, "setBossUseBes"):
            alg.setBossUseBes(self.use_bes)
        if hasattr(alg, "setVerbose"):
            alg.setVerbose(self.verbose)
        if hasattr(alg, "setCompleteRuleSetUsed"):
            alg.setCompleteRuleSetUsed(self.complete_rule_set)
        if hasattr(alg, "setMaxDiscriminatingPathLength"):
            alg.setMaxDiscriminatingPathLength(self.max_path_length)
        if knowledge is not None and hasattr(alg, "setKnowledge"):
            alg.setKnowledge(knowledge)
        
        # Run search
        pag = alg.search()
        
        return self._pag_to_adjacency(pag, columns)


def run_boss_fci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    include_undirected: bool = True,
    penalty_discount: float = 2.0,
    num_starts: int = 1,
    use_bes: bool = True,
    prior: Optional[dict] = None,
) -> np.ndarray:
    """Convenience function for BOSS-FCI."""
    boss_fci = TetradBossFCI(
        alpha=alpha,
        depth=depth,
        include_undirected=include_undirected,
        penalty_discount=penalty_discount,
        num_starts=num_starts,
        use_bes=use_bes,
    )
    return boss_fci.run(data, columns, prior=prior)

