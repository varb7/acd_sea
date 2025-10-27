#!/usr/bin/env python3
"""
PyTetrad GES wrapper (score-based; outputs CPDAG; optional CPDAG→DAG orientation).
"""

import os, glob
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype


class TetradGES:
    def __init__(self, **kwargs):
        self.penalty_discount = kwargs.get("penalty_discount", 2.0)
        self.equivalent_sample_size = kwargs.get("equivalent_sample_size", 10.0)
        self.orient_cpdag_to_dag = kwargs.get("orient_cpdag_to_dag", True)
        self.include_undirected = kwargs.get("include_undirected", True)
        self.use_prior_knowledge = kwargs.get("use_prior_knowledge", False)
        self.prior_knowledge = kwargs.get("prior_knowledge", None)
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
        import edu.cmu.tetrad.search.score as score
        import edu.cmu.tetrad.graph as graph
        import pytetrad.tools.translate as ptt
        self.search, self.score, self.graph, self.ptt = search, score, graph, ptt

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]): cats.append(c)
            elif is_float_dtype(df[c]): cont.append(c)
            else:
                try: df[c] = df[c].astype("int64"); cats.append(c)
                except Exception: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64"); cont.append(c)
        return cats, cont

    def _convert(self, df: pd.DataFrame):
        df = df.copy(); cats, cont = self._detect_data_types(df)
        for c in df.columns: df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"): tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _score_fn(self, tetrad_data, cats, cont):
        if cats and cont:
            sc = self.score.ConditionalGaussianScore(tetrad_data)
            sc.setPenaltyDiscount(self.penalty_discount)
            return sc
        elif cats and not cont:
            sc = self.score.BDeuScore(tetrad_data)
            sc.setEquivalentSampleSize(self.equivalent_sample_size)
            return sc
        else:
            sc = self.score.SemBicScore(tetrad_data, self.penalty_discount, True)
            return sc

    def _graph_to_adjacency(self, g, columns):
        n = len(columns); adj = np.zeros((n, n), dtype=int); Endpoint = self.graph.Endpoint
        for i, a in enumerate(columns):
            na = g.getNode(a)
            for j, b in enumerate(columns):
                if i == j: continue
                nb = g.getNode(b); e = g.getEdge(na, nb)
                if e is None: continue
                if e.getNode1() == na:
                    ea, eb = e.getEndpoint1(), e.getEndpoint2()
                else:
                    ea, eb = e.getEndpoint2(), e.getEndpoint1()
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                elif self.include_undirected and ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                    adj[i, j] = 1; adj[j, i] = 1
        return adj

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None, prior: Optional[dict] = None) -> np.ndarray:
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
                print(f"[WARNING] Could not build knowledge for GES: {e}")
        
        tetrad_data, cats, cont = self._convert(df)
        sc = self._score_fn(tetrad_data, cats, cont)
        alg = self.search.Ges(sc)
        
        # Apply prior knowledge if provided
        if knowledge is not None and hasattr(alg, "setKnowledge"):
            alg.setKnowledge(knowledge)
        
        cpdag = alg.search()
        if self.orient_cpdag_to_dag:
            try:
                dag = self.graph.GraphTransforms.dagFromCpdag(cpdag)
            except Exception:
                dag = cpdag
        else:
            dag = cpdag
        return self._graph_to_adjacency(dag, columns)


def run_ges(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    penalty_discount: float = 2.0,
    equivalent_sample_size: float = 10.0,
    orient_cpdag_to_dag: bool = True,
    include_undirected: bool = True,
    prior: Optional[Dict] = None,
) -> np.ndarray:
    ges = TetradGES(
        penalty_discount=penalty_discount,
        equivalent_sample_size=equivalent_sample_size,
        orient_cpdag_to_dag=orient_cpdag_to_dag,
        include_undirected=include_undirected,
    )
    return ges.run(data, columns, prior=prior)


