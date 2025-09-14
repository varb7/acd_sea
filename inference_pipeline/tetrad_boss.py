#!/usr/bin/env python3
"""
PyTetrad BOSS wrapper (score-based; CPDAG output with optional DAG orientation).

Uses pytetrad.tools.search (ts.boss) to run BOSS with chosen score:
  - sem-bic for continuous
  - bdeu for discrete
  - cg-bic for mixed

Parameters:
  penalty_discount: float (BIC penalty multiplier)
  use_bes: bool (final backward-equivalence search)
  num_starts: int (random restarts)
  threads: int (parallelism)
  orient_cpdag_to_dag: bool (GraphTransforms.dagFromCpdag)
  include_undirected: bool (include TAIL-TAIL CPDAG edges as symmetric)
"""

import os, glob
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype


class TetradBOSS:
    def __init__(self, **kwargs):
        self.penalty_discount = kwargs.get("penalty_discount", 2.0)
        self.use_bes = kwargs.get("use_bes", True)
        self.num_starts = kwargs.get("num_starts", 10)
        self.threads = kwargs.get("threads", max(1, os.cpu_count() or 1))
        self.orient_cpdag_to_dag = kwargs.get("orient_cpdag_to_dag", True)
        self.include_undirected = kwargs.get("include_undirected", True)
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
        import pytetrad.tools.search as ts
        import pytetrad.tools.translate as ptt
        import edu.cmu.tetrad.graph as graph
        self.ts = ts
        self.ptt = ptt
        self.graph = graph

    def _detect_data_types(self, df: pd.DataFrame):
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

    def _choose_score_string(self, cats, cont) -> str:
        if cats and cont:
            return "cg-bic"
        elif cats and not cont:
            return "bdeu"
        else:
            return "sem-bic"

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

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if columns is None: raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy(); columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty: raise ValueError("Input data cannot be empty.")

        tetrad_data, cats, cont = self._convert(df)
        score_name = self._choose_score_string(cats, cont)

        # Run BOSS via pytetrad.tools.search
        g = self.ts.boss(
            data=tetrad_data,
            score=score_name,
            penalty_discount=float(self.penalty_discount),
            use_bes=bool(self.use_bes),
            num_starts=int(self.num_starts),
            threads=int(self.threads),
        )

        # Optional CPDAG→DAG orientation
        if self.orient_cpdag_to_dag:
            try:
                g = self.graph.GraphTransforms.dagFromCpdag(g)
            except Exception:
                pass

        return self._graph_to_adjacency(g, columns)


def run_boss_tetrad(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    penalty_discount: float = 2.0,
    use_bes: bool = True,
    num_starts: int = 10,
    threads: int = 8,
    orient_cpdag_to_dag: bool = True,
    include_undirected: bool = True,
) -> np.ndarray:
    boss = TetradBOSS(
        penalty_discount=penalty_discount,
        use_bes=use_bes,
        num_starts=num_starts,
        threads=threads,
        orient_cpdag_to_dag=orient_cpdag_to_dag,
        include_undirected=include_undirected,
    )
    return boss.run(data, columns)


