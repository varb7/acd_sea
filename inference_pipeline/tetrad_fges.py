#!/usr/bin/env python3
"""
Modular FGES (Fast Greedy Equivalence Search) implementation using PyTetrad.

- Robust JVM bootstrap (bundled jar / env / ./resources).
- Safe dtype handling: ints/categories -> discrete; floats -> continuous.
- Correct score construction: use setters (penalty, ESS), not extra ctor args.
- Correct adjacency: direct edges only (no transitive closure).
- Chooses score by data type: CG (mixed), BDeu (discrete), SemBic (continuous).
"""

import os, glob, warnings
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype

warnings.filterwarnings("ignore")


class TetradFGES:
    """
    Clean FGES wrapper using PyTetrad.
    Parameters:
      penalty_discount: float, score complexity penalty (default 2.0)
      max_degree: int, limit degree per node (-1 = unlimited)
      parallel: bool, try to use multiple threads if supported by the JAR
      equivalent_sample_size: float, for BDeu score on all-discrete data
    """

    def __init__(self, **kwargs):
        self.penalty_discount = kwargs.get("penalty_discount", 2.0)
        self.max_degree = kwargs.get("max_degree", -1)
        self.parallel = kwargs.get("parallel", False)
        self.equivalent_sample_size = kwargs.get("equivalent_sample_size", 10.0)
        # When True, include undirected CPDAG edges (TAIL-TAIL) as symmetric edges in adjacency
        self.include_undirected = kwargs.get("include_undirected", True)
        # When True, convert CPDAG to a DAG using GraphTransforms.dagFromCpdag
        self.orient_cpdag_to_dag = kwargs.get("orient_cpdag_to_dag", True)

        self._ensure_jvm()
        self._import_tetrad_modules()

    # ---------------- JVM + imports ----------------

    def _ensure_jvm(self):
        if jpype.isJVMStarted():
            return
        jars = []
        # 1) bundled jar from py-tetrad
        try:
            jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
        except Exception:
            pass
        # 2) env override
        if os.getenv("TETRAD_JAR"):
            jars.append(os.getenv("TETRAD_JAR"))
        # 3) local resources
        jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
        jars = [j for j in jars if j and os.path.exists(j)]
        if not jars:
            raise RuntimeError(
                "No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, or drop a jar in ./resources/"
            )
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        try:
            import edu.cmu.tetrad.search as search
            import edu.cmu.tetrad.search.score as score
            import edu.cmu.tetrad.graph as graph
            import pytetrad.tools.translate as ptt
        except Exception as e:
            raise RuntimeError(f"Failed to import Tetrad modules: {e}")
        self.search = search
        self.score = score
        self.graph = graph
        self.ptt = ptt

    # ---------------- Type handling ----------------

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Return (categorical_cols, continuous_cols) based on dtype (no uniqueness heuristics)."""
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]):
                cats.append(c)
            elif is_float_dtype(df[c]):
                cont.append(c)
            else:
                # Best-effort fallback: try to coerce to int; else numeric float
                try:
                    df[c] = df[c].astype("int64")
                    cats.append(c)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    cont.append(c)
        return cats, cont

    def _convert_to_tetrad_format(self, df: pd.DataFrame):
        """Coerce dtypes explicitly, return (Tetrad DataSet, cats, cont)."""
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        # Coerce explicitly
        for c in df.columns:
            if c in cats:
                df[c] = df[c].astype("int64")
            else:
                df[c] = df[c].astype("float64")

        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    # ---------------- Scores + FGES ----------------

    def _create_score_function(self, tetrad_data, cats, cont):
        """Pick score by data type; set penalty/ESS via setters."""
        if cats and cont:
            sc = self.score.ConditionalGaussianScore(tetrad_data)  # mixed
            sc.setPenaltyDiscount(self.penalty_discount)
            self.last_score_type = "ConditionalGaussianScore (mixed)"
        elif cats and not cont:
            sc = self.score.BDeuScore(tetrad_data)  # discrete
            sc.setEquivalentSampleSize(self.equivalent_sample_size)
            self.last_score_type = "BDeuScore (discrete)"
        else:
            sc = self.score.SemBicScore(tetrad_data, self.penalty_discount, True)  # continuous
            self.last_score_type = "SemBicScore (continuous)"
        return sc

    def _run_fges(self, score_function):
        fges = self.search.Fges(score_function)
        if self.max_degree is not None and self.max_degree >= 0:
            fges.setMaxDegree(self.max_degree)
        # Try to enable parallelism if supported by this Tetrad build
        if self.parallel:
            for meth in ("setNumThreads", "setParallelism", "setUseParallel"):
                if hasattr(fges, meth):
                    try:
                        if meth == "setNumThreads":
                            getattr(fges, meth)(max(1, os.cpu_count() or 1))
                        elif meth == "setUseParallel":
                            getattr(fges, meth)(True)
                        else:
                            getattr(fges, meth)(True)  # generic boolean
                    except Exception:
                        pass
        return fges.search()

    # ---------------- Adjacency (direct edges only) ----------------

    def _dag_to_adjacency_matrix(self, dag, columns: list) -> np.ndarray:
        """Mark a→b iff there is a DIRECT edge a→b; optionally include undirected CPDAG edges."""
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint  # TAIL, ARROW, ...

        # Iterate declared edges (most robust across versions)
        for e in list(dag.getEdges()):
            n1 = e.getNode1(); n2 = e.getNode2()
            a = n1.getName(); b = n2.getName()
            ea = e.getProximalEndpoint(n1); eb = e.getProximalEndpoint(n2)
            if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                i = columns.index(a); j = columns.index(b)
                adj[i, j] = 1
            elif eb == Endpoint.TAIL and ea == Endpoint.ARROW:
                i = columns.index(b); j = columns.index(a)
                adj[i, j] = 1
            elif self.include_undirected and ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                i = columns.index(a); j = columns.index(b)
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    # ---------------- Public API ----------------

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None) -> np.ndarray:
        """Run FGES and return a directed adjacency matrix (parents → children)."""
        if isinstance(data, np.ndarray):
            if columns is None:
                raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy(); columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty:
            raise ValueError("Input data cannot be empty.")

        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        sc = self._create_score_function(tetrad_data, cats, cont)
        graph_out = self._run_fges(sc)
        # FGES typically returns a CPDAG; optionally orient to a DAG
        if self.orient_cpdag_to_dag:
            try:
                # Static method call in Tetrad
                dag = self.graph.GraphTransforms.dagFromCpdag(graph_out)
            except Exception:
                dag = graph_out  # fallback
        else:
            dag = graph_out
        return self._dag_to_adjacency_matrix(dag, columns)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "penalty_discount": self.penalty_discount,
            "max_degree": self.max_degree,
            "parallel": self.parallel,
            "equivalent_sample_size": self.equivalent_sample_size,
            "last_score_type": getattr(self, "last_score_type", "Unknown"),
        }

    def set_parameters(self, **kwargs):
        if "penalty_discount" in kwargs:
            self.penalty_discount = kwargs["penalty_discount"]
        if "max_degree" in kwargs:
            self.max_degree = kwargs["max_degree"]
        if "parallel" in kwargs:
            self.parallel = kwargs["parallel"]
        if "equivalent_sample_size" in kwargs:
            self.equivalent_sample_size = kwargs["equivalent_sample_size"]


# ---------------- Convenience function ----------------

def run_fges(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    penalty_discount: float = 2.0,
    max_degree: int = -1,
    parallel: bool = False,
    equivalent_sample_size: float = 10.0,
    orient_cpdag_to_dag: bool = True,
) -> np.ndarray:
    fges = TetradFGES(
        penalty_discount=penalty_discount,
        max_degree=max_degree,
        parallel=parallel,
        equivalent_sample_size=equivalent_sample_size,
        orient_cpdag_to_dag=orient_cpdag_to_dag,
    )
    return fges.run(data, columns)


# ---------------- Quick sanity demo ----------------

if __name__ == "__main__":
    print("Tetrad FGES Module (sanity test)")
    np.random.seed(42)
    n = 1000

    # Mixed toy DAG:
    # cat -> x -> y, and cat -> y; noise unrelated
    cat = np.random.choice([0, 1, 2], size=n)
    x = (cat - cat.mean()) + np.random.normal(0, 1, size=n)
    y = 1.0 * x + 0.8 * (cat == 1) + 1.5 * (cat == 2) + np.random.normal(0, 1, size=n)
    noise = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({"cat": cat.astype("int64"),
                       "x": x.astype("float64"),
                       "y": y.astype("float64"),
                       "noise": noise.astype("float64")})

    print("Running FGES (default params)…")
    adj1 = run_fges(df)
    print("Edges (default penalty=2.0):", int(np.sum(adj1)))
    print(adj1)

    print("\nRunning FGES (more aggressive, penalty=0.5, max_degree=3)…")
    adj2 = run_fges(df, penalty_discount=0.5, max_degree=3)
    print("Edges (penalty=0.5):", int(np.sum(adj2)))
    print(adj2)
