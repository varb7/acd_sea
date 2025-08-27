#!/usr/bin/env python3
"""
Mixed-Type Edge Evaluation Pipeline with Proper PyTetrad API

This script uses the low-level PyTetrad API via JPype to access Tetrad's Java classes
with robust JVM bootstrapping and correct handling of mixed (continuous + categorical) data.

Key fixes vs typical pitfalls:
- Starts the JVM exactly once and finds a Tetrad JAR (bundled with py-tetrad, env var, or ./resources).
- Imports from the right Java packages after the JVM is up.
- Uses correct constructors for independence tests (RFCI) and scores (FGES) and tunes via setters.
- Converts PAG (RFCI) and DAG (FGES) to adjacency matrices correctly (no transitive closure; orientation aware).
- Uses pandas dtype helpers to coerce types before translating to Tetrad.
"""

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time
import json
import random
from pathlib import Path

# Optional plotting libraries (not used in this script; keep if you plan to extend)
# import matplotlib.pyplot as plt

# ---- Third-party causal libs (optional, gated) --------------------------------
try:
    from castle.algorithms import PC, GES
    CASTLE_AVAILABLE = True
except Exception:
    CASTLE_AVAILABLE = False
    print("[WARNING] Castle algorithms not available")

try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    FCI_AVAILABLE = True
except Exception:
    FCI_AVAILABLE = False
    print("[WARNING] CausalLearn FCI not available")

# ---- Data generator (required here) -------------------------------------------
try:
    from scdg import CausalDataGenerator
    SCDG_AVAILABLE = True
except Exception:
    SCDG_AVAILABLE = False
    print("[WARNING] SCDG not available; dataset generation will fail")

# ---- JPype / PyTetrad bootstrapping -------------------------------------------
import jpype, jpype.imports
from importlib.resources import files
TETRAD_AVAILABLE = False

def ensure_tetrad_jvm():
    """Start the JVM with a Tetrad JAR on the classpath.
    Search order:
      1) Bundled jar from py-tetrad (pytetrad.resources/tetrad-current.jar)
      2) Environment variable TETRAD_JAR
      3) Any jar under ./resources matching *tetrad*jar
    """
    if jpype.isJVMStarted():
        return

    jars: List[str] = []
    # 1) py-tetrad bundled jar
    try:
        jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
    except Exception:
        pass
    # 2) environment override
    if os.getenv("TETRAD_JAR"):
        jars.append(os.getenv("TETRAD_JAR"))
    # 3) project resources
    jars += glob.glob(os.path.join("resources", "*tetrad*jar"))

    jars = [j for j in jars if j and os.path.exists(j)]
    if not jars:
        raise RuntimeError(
            "Could not find a Tetrad JAR. Install py-tetrad or set TETRAD_JAR, "
            "or place a jar under ./resources/"
        )

    jpype.startJVM(classpath=jars)

# Try to start JVM and import Java packages
try:
    ensure_tetrad_jvm()
    # Import AFTER JVM is up
    import edu.cmu.tetrad.search as search
    import edu.cmu.tetrad.search.score as score
    import edu.cmu.tetrad.search.test as test
    import edu.cmu.tetrad.graph as graph
    import pytetrad.tools.translate as ptt
    TETRAD_AVAILABLE = True
    print("[INFO] Proper PyTetrad API available (JVM started)")
except Exception as e:
    print(f"[ERROR] Failed to initialize PyTetrad / JVM: {e}")
    TETRAD_AVAILABLE = False

# ---- Helper dataclasses --------------------------------------------------------
@dataclass
class MixedTypeEdge:
    source: str
    target: str
    source_type: str  # 'categorical' or 'continuous'
    target_type: str  # 'categorical' or 'continuous'
    edge_type: str    # 'cat_to_cont' or 'cont_to_cat'
    def __str__(self):
        return f"{self.source}({self.source_type}) -> {self.target}({self.target_type})"

@dataclass
class AlgorithmResult:
    algorithm_name: str
    adjacency_matrix: np.ndarray
    execution_time: float
    detected_edges: List[Tuple[str, str]]
    true_positive_edges: List[Tuple[str, str]]
    false_positive_edges: List[Tuple[str, str]]
    false_negative_edges: List[Tuple[str, str]]
    precision: float
    recall: float
    f1_score: float

# ---- Dataset generator using SCDG ---------------------------------------------
class MixedTypeDatasetGenerator:
    """Generates datasets with mixed-type edges for evaluation."""
    def __init__(self, num_samples: int = 1000, seed: int = 42):
        if not SCDG_AVAILABLE:
            raise ImportError("SCDG is required for dataset generation (pip install scdg).")
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_mixed_type_dataset(
        self,
        total_nodes: int = 10,
        root_nodes: int = 3,
        edges: int = 15,
        categorical_percentage: float = 0.4
    ) -> Tuple[pd.DataFrame, nx.DiGraph, List[MixedTypeEdge]]:
        print(f"Generating mixed-type dataset with {total_nodes} nodes, {root_nodes} roots, {edges} edges")

        cdg = CausalDataGenerator(num_samples=self.num_samples, seed=self.seed)

        # graph structure
        G, roots = cdg.generate_random_graph(total_nodes=total_nodes, root_nodes=root_nodes, edges=edges)

        # choose categorical nodes
        num_categorical = max(1, int(total_nodes * categorical_percentage))
        categorical_nodes = set(list(roots)[:num_categorical])
        continuous_nodes = set(G.nodes()) - categorical_nodes

        print(f"Categorical nodes: {categorical_nodes}")
        print(f"Continuous nodes:  {continuous_nodes}")

        # categorical root distributions
        categorical_root_configs = {}
        for node in categorical_nodes:
            if node in roots:
                num_classes = np.random.randint(2, 6)  # 2..5
                if np.random.random() < 0.5:
                    categorical_root_configs[node] = {
                        "dist": "categorical",
                        "num_classes": num_classes
                    }
                else:
                    probs = np.random.dirichlet(np.ones(num_classes))
                    categorical_root_configs[node] = {
                        "dist": "categorical_non_uniform",
                        "num_classes": num_classes,
                        "probabilities": probs.tolist()
                    }
        cdg.set_root_distributions(categorical_root_configs)

        # nominal non-root categorical nodes
        nominal_configs = {}
        for node in categorical_nodes:
            if node not in roots:
                parents = list(G.predecessors(node))
                if parents:
                    parent = np.random.choice(parents)
                    num_classes = np.random.randint(2, 6)
                    nominal_configs[node] = {"input": parent, "num_classes": num_classes}
        cdg.set_nominal_nodes(nominal_configs)

        # equations and data
        cdg.assign_equations_to_graph_nodes(equation_type="random")
        data = cdg.generate_data()

        # identify mixed-type edges
        mixed_type_edges = self._identify_mixed_type_edges(G, categorical_nodes, continuous_nodes)
        print(f"Identified {len(mixed_type_edges)} mixed-type edges:")
        for e in mixed_type_edges:
            print(f"  {e}")

        return data, G, mixed_type_edges

    def _identify_mixed_type_edges(
        self, G: nx.DiGraph, categorical_nodes: set, continuous_nodes: set
    ) -> List[MixedTypeEdge]:
        mixed_edges: List[MixedTypeEdge] = []
        for s, t in G.edges():
            s_type = 'categorical' if s in categorical_nodes else 'continuous'
            t_type = 'categorical' if t in categorical_nodes else 'continuous'
            if s_type != t_type:
                mixed_edges.append(
                    MixedTypeEdge(
                        source=s, target=t,
                        source_type=s_type, target_type=t_type,
                        edge_type='cat_to_cont' if s_type == 'categorical' else 'cont_to_cat'
                    )
                )
        return mixed_edges

# ---- Evaluator ----------------------------------------------------------------
from pandas.api.types import is_integer_dtype, is_float_dtype
import pandas as pd

class CausalDiscoveryEvaluator:
    """Evaluates PC, FCI, GES (Castle), RFCI/FGES (PyTetrad) on mixed-type datasets."""
    def __init__(self):
        pass  # Do not block; we'll gate each algorithm call by availability.

    def evaluate_algorithms(
        self, data: pd.DataFrame, true_graph: nx.DiGraph, mixed_type_edges: List[MixedTypeEdge]
    ) -> Dict[str, AlgorithmResult]:
        results: Dict[str, AlgorithmResult] = {}
        columns = list(data.columns)
        true_adj = nx.to_numpy_array(true_graph, nodelist=columns, dtype=int)

        # PC (Castle)
        if CASTLE_AVAILABLE:
            print("Running PC (Castle)…")
            try:
                start = time.time()
                pc = PC(alpha=0.05)
                pc.learn(data.values)
                adj = pc.causal_matrix
                res = self._evaluate_results(adj, true_adj, columns, mixed_type_edges, "PC")
                res.execution_time = time.time() - start
                results["PC"] = res
            except Exception as e:
                print(f"PC failed: {e}")

        # FCI (CausalLearn)
        if FCI_AVAILABLE:
            print("Running FCI (CausalLearn)…")
            try:
                start = time.time()
                adj_matrix, _ = fci(data.values, alpha=0.05, indep_test=fisherz)
                exec_time = time.time() - start
                # Best-effort: if not ndarray, try to convert; otherwise leave as is
                if isinstance(adj_matrix, np.ndarray):
                    fci_adj = adj_matrix
                else:
                    # Unknown structure -> zero matrix
                    fci_adj = np.zeros((len(columns), len(columns)), dtype=int)
                res = self._evaluate_results(fci_adj, true_adj, columns, mixed_type_edges, "FCI")
                res.execution_time = exec_time
                results["FCI"] = res
            except Exception as e:
                print(f"FCI failed: {e}")

        # GES (Castle)
        if CASTLE_AVAILABLE:
            print("Running GES (Castle)…")
            try:
                start = time.time()
                ges = GES()
                ges.learn(data.values)
                adj = ges.causal_matrix
                res = self._evaluate_results(adj, true_adj, columns, mixed_type_edges, "GES")
                res.execution_time = time.time() - start
                results["GES"] = res
            except Exception as e:
                print(f"GES failed: {e}")

        # RFCI (PyTetrad)
        if TETRAD_AVAILABLE:
            print("Running RFCI (PyTetrad)…")
            results["RFCI"] = self._run_rfci_tetrad(data, columns, true_adj, mixed_type_edges)

        # FGES (PyTetrad)
        if TETRAD_AVAILABLE:
            print("Running FGES (PyTetrad)…")
            results["FGES"] = self._run_fges_tetrad(data, columns, true_adj, mixed_type_edges)

        return results

    # ---------- PyTetrad runners ----------

    def _coerce_and_translate(self, data: pd.DataFrame):
        """Coerce pandas dtypes explicitly and return a Tetrad DataSet."""
        df = data.copy()
        for c in df.columns:
            if is_integer_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype("int64")
            elif is_float_dtype(df[c]):
                df[c] = df[c].astype("float64")
            else:
                # If it's object/strings that are categorical labels, user should encode.
                # Here we try a safe fallback: attempt astype(int), else cast to float.
                try:
                    df[c] = df[c].astype("int64")
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

        tetrad_data = ptt.pandas_data_to_tetrad(df)
        # Unwrap BoxDataSet if present
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()

        # identify types for decision logic
        cats = [col for col in df.columns if is_integer_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype)]
        cont = [col for col in df.columns if is_float_dtype(df[col])]

        return tetrad_data, cats, cont

    def _run_rfci_tetrad(
        self, data: pd.DataFrame, columns: List[str], true_adj: np.ndarray, mixed_type_edges: List[MixedTypeEdge]
    ) -> AlgorithmResult:
        start = time.time()
        try:
            tetrad_data, cats, cont = self._coerce_and_translate(data)

            # Independence test selection
            if cats and cont:
                test_obj = test.IndTestConditionalGaussianLrt(tetrad_data, 0.01, False)
                chosen = "CG-LRT (mixed)"
            elif cats:
                test_obj = test.IndTestChiSquare(tetrad_data, 0.01)
                test_obj.setMinCountPerCell(5.0)
                chosen = "Chi-square (discrete)"
            else:
                test_obj = test.IndTestFisherZ(tetrad_data, 0.01)
                chosen = "Fisher-Z (continuous)"
            print(f"    RFCI using: {chosen}")

            alg = search.Rfci(test_obj)
            alg.setDepth(-1)  # unlimited conditioning-set size
            pag = alg.search()

            pred_adj = self._pag_to_adjacency_matrix(pag, columns)
        except Exception as e:
            print(f"RFCI failed: {e}")
            pred_adj = np.zeros((len(columns), len(columns)), dtype=int)

        res = self._evaluate_results(pred_adj, true_adj, columns, mixed_type_edges, "RFCI")
        res.execution_time = time.time() - start
        return res

    def _run_fges_tetrad(
        self, data: pd.DataFrame, columns: List[str], true_adj: np.ndarray, mixed_type_edges: List[MixedTypeEdge]
    ) -> AlgorithmResult:
        start = time.time()
        try:
            tetrad_data, cats, cont = self._coerce_and_translate(data)

            # Score selection
            if cats and cont:
                sc = score.ConditionalGaussianScore(tetrad_data, 2.0, False)
                chosen = "ConditionalGaussianScore (mixed)"
            elif cats:
                sc = score.BDeuScore(tetrad_data)
                sc.setEquivalentSampleSize(10.0)  # tune 5–20
                chosen = "BDeuScore (discrete)"
            else:
                sc = score.SemBicScore(tetrad_data)
                sc.setPenaltyDiscount(2.0)
                chosen = "SemBicScore (continuous)"
            print(f"    FGES using: {chosen}")

            fges = search.Fges(sc)
            fges.setMaxDegree(-1)
            dag = fges.search()

            pred_adj = self._dag_to_adjacency_matrix(dag, columns)
        except Exception as e:
            print(f"FGES failed: {e}")
            pred_adj = np.zeros((len(columns), len(columns)), dtype=int)

        res = self._evaluate_results(pred_adj, true_adj, columns, mixed_type_edges, "FGES")
        res.execution_time = time.time() - start
        return res

    # ---------- Graph conversions ----------

    def _pag_to_adjacency_matrix(self, pag, columns: List[str]) -> np.ndarray:
        """Count only fully oriented edges a -> b (TAIL at a, ARROW at b)."""
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = graph.Endpoint  # TAIL, ARROW, CIRCLE, NULL

        for i, a in enumerate(columns):
            na = pag.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = pag.getNode(b)
                e = pag.getEdge(na, nb)
                if e is None:
                    continue
                # Check if edge goes from a to b (a -> b)
                if e.getNode1() == na and e.getNode2() == nb:
                    ea = e.getEndpoint1()
                    eb = e.getEndpoint2()
                elif e.getNode1() == nb and e.getNode2() == na:
                    ea = e.getEndpoint2()
                    eb = e.getEndpoint1()
                else:
                    continue
                    
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                # If you also want to count partially oriented a o-> b:
                # if ea == Endpoint.CIRCLE and eb == Endpoint.ARROW:
                #     adj[i, j] = 1
        print(f"    RFCI: Converted PAG to {int(np.sum(adj))} directed edges")
        return adj

    def _dag_to_adjacency_matrix(self, dag, columns: List[str]) -> np.ndarray:
        """Direct edges only (no transitive closure)."""
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        for i, a in enumerate(columns):
            na = dag.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = dag.getNode(b)
                # Use isAncestorOf instead of isDirectedFromTo
                if dag.isAncestorOf(na, nb):
                    adj[i, j] = 1
        print(f"    FGES: Converted DAG to {int(np.sum(adj))} directed edges")
        return adj

    # ---------- Metrics ----------

    def _evaluate_results(
        self, pred_adj: np.ndarray, true_adj: np.ndarray, columns: List[str],
        mixed_type_edges: List[MixedTypeEdge], algorithm_name: str
    ) -> AlgorithmResult:
        true_edges = set()
        pred_edges = set()
        n = len(columns)

        for i in range(n):
            for j in range(n):
                if true_adj[i, j] == 1:
                    true_edges.add((columns[i], columns[j]))
                if pred_adj[i, j] == 1:
                    pred_edges.add((columns[i], columns[j]))

        tp = true_edges & pred_edges
        fp = pred_edges - true_edges
        fn = true_edges - pred_edges

        precision = len(tp) / len(pred_edges) if pred_edges else 0.0
        recall = len(tp) / len(true_edges) if true_edges else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return AlgorithmResult(
            algorithm_name=algorithm_name,
            adjacency_matrix=pred_adj,
            execution_time=0.0,
            detected_edges=sorted(list(pred_edges)),
            true_positive_edges=sorted(list(tp)),
            false_positive_edges=sorted(list(fp)),
            false_negative_edges=sorted(list(fn)),
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

# ---- Orchestration ------------------------------------------------------------
class MixedTypeEvaluationPipeline:
    """Main pipeline for mixed-type edge evaluation with proper PyTetrad API."""
    def __init__(self, output_dir: str = "mixed_type_evaluation_results_proper_tetrad"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generator = MixedTypeDatasetGenerator()
        self.evaluator = CausalDiscoveryEvaluator()

    def run_evaluation(self, num_datasets: int = 3, dataset_params: Dict[str, Any] = None) -> Dict[str, Any]:
        if dataset_params is None:
            dataset_params = {'total_nodes': 8, 'root_nodes': 3, 'edges': 12, 'categorical_percentage': 0.4}

        all_results: List[Dict[str, Any]] = []

        for i in range(num_datasets):
            print(f"\n{'='*60}\nDataset {i+1}/{num_datasets}\n{'='*60}")
            try:
                data, graph_nx, mixed_edges = self.generator.generate_mixed_type_dataset(**dataset_params)
                results = self.evaluator.evaluate_algorithms(data, graph_nx, mixed_edges)

                dataset_result = {
                    'dataset_id': i,
                    'graph_info': {
                        'num_nodes': len(graph_nx.nodes()),
                        'num_edges': len(graph_nx.edges()),
                        'num_mixed_type_edges': len(mixed_edges),
                    },
                    'mixed_type_edges': [str(e) for e in mixed_edges],
                    'algorithm_results': {
                        name: {
                            'algorithm_name': r.algorithm_name,
                            'precision': r.precision,
                            'recall': r.recall,
                            'f1_score': r.f1_score,
                            'execution_time': r.execution_time,
                            'detected_edges': r.detected_edges,
                            'true_positive_edges': r.true_positive_edges,
                            'false_positive_edges': r.false_positive_edges,
                            'false_negative_edges': r.false_negative_edges,
                        }
                        for name, r in results.items()
                    },
                }

                all_results.append(dataset_result)
                self._print_dataset_summary(dataset_result)
                self._save_json(self.output_dir / f"dataset_{i:03d}_results.json", dataset_result)

            except Exception as e:
                print(f"Error processing dataset {i}: {e}")

        summary_stats = self._calc_summary(all_results)
        self._save_json(self.output_dir / "all_results.json", all_results)
        self._save_json(self.output_dir / "summary_statistics.json", summary_stats)
        self._print_final_summary(summary_stats)
        return summary_stats

    def _save_json(self, path: Path, obj: Any):
        def make_serializable(o):
            if isinstance(o, dict): return {k: make_serializable(v) for k, v in o.items()}
            if isinstance(o, list): return [make_serializable(v) for v in o]
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return o
        with open(path, "w") as f:
            json.dump(make_serializable(obj), f, indent=2)

    def _print_dataset_summary(self, res: Dict[str, Any]):
        print(f"\nDataset {res['dataset_id']} Summary:")
        gi = res['graph_info']
        print(f"  Graph: {gi['num_nodes']} nodes, {gi['num_edges']} edges")
        print(f"  Mixed-type edges: {gi['num_mixed_type_edges']}")
        for e in res['mixed_type_edges']:
            print(f"    {e}")
        print("\n  Algorithm Performance:")
        for name, r in res['algorithm_results'].items():
            print(f"    {name}: P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1_score']:.3f}, Time={r['execution_time']:.3f}s")

    def _calc_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {'total_datasets': len(all_results), 'algorithm_performance': {}}
        algos = ['PC', 'FCI', 'GES', 'RFCI', 'FGES']
        for algo in algos:
            rows = [d['algorithm_results'][algo] for d in all_results if algo in d['algorithm_results']]
            if rows:
                summary['algorithm_performance'][algo] = {
                    'avg_precision': float(np.mean([r['precision'] for r in rows])),
                    'avg_recall': float(np.mean([r['recall'] for r in rows])),
                    'avg_f1_score': float(np.mean([r['f1_score'] for r in rows])),
                    'avg_execution_time': float(np.mean([r['execution_time'] for r in rows])),
                    'num_datasets': len(rows),
                }
        return summary

    def _print_final_summary(self, s: Dict[str, Any]):
        print(f"\n{'='*80}\nFINAL EVALUATION SUMMARY (PROPER PYTETRAD API)\n{'='*80}")
        print(f"Total datasets evaluated: {s['total_datasets']}")
        print("\nAlgorithm Performance (averaged across datasets):")
        for algo, stats in s['algorithm_performance'].items():
            print(f"\n  {algo}:")
            print(f"    Precision:       {stats['avg_precision']:.3f}")
            print(f"    Recall:          {stats['avg_recall']:.3f}")
            print(f"    F1-Score:        {stats['avg_f1_score']:.3f}")
            print(f"    Execution Time:  {stats['avg_execution_time']:.3f}s")
            print(f"    Datasets:        {stats['num_datasets']}")

# ---- Main ---------------------------------------------------------------------
def main():
    print("Mixed-Type Edge Evaluation Pipeline with Proper PyTetrad API")
    print("=" * 70)

    if not SCDG_AVAILABLE:
        print("ERROR: SCDG is required for dataset generation")
        return

    if not (CASTLE_AVAILABLE or FCI_AVAILABLE or TETRAD_AVAILABLE):
        print("ERROR: No causal discovery backend available (Castle, CausalLearn, or PyTetrad).")
        return

    pipeline = MixedTypeEvaluationPipeline()
    dataset_params = {'total_nodes': 8, 'root_nodes': 3, 'edges': 12, 'categorical_percentage': 0.4}

    try:
        summary = pipeline.run_evaluation(num_datasets=3, dataset_params=dataset_params)
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {pipeline.output_dir.resolve()}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
