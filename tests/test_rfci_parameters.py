#!/usr/bin/env python3
# test_rfci_parameters.py

import os, glob, numpy as np, pandas as pd, jpype, jpype.imports
from importlib.resources import files
from scdg import CausalDataGenerator
import random

def ensure_tetrad_jvm():
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
    if not jars:
        raise RuntimeError("No Tetrad JAR found")
    jpype.startJVM(classpath=jars)

ensure_tetrad_jvm()

import edu.cmu.tetrad.search as search
import edu.cmu.tetrad.search.test as test
import edu.cmu.tetrad.graph as graph
import pytetrad.tools.translate as ptt

# Generate the same dataset for consistent comparison
np.random.seed(42)
random.seed(42)

print("=== RFCI Parameter Tuning Analysis ===\n")

cdg = CausalDataGenerator(num_samples=1000, seed=42)
G, roots = cdg.generate_random_graph(total_nodes=8, root_nodes=3, edges=12)

num_categorical = max(1, int(8 * 0.4))
categorical_nodes = set(list(roots)[:num_categorical])
continuous_nodes = set(G.nodes()) - categorical_nodes

print(f"Dataset setup:")
print(f"  Categorical nodes: {categorical_nodes}")
print(f"  Continuous nodes: {continuous_nodes}")

# Set up the data generation
categorical_root_configs = {}
for node in categorical_nodes:
    if node in roots:
        num_classes = np.random.randint(2, 6)
        categorical_root_configs[node] = {
            "dist": "categorical",
            "num_classes": num_classes
        }
cdg.set_root_distributions(categorical_root_configs)

nominal_configs = {}
for node in categorical_nodes:
    if node not in roots:
        parents = list(G.predecessors(node))
        if parents:
            parent = np.random.choice(parents)
            num_classes = np.random.randint(2, 6)
            nominal_configs[node] = {"input": parent, "num_classes": num_classes}
cdg.set_nominal_nodes(nominal_configs)

cdg.assign_equations_to_graph_nodes(equation_type="random")
data = cdg.generate_data()

# Ground truth
true_edges = list(G.edges())
print(f"\nGround truth edges ({len(true_edges)}):")
for s, t in true_edges:
    s_type = 'cat' if s in categorical_nodes else 'cont'
    t_type = 'cat' if t in categorical_nodes else 'cont'
    print(f"  {s}({s_type}) -> {t}({t_type})")

# Convert to Tetrad
columns = list(data.columns)
df = data.copy()
for c in df.columns:
    if c in categorical_nodes:
        df[c] = df[c].astype("int64")
    else:
        df[c] = df[c].astype("float64")

tetrad_data = ptt.pandas_data_to_tetrad(df)
if hasattr(tetrad_data, "getDataSet"):
    tetrad_data = tetrad_data.getDataSet()

# Test different parameter combinations
alphas = [0.01, 0.05, 0.1]  # More conservative to more aggressive
depths = [1, 2, 3, -1]       # Limited conditioning sets to unlimited
Endpoint = graph.Endpoint

print(f"\nTesting parameter combinations:")
print(f"  Alpha values: {alphas}")
print(f"  Depth values: {depths} (where -1 = unlimited)")
print(f"  Total combinations: {len(alphas) * len(depths)}")

results = []

for alpha in alphas:
    for depth in depths:
        print(f"\n" + "="*60)
        print(f"RFCI: alpha={alpha}, depth={depth}")
        print("="*60)
        
        test_obj = test.IndTestConditionalGaussianLrt(tetrad_data, alpha, False)
        alg = search.Rfci(test_obj)
        alg.setDepth(depth)
        
        pag = alg.search()
        edges = list(pag.getEdges())
        
        print(f"Total PAG edges found: {len(edges)}")
        
        # Categorize edges by type
        directed_edges = []      # -> (TAIL-ARROW)
        partially_oriented = []  # o-> (CIRCLE-ARROW)
        undirected = []         # -- (TAIL-TAIL)
        bidirectional = []      # o-o (CIRCLE-CIRCLE)
        other_edges = []        # any other combination
        
        for e in edges:
            a = e.getNode1().getName()
            b = e.getNode2().getName()
            ea = e.getEndpoint1()
            eb = e.getEndpoint2()
            
            if (ea == Endpoint.TAIL and eb == Endpoint.ARROW):
                directed_edges.append((a, b))
            elif (ea == Endpoint.ARROW and eb == Endpoint.TAIL):
                directed_edges.append((b, a))
            elif (ea == Endpoint.CIRCLE and eb == Endpoint.ARROW):
                partially_oriented.append((a, b))
            elif (ea == Endpoint.ARROW and eb == Endpoint.CIRCLE):
                partially_oriented.append((b, a))
            elif (ea == Endpoint.TAIL and eb == Endpoint.TAIL):
                undirected.append((a, b))
            elif (ea == Endpoint.CIRCLE and eb == Endpoint.CIRCLE):
                bidirectional.append((a, b))
            else:
                other_edges.append((a, b, ea.name(), eb.name()))
        
        print(f"\nEdge breakdown:")
        print(f"  Fully directed (->):     {len(directed_edges)}")
        print(f"  Partially oriented (o->): {len(partially_oriented)}")
        print(f"  Undirected (--):         {len(undirected)}")
        print(f"  Bidirectional (o-o):     {len(bidirectional)}")
        print(f"  Other types:             {len(other_edges)}")
        
        # Calculate metrics using only fully directed edges (strict approach)
        true_edge_set = set(true_edges)
        detected_directed = set(directed_edges)
        
        tp_strict = true_edge_set & detected_directed
        fp_strict = detected_directed - true_edge_set
        fn_strict = true_edge_set - detected_directed
        
        precision_strict = len(tp_strict) / len(detected_directed) if detected_directed else 0.0
        recall_strict = len(tp_strict) / len(true_edge_set) if true_edge_set else 0.0
        f1_strict = 2 * precision_strict * recall_strict / (precision_strict + recall_strict) if (precision_strict + recall_strict) else 0.0
        
        print(f"\nMetrics (only fully directed edges):")
        print(f"  Precision: {precision_strict:.3f}")
        print(f"  Recall:    {recall_strict:.3f}")
        print(f"  F1-Score:  {f1_strict:.3f}")
        
        # Calculate metrics including all edge types (relaxed approach)
        all_detected = set(directed_edges + partially_oriented + [(s,t) for s,t in bidirectional] + [(t,s) for s,t in bidirectional])
        
        tp_relaxed = true_edge_set & all_detected
        fp_relaxed = all_detected - true_edge_set
        fn_relaxed = true_edge_set - all_detected
        
        precision_relaxed = len(tp_relaxed) / len(all_detected) if all_detected else 0.0
        recall_relaxed = len(tp_relaxed) / len(true_edge_set) if true_edge_set else 0.0
        f1_relaxed = 2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed) if (precision_relaxed + recall_relaxed) else 0.0
        
        print(f"\nMetrics (including all edge types):")
        print(f"  Precision: {precision_relaxed:.3f}")
        print(f"  Recall:    {recall_relaxed:.3f}")
        print(f"  F1-Score:  {f1_relaxed:.3f}")
        
        # Store results for summary
        results.append({
            'alpha': alpha,
            'depth': depth,
            'total_edges': len(edges),
            'directed_edges': len(directed_edges),
            'precision_strict': precision_strict,
            'recall_strict': recall_strict,
            'f1_strict': f1_strict,
            'precision_relaxed': precision_relaxed,
            'recall_relaxed': recall_relaxed,
            'f1_relaxed': f1_relaxed
        })

# Summary table
print(f"\n" + "="*100)
print("PARAMETER TUNING SUMMARY")
print("="*100)
print(f"{'Alpha':<6} {'Depth':<6} {'Total':<6} {'Dir':<4} {'P(strict)':<10} {'R(strict)':<10} {'F1(strict)':<10} {'P(relax)':<10} {'R(relax)':<10} {'F1(relax)':<10}")
print("-" * 100)

for r in results:
    print(f"{r['alpha']:<6} {r['depth']:<6} {r['total_edges']:<6} {r['directed_edges']:<4} {r['precision_strict']:<10.3f} {r['recall_strict']:<10.3f} {r['f1_strict']:<10.3f} {r['precision_relaxed']:<10.3f} {r['recall_relaxed']:<10.3f} {r['f1_relaxed']:<10.3f}")

# Find best parameters for different objectives
print(f"\n" + "="*60)
print("BEST PARAMETER COMBINATIONS")
print("="*60)

# Best F1 (strict - only fully directed)
best_f1_strict = max(results, key=lambda x: x['f1_strict'])
print(f"Best F1 (strict): alpha={best_f1_strict['alpha']}, depth={best_f1_strict['depth']}")
print(f"  F1: {best_f1_strict['f1_strict']:.3f}, Precision: {best_f1_strict['precision_strict']:.3f}, Recall: {best_f1_strict['recall_strict']:.3f}")

# Best F1 (relaxed - including all edge types)
best_f1_relaxed = max(results, key=lambda x: x['f1_relaxed'])
print(f"\nBest F1 (relaxed): alpha={best_f1_relaxed['alpha']}, depth={best_f1_relaxed['depth']}")
print(f"  F1: {best_f1_relaxed['f1_relaxed']:.3f}, Precision: {best_f1_relaxed['precision_relaxed']:.3f}, Recall: {best_f1_relaxed['recall_relaxed']:.3f}")

# Best recall (strict)
best_recall_strict = max(results, key=lambda x: x['recall_strict'])
print(f"\nBest Recall (strict): alpha={best_recall_strict['alpha']}, depth={best_recall_strict['depth']}")
print(f"  Recall: {best_recall_strict['recall_strict']:.3f}, Precision: {best_recall_strict['precision_strict']:.3f}, F1: {best_recall_strict['f1_strict']:.3f}")

# Best recall (relaxed)
best_recall_relaxed = max(results, key=lambda x: x['recall_relaxed'])
print(f"\nBest Recall (relaxed): alpha={best_recall_relaxed['alpha']}, depth={best_recall_relaxed['depth']}")
print(f"  Recall: {best_recall_relaxed['recall_relaxed']:.3f}, Precision: {best_recall_relaxed['precision_relaxed']:.3f}, F1: {best_recall_relaxed['f1_relaxed']:.3f}")

print(f"\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Higher alpha (0.05, 0.1) → more aggressive → more edges → higher recall")
print("2. Lower depth (1, 2) → less conditioning → more edges → higher recall")
print("3. Trade-off: higher recall usually means lower precision")
print("4. For mixed-type data, moderate parameters often work best")
print("5. Consider your use case: precision vs recall preference")
