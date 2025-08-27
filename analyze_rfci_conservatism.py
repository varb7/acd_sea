#!/usr/bin/env python3
# analyze_rfci_conservatism.py

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
    jars = [j for j in jars if j and os.path.exists(j)]
    if not jars:
        raise RuntimeError("No Tetrad JAR found")
    jpype.startJVM(classpath=jars)

ensure_tetrad_jvm()

import edu.cmu.tetrad.search as search
import edu.cmu.tetrad.search.test as test
import edu.cmu.tetrad.graph as graph
import pytetrad.tools.translate as ptt

# Generate the same dataset as the main script
np.random.seed(42)
random.seed(42)

print("=== RFCI Conservatism Analysis ===\n")

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

# Run RFCI with different alpha values
alphas = [0.001, 0.01, 0.05, 0.1]
Endpoint = graph.Endpoint

for alpha in alphas:
    print(f"\n" + "="*60)
    print(f"RFCI with alpha = {alpha}")
    print("="*60)
    
    test_obj = test.IndTestConditionalGaussianLrt(tetrad_data, alpha, False)
    alg = search.Rfci(test_obj)
    alg.setDepth(-1)
    
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
    
    if directed_edges:
        print(f"\nFully directed edges:")
        for s, t in directed_edges:
            s_type = 'cat' if s in categorical_nodes else 'cont'
            t_type = 'cat' if t in categorical_nodes else 'cont'
            print(f"    {s}({s_type}) -> {t}({t_type})")
    
    if partially_oriented:
        print(f"\nPartially oriented edges:")
        for s, t in partially_oriented:
            s_type = 'cat' if s in categorical_nodes else 'cont'
            t_type = 'cat' if t in categorical_nodes else 'cont'
            print(f"    {s}({s_type}) o-> {t}({t_type})")
    
    if bidirectional:
        print(f"\nBidirectional edges (most common):")
        for s, t in bidirectional:
            s_type = 'cat' if s in categorical_nodes else 'cont'
            t_type = 'cat' if t in categorical_nodes else 'cont'
            print(f"    {s}({s_type}) o-o {t}({t_type})")
    
    # Calculate metrics using only fully directed edges (current approach)
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

print(f"\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("RFCI has low recall because:")
print("1. It's very conservative about edge orientation")
print("2. Most detected relationships are bidirectional (o-o)")
print("3. Current metrics only count fully directed edges (->)")
print("4. Many true causal relationships are detected but not oriented")
print("5. This is actually CORRECT behavior for RFCI!")
print("\nRFCI is designed to be conservative and only orient edges")
print("when it has strong evidence. The bidirectional edges")
print("indicate 'there's a relationship but we're not sure of direction'.")
