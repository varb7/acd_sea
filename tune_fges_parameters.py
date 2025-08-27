#!/usr/bin/env python3
# tune_fges_parameters.py

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
import edu.cmu.tetrad.search.score as score
import pytetrad.tools.translate as ptt

# Generate a consistent dataset for testing
np.random.seed(42)
random.seed(42)

print("=== FGES Parameter Tuning Analysis ===\n")

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
penalties = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
max_degrees = [2, 3, 4, -1]  # -1 = unlimited
parallel_flags = [False, True]  # Parallel execution

print(f"\nTesting parameter combinations:")
print(f"  Penalty discounts: {penalties}")
print(f"  Max degrees: {max_degrees} (where -1 = unlimited)")
print(f"  Parallel execution: {parallel_flags}")
print(f"  Total combinations: {len(penalties) * len(max_degrees) * len(parallel_flags)}")

results = []

for penalty in penalties:
    for max_degree in max_degrees:
        for parallel in parallel_flags:
            print(f"\n" + "="*60)
            print(f"FGES: penalty={penalty}, max_degree={max_degree}, parallel={parallel}")
            print("="*60)
            
            try:
                # Create score with current parameters
                sc = score.ConditionalGaussianScore(tetrad_data, penalty, parallel)
                
                # Create FGES with current parameters
                fges = search.Fges(sc)
                fges.setMaxDegree(max_degree)
                
                # Run FGES
                dag = fges.search()
                edges = list(dag.getEdges())
                
                print(f"Total DAG edges found: {len(edges)}")
                
                # Convert to adjacency matrix
                n = len(columns)
                adj = np.zeros((n, n), dtype=int)
                
                for i, a in enumerate(columns):
                    na = dag.getNode(a)
                    for j, b in enumerate(columns):
                        if i == j:
                            continue
                        nb = dag.getNode(b)
                        if dag.isAncestorOf(na, nb):
                            adj[i, j] = 1
                
                # Calculate metrics
                true_edge_set = set(true_edges)
                detected_edge_set = set()
                
                for i in range(n):
                    for j in range(n):
                        if adj[i, j] == 1:
                            detected_edge_set.add((columns[i], columns[j]))
                
                tp = true_edge_set & detected_edge_set
                fp = detected_edge_set - true_edge_set
                fn = true_edge_set - detected_edge_set
                
                precision = len(tp) / len(detected_edge_set) if detected_edge_set else 0.0
                recall = len(tp) / len(true_edge_set) if true_edge_set else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                
                print(f"\nMetrics:")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall:    {recall:.3f}")
                print(f"  F1-Score:  {f1:.3f}")
                
                # Store results
                results.append({
                    'penalty': penalty,
                    'max_degree': max_degree,
                    'parallel': parallel,
                    'total_edges': len(edges),
                    'detected_edges': len(detected_edge_set),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'penalty': penalty,
                    'max_degree': max_degree,
                    'parallel': parallel,
                    'total_edges': 0,
                    'detected_edges': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'error': str(e)
                })

# Summary table
print(f"\n" + "="*120)
print("FGES PARAMETER TUNING SUMMARY")
print("="*120)
print(f"{'Penalty':<8} {'MaxDeg':<7} {'Parallel':<8} {'Total':<6} {'Detected':<9} {'Precision':<10} {'Recall':<8} {'F1':<6}")
print("-" * 120)

for r in results:
    if 'error' in r:
        print(f"{r['penalty']:<8} {r['max_degree']:<7} {r['parallel']:<8} {'ERROR':<6} {'ERROR':<9} {'ERROR':<10} {'ERROR':<8} {'ERROR':<6}")
    else:
        print(f"{r['penalty']:<8} {r['max_degree']:<7} {r['parallel']:<8} {r['total_edges']:<6} {r['detected_edges']:<9} {r['precision']:<10.3f} {r['recall']:<8.3f} {r['f1_score']:<6.3f}")

# Find best parameters for different objectives
print(f"\n" + "="*60)
print("BEST PARAMETER COMBINATIONS")
print("="*60)

# Filter out error results
valid_results = [r for r in results if 'error' not in r]

if valid_results:
    # Best F1
    best_f1 = max(valid_results, key=lambda x: x['f1_score'])
    print(f"Best F1: penalty={best_f1['penalty']}, max_degree={best_f1['max_degree']}, parallel={best_f1['parallel']}")
    print(f"  F1: {best_f1['f1_score']:.3f}, Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}")
    
    # Best Recall
    best_recall = max(valid_results, key=lambda x: x['recall'])
    print(f"\nBest Recall: penalty={best_recall['penalty']}, max_degree={best_recall['max_degree']}, parallel={best_recall['parallel']}")
    print(f"  Recall: {best_recall['recall']:.3f}, Precision: {best_recall['precision']:.3f}, F1: {best_recall['f1_score']:.3f}")
    
    # Best Precision
    best_precision = max(valid_results, key=lambda x: x['precision'])
    print(f"\nBest Precision: penalty={best_precision['penalty']}, max_degree={best_precision['max_degree']}, parallel={best_precision['parallel']}")
    print(f"  Precision: {best_precision['precision']:.3f}, Recall: {best_precision['recall']:.3f}, F1: {best_precision['f1_score']:.3f}")
    
    # Most edges (highest recall potential)
    most_edges = max(valid_results, key=lambda x: x['detected_edges'])
    print(f"\nMost Edges: penalty={most_edges['penalty']}, max_degree={most_edges['max_degree']}, parallel={most_edges['parallel']}")
    print(f"  Edges: {most_edges['detected_edges']}, Recall: {most_edges['recall']:.3f}, F1: {most_edges['f1_score']:.3f}")

print(f"\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Penalty discount controls sparsity:")
print("   - Lower (0.5-1.0): More edges, higher recall, lower precision")
print("   - Higher (3.0-5.0): Fewer edges, lower recall, higher precision")
print("   - Sweet spot (1.5-2.0): Balanced precision-recall")
print("\n2. Max degree limits overfitting:")
print("   - Lower (2-3): More interpretable, prevents complex patterns")
print("   - Higher (4+): Allows complex relationships, may overfit")
print("   - Unlimited (-1): Maximum flexibility, highest recall potential")
print("\n3. Parallel execution:")
print("   - True: Faster on multi-core systems, may use more memory")
print("   - False: Single-threaded, more memory efficient")
print("\n4. For mixed-type data:")
print("   - Start with penalty=1.5, max_degree=3")
print("   - Adjust penalty based on desired precision vs recall")
print("   - Use parallel=True for large datasets")
