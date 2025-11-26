#!/usr/bin/env python3
"""
Comprehensive comparison: All algorithms with and without KCI switching
"""
import numpy as np
import pandas as pd
import warnings
import inspect
warnings.filterwarnings("ignore")

# Import standard versions (FIXED PATHS)
import sys
sys.path.insert(0, 'd:/acd_sea/src')
from acd_sea.inference_pipeline.tetrad_pc import run_pc
from acd_sea.inference_pipeline.tetrad_fci import run_fci
from acd_sea.inference_pipeline.tetrad_cfci import run_cfci
from acd_sea.inference_pipeline.tetrad_rfci import run_rfci
from acd_sea.inference_pipeline.tetrad_cpc import run_cpc
from acd_sea.inference_pipeline.tetrad_gfci import run_gfci
from acd_sea.inference_pipeline.tetrad_fci_max import run_fci_max

# No KCI versions needed - using standard implementations only


def gen_linear(n=500, seed=42):
    """Linear 5-variable network: a -> b -> c, a -> d -> e, b -> e (more complex)"""
    np.random.seed(seed)
    
    # Root node
    a = np.random.normal(0, 1, n)
    
    # Linear relationships
    b = 0.8 * a + np.random.normal(0, 0.3, n)
    c = 0.7 * b + np.random.normal(0, 0.3, n)
    d = 0.6 * a + np.random.normal(0, 0.4, n)
    e = 0.5 * b + 0.4 * d + np.random.normal(0, 0.3, n)
    
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}).astype("float64")
    
    # True adjacency matrix (5x5)
    # a -> b, b -> c, a -> d, d -> e, b -> e
    true = np.array([
        [0, 1, 0, 1, 0],  # a -> b, d
        [0, 0, 1, 0, 1],  # b -> c, e  
        [0, 0, 0, 0, 0],  # c -> none
        [0, 0, 0, 0, 1],  # d -> e
        [0, 0, 0, 0, 0]   # e -> none
    ])
    return df, true


def gen_nonlinear(n=500, seed=42):
    """Non-linear 5-variable network with diverse relationships"""
    np.random.seed(seed)
    
    # Root node
    a = np.random.uniform(-2, 2, n)
    
    # Non-linear relationships
    b = 0.5 * a**2 + np.random.normal(0, 0.4, n)  # Quadratic
    c = np.sin(b) + np.random.normal(0, 0.3, n)   # Sinusoidal
    d = np.exp(0.3 * a) + np.random.normal(0, 0.5, n)  # Exponential
    e = np.tanh(0.6 * b) + 0.4 * np.log(np.abs(d) + 1) + np.random.normal(0, 0.3, n)  # Mixed non-linear
    
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}).astype("float64")
    
    # Same structure as linear case for comparison
    # a -> b, b -> c, a -> d, d -> e, b -> e
    true = np.array([
        [0, 1, 0, 1, 0],  # a -> b, d
        [0, 0, 1, 0, 1],  # b -> c, e  
        [0, 0, 0, 0, 0],  # c -> none
        [0, 0, 0, 0, 1],  # d -> e
        [0, 0, 0, 0, 0]   # e -> none
    ])
    return df, true


def gen_confounded_nonlinear(n=500, seed=42):
    """Complex 5-variable network with confounding and colliders"""
    np.random.seed(seed)
    
    # Hidden confounder (not observed)
    h = np.random.normal(0, 1, n)
    
    # Observed variables with confounding
    a = 0.7 * h + np.random.normal(0, 0.5, n)
    b = np.sin(0.8 * a) + 0.5 * h + np.random.normal(0, 0.4, n)  # Confounded by h
    c = np.exp(0.4 * a) * np.cos(0.6 * b) + np.random.normal(0, 0.3, n)  # Collider: a->c<-b
    d = np.tanh(0.5 * b) + np.random.normal(0, 0.4, n)
    e = 0.6 * c + np.log(np.abs(d) + 1) + np.random.normal(0, 0.3, n)  # Collider: c->e<-d
    
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}).astype("float64")
    
    # True structure (without hidden confounder)
    # a -> b (confounded), a -> c, b -> c, b -> d, c -> e, d -> e
    true = np.array([
        [0, 1, 1, 0, 0],  # a -> b, c
        [0, 0, 1, 1, 0],  # b -> c, d
        [0, 0, 0, 0, 1],  # c -> e
        [0, 0, 0, 0, 1],  # d -> e
        [0, 0, 0, 0, 0]   # e -> none
    ])
    return df, true


def metrics(pred, true):
    p, t = (pred > 0).astype(int), (true > 0).astype(int)
    tp = np.sum(p * t)
    fp = np.sum(p * (1 - t))
    fn = np.sum((1 - p) * t)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "prec": prec, "rec": rec, "edges": np.sum(p)}


def safe_run(func, df, true_matrix, alpha=0.01, depth=3):
    """Run algorithm safely with standard parameters"""
    try:
        # Standard Tetrad parameters
        kwargs = {
            'alpha': alpha, 
            'depth': depth,
            'include_undirected': True,
            'prior': None
        }
        
        result = func(df, **kwargs)
        return metrics(result, true_matrix)
    except Exception as e:
        print(f"ERROR in safe_run with {func.__name__}: {e}")
        return None


def main():
    print("=" * 92)
    print("Constraint-Based Algorithm Comparison (Tetrad Implementations)")
    print("=" * 92)
    
    algs = [
        ("PC",      run_pc),
        ("FCI",     run_fci),
        ("CFCI",    run_cfci),
        ("RFCI",    run_rfci),
        ("CPC",     run_cpc),
        ("GFCI",    run_gfci),
        ("FCI-Max", run_fci_max),
    ]
    
    # Linear data
    print("\n" + "=" * 92)
    print("LINEAR GAUSSIAN DATA (5 variables: a->b->c, a->d->e, b->e)")
    print("=" * 92)
    df_lin, true_lin = gen_linear()  # FIXED: Get true matrix
    print(f"\n{'Algorithm':<12} {'F1 Score':>12} {'Precision':>12} {'Recall':>12} {'Edges':>10}")
    print("-" * 70)
    
    lin_results = {}
    for name, func in algs:
        result = safe_run(func, df_lin, true_lin)
        
        if result:
            print(f"{name:<12} {result['f1']:>12.3f} {result['prec']:>12.3f} {result['rec']:>12.3f} {result['edges']:>10.0f}")
            lin_results[name] = result
        else:
            print(f"{name:<12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'N/A':>10}")
    
    # Non-linear data
    print("\n" + "=" * 92)
    print("NON-LINEAR DATA (5 variables: quadratic, sinusoidal, exponential relationships)")
    print("=" * 92)
    df_nl, true_nl = gen_nonlinear()  # FIXED: Get true matrix
    print(f"\n{'Algorithm':<12} {'F1 Score':>12} {'Precision':>12} {'Recall':>12} {'Edges':>10}")
    print("-" * 70)
    
    nl_results = {}
    for name, func in algs:
        result = safe_run(func, df_nl, true_nl)
        
        if result:
            print(f"{name:<12} {result['f1']:>12.3f} {result['prec']:>12.3f} {result['rec']:>12.3f} {result['edges']:>10.0f}")
            nl_results[name] = result
        else:
            print(f"{name:<12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'N/A':>10}")
    
    # Confounded non-linear data (most challenging)
    print("\n" + "=" * 92)
    print("CONFOUNDED NON-LINEAR DATA (5 variables: hidden confounding + colliders)")
    print("=" * 92)
    df_conf, true_conf = gen_confounded_nonlinear()
    print(f"\n{'Algorithm':<12} {'F1 Score':>12} {'Precision':>12} {'Recall':>12} {'Edges':>10}")
    print("-" * 70)
    
    conf_results = {}
    for name, func in algs:
        result = safe_run(func, df_conf, true_conf)
        
        if result:
            print(f"{name:<12} {result['f1']:>12.3f} {result['prec']:>12.3f} {result['rec']:>12.3f} {result['edges']:>10.0f}")
            conf_results[name] = result
        else:
            print(f"{name:<12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'N/A':>10}")
    
    # Summary
    print("\n" + "=" * 92)
    print("SUMMARY: Algorithm Performance Comparison")
    print("=" * 92)
    print(f"{'Algorithm':<12} {'Linear F1':>12} {'NL F1':>12} {'Conf F1':>12} {'Avg F1':>12} {'Best For':>15}")
    print("-" * 92)
    
    best_linear = max(lin_results.items(), key=lambda x: x[1]['f1']) if lin_results else ("N/A", {"f1": 0})
    best_nl = max(nl_results.items(), key=lambda x: x[1]['f1']) if nl_results else ("N/A", {"f1": 0})
    best_conf = max(conf_results.items(), key=lambda x: x[1]['f1']) if conf_results else ("N/A", {"f1": 0})
    
    for name, _ in algs:
        lin_f1 = lin_results.get(name, {}).get('f1', 0.0)
        nl_f1 = nl_results.get(name, {}).get('f1', 0.0)
        conf_f1 = conf_results.get(name, {}).get('f1', 0.0)
        
        # Calculate average (only for successful runs)
        scores = [s for s in [lin_f1, nl_f1, conf_f1] if s > 0]
        avg_f1 = sum(scores) / len(scores) if scores else 0.0
        
        # Determine what this algorithm is best for
        best_for = []
        if name == best_linear[0]: best_for.append("Linear")
        if name == best_nl[0]: best_for.append("NonLin")
        if name == best_conf[0]: best_for.append("Confnd")
        best_str = ",".join(best_for) if best_for else "-"
        
        print(f"{name:<12} {lin_f1:>12.3f} {nl_f1:>12.3f} {conf_f1:>12.3f} {avg_f1:>12.3f} {best_str:>15}")
    
    print("\n" + "=" * 92)
    print("CONCLUSION:")
    print("=" * 92)
    print(f"Best Overall Algorithm: {max([(name, sum([lin_results.get(name, {}).get('f1', 0), nl_results.get(name, {}).get('f1', 0), conf_results.get(name, {}).get('f1', 0)])) for name, _ in algs], key=lambda x: x[1])[0]}")
    print(f"Best for Linear Data: {best_linear[0]} (F1: {best_linear[1]['f1']:.3f})")
    print(f"Best for Non-Linear Data: {best_nl[0]} (F1: {best_nl[1]['f1']:.3f})")
    print(f"Best for Confounded Data: {best_conf[0]} (F1: {best_conf[1]['f1']:.3f})")
    print("\nAlgorithm characteristics:")
    print("- PC/CPC: Fast, good for sparse networks")
    print("- FCI/RFCI/CFCI: Handle latent confounders, slower")
    print("- GFCI: Hybrid score+constraint, good balance")
    print("- FCI-Max: Maximal orientation, most edges")
    print("=" * 92)


if __name__ == "__main__":
    main()