# run_ci_benchmarks.py  (CSV version, 3 nodes, n=1000, R=100)
import math, time, numpy as np, pandas as pd
import logging
from collections import namedtuple
from typing import List, Optional

# --- core CI tests from causal-learn ---
from causallearn.utils.cit import CIT

# --- CG-LRT pieces ---
import statsmodels.api as sm
from scipy.stats import chi2

# ====== IMPORT YOUR GENERATOR (UNCHANGED) ======
try:
    from scdg import CausalDataGenerator  # adjust if your class lives elsewhere
except ImportError:
    raise

RNG = np.random.default_rng(7)

# --------------- logging setup ---------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def _progress_str(done: int, total: int) -> str:
    total = max(total, 1)
    pct = 100.0 * float(done) / float(total)
    return f"{done}/{total} ({pct:0.1f}%)"

# --------------- helpers ---------------
def _is_categorical_node(cdg, name: str) -> bool:
    return cdg._is_categorical_node(name)

def _all_continuous(cdg, vars: List[str]) -> bool:
    return all(not _is_categorical_node(cdg, v) for v in vars)

def _all_categorical(cdg, vars: List[str]) -> bool:
    return all(_is_categorical_node(cdg, v) for v in vars)

def _xy_cont_z_cat(cdg, x: str, y: str, S: List[str]) -> bool:
    return (not _is_categorical_node(cdg, x)) and (not _is_categorical_node(cdg, y)) and all(_is_categorical_node(cdg, s) for s in S)

# --------------- CG-LRT (continuous + CLG stratified) ---------------
def cg_lrt_pvalue(df: pd.DataFrame, x: str, y: str, S: Optional[List[str]] = None) -> float:
    S = S or []
    # If no conditioning, single stratum
    if len(S) == 0:
        return _lrt_single_stratum(df, x, y, S)
    # Split S into categorical vs continuous
    cat_S, cont_S = [], []
    for s in S:
        if pd.api.types.is_numeric_dtype(df[s]) and df[s].nunique() > 10:
            cont_S.append(s)
        else:
            cat_S.append(s)
    if not cat_S:
        return _lrt_single_stratum(df, x, y, cont_S)
    # Stratify on categorical S; include continuous S within each stratum
    L_total, df_total, any_used = 0.0, 0, False
    for _, sub in df.groupby(cat_S, observed=True, sort=False):
        if len(sub) < 8:
            continue
        _, (L, df_add) = _lrt_details(sub, x, y, cont_S)
        L_total += L
        df_total += df_add
        any_used = True
    if not any_used or df_total <= 0:
        return _lrt_single_stratum(df, x, y, S)
    return float(chi2.sf(L_total, df_total))

def _lrt_single_stratum(df: pd.DataFrame, x: str, y: str, S: List[str]) -> float:
    pval, _ = _lrt_details(df, x, y, S)
    return pval

def _lrt_details(df: pd.DataFrame, x: str, y: str, S: List[str]):
    """Run two nested OLS fits and return (pvalue, (Λ, df))."""
    # Null: x ~ S ; Full: x ~ S + y
    y_null = df[x].astype(float).values
    X_null = pd.DataFrame({s: df[s].astype(float).values for s in S}) if S else pd.DataFrame(index=df.index)
    X_null = sm.add_constant(X_null, has_constant='add')
    res0 = sm.OLS(y_null, X_null).fit()

    X_full = X_null.copy()
    X_full[y] = df[y].astype(float).values
    res1 = sm.OLS(y_null, X_full).fit()

    L = -2.0 * (res0.llf - res1.llf)
    df_add = res1.df_model - res0.df_model  # number of added params (usually 1)
    if df_add <= 0:
        df_add = 1.0
    p = float(chi2.sf(L, df_add))
    return p, (L, int(df_add))

# --------------- causal-learn CI runner ---------------
def run_cit(df: pd.DataFrame, test_name: str, x: str, y: str, S: Optional[List[str]] = None, **kwargs):
    S = S or []
    idx = {col: i for i, col in enumerate(df.columns)}
    cit = CIT(df.to_numpy(dtype=float), test_name, **kwargs)
    p = float(cit(idx[x], idx[y], [idx[s] for s in S]))
    return p

# --------------- FAST KCI auto-detect ---------------
def try_fastkci(df: pd.DataFrame, x: str, y: str, S: Optional[List[str]] = None):
    S = S or []
    try:
        return run_cit(df, "fastkci", x, y, S)
    except Exception:
        pass
    try:
        return run_cit(df, "rcit", x, y, S)
    except Exception:
        return None  # skipped

# --------------- dataset builders using YOUR generator ---------------
def build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2, equation_type="linear",
                  cat_roots=False, nominal_pct=0.0, seed=None):
    cdg = CausalDataGenerator(num_samples=n, seed=seed)
    cat_root_flag = "all" if cat_roots else None
    df = cdg.generate_data_pipeline(
        total_nodes=total_nodes,
        root_nodes=root_nodes,
        edges=edges,
        equation_type=equation_type,
        categorical_percentage=nominal_pct,
        categorical_root_nodes=cat_root_flag
    )
    return cdg, df

Motif = namedtuple("Motif", ["x", "y", "z", "kind"])  # kind: fork_null, chain_null, collider_alt, uncond_alt, uncond_null

def find_fork(G) -> Optional[Motif]:
    for z in G.nodes:
        ch = list(G.successors(z))
        if len(ch) >= 2:
            return Motif(ch[0], ch[1], z, "fork_null")  # X ⟂ Y | Z
    return None

def find_chain(G) -> Optional[Motif]:
    for z in G.nodes:
        preds = list(G.predecessors(z))
        succs = list(G.successors(z))
        if preds and succs and preds[0] != succs[0]:
            return Motif(preds[0], succs[0], z, "chain_null")  # X ⟂ Y | Z
    return None

def find_collider(G) -> Optional[Motif]:
    for z in G.nodes:
        preds = list(G.predecessors(z))
        if len(preds) >= 2:
            return Motif(preds[0], preds[1], z, "collider_alt")  # X ⟂/ Y | Z
    return None

def uncond_alt_from_chain(G) -> Optional[Motif]:
    m = find_chain(G)
    return Motif(m.x, m.y, None, "uncond_alt") if m else None

def uncond_null_pair(G) -> Optional[Motif]:
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            return Motif(nodes[i], nodes[j], None, "uncond_null")
    return None

def regenerate_until(cdg, finder, max_tries=25):
    for _ in range(max_tries):
        cdg.generate_random_graph(total_nodes=len(cdg.G.nodes), root_nodes=len(cdg.root_nodes), edges=len(cdg.G.edges))
        cdg.assign_equations_to_graph_nodes(equation_type="random")
        if not cdg.root_ranges:
            cdg.root_ranges = cdg._assign_random_distributions_to_root_nodes(list(cdg.root_nodes))
        df = cdg.generate_data()
        m = finder(cdg.G)
        if m:
            return m, df
    return None, cdg.data

# --------------- benchmark core ---------------
def evaluate_suite(suite_name, builder_fn, motif_finder, tests: List[str], reps=100, alpha=0.05,
                   filter_fn=None):
    rows = []
    total_tasks = reps * max(len(tests), 1)
    tasks_done = 0
    for r in range(reps):
        cdg, df = builder_fn()
        m, df = regenerate_until(cdg, motif_finder)
        if not m:
            tasks_done += len(tests)
            continue
        x, y, z, kind = m
        S = [z] if z is not None else []
        if filter_fn and not filter_fn(cdg, x, y, S):
            tasks_done += len(tests)
            continue

        for t in tests:
            try:
                t0 = time.perf_counter()
                if t == "cg_lrt":
                    p = cg_lrt_pvalue(df, x, y, S)
                elif t == "fastkci":
                    p = try_fastkci(df, x, y, S)
                    if p is None:
                        rows.append(dict(suite=suite_name, rep=r, test=t, x=x, y=y, S=tuple(S),
                                         motif=kind, pvalue=np.nan, runtime_s=np.nan, reject=np.nan, status="skipped"))
                        tasks_done += 1
                        continue
                else:
                    p = run_cit(df, t, x, y, S)
                dt = time.perf_counter() - t0
                rows.append(dict(suite=suite_name, rep=r, test=t, x=x, y=y, S=tuple(S),
                                 motif=kind, pvalue=float(p), runtime_s=dt, reject=int(float(p) < alpha),
                                 status="ok"))
                tasks_done += 1
            except Exception as e:
                rows.append(dict(suite=suite_name, rep=r, test=t, x=x, y=y, S=tuple(S),
                                 motif=kind, pvalue=np.nan, runtime_s=np.nan, reject=np.nan,
                                 status=f"error: {e}"))
                tasks_done += 1
    return pd.DataFrame(rows)

def summarize(per_eval: pd.DataFrame) -> pd.DataFrame:
    out = []
    if per_eval.empty:
        return pd.DataFrame()
    grp = per_eval.dropna(subset=["reject"]).groupby(["suite","test"])
    for (suite, test), g in grp:
        n = len(g)
        rate = g["reject"].mean()
        se = math.sqrt(max(rate*(1-rate), 1e-12)/n) if n>0 else np.nan
        lo, hi = rate - 1.96*se, rate + 1.96*se
        out.append(dict(
            suite=suite, test=test, n=n,
            reject_rate=rate, reject_rate_95ci_low=max(0.0, lo), reject_rate_95ci_high=min(1.0, hi),
            mean_runtime_s=g["runtime_s"].mean()
        ))
    return pd.DataFrame(out).sort_values(["suite","test"])

# --------------- main experiment ---------------
def main(output_prefix="ci_benchmark_results", reps=100, alpha=0.05):
    results = []

    # A) Fork (linear): X ⟂ Y | Z   → null for conditional tests
    def builder_fork_lin():
        # 3 nodes, 1 root (Z), 2 edges: Z->X, Z->Y
        return build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2,
                             equation_type="linear", cat_roots=False, nominal_pct=0.0,
                             seed=int(RNG.integers(1e9)))
    tests_A = ["fisherz", "cg_lrt", "kci", "fastkci"]
    dfA = evaluate_suite("Fork_null_X⟂Y|Z_linear", builder_fork_lin, find_fork,
                         tests_A, reps=reps, alpha=alpha,
                         filter_fn=lambda cdg,x,y,S: _all_continuous(cdg, [x,y]+S))
    results.append(dfA)

    # B) Fork (nonlinear): X ⟂ Y | Z  → still null; shows mis-spec for linear tests
    def builder_fork_nl():
        return build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2,
                             equation_type="non_linear", cat_roots=False, nominal_pct=0.0,
                             seed=int(RNG.integers(1e9)))
    tests_B = ["kci", "fastkci", "cg_lrt"]  # include cg_lrt to observe behavior under mis-spec
    dfB = evaluate_suite("Fork_null_X⟂Y|Z_nonlinear", builder_fork_nl, find_fork,
                         tests_B, reps=reps, alpha=alpha,
                         filter_fn=lambda cdg,x,y,S: _all_continuous(cdg, [x,y]+S))
    results.append(dfB)

    # C) Chain: X → Z → Y, unconditional alt (X and Y dependent)
    def builder_chain():
        # 3 nodes, 1 root, 2 edges form X->Z->Y
        return build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2,
                             equation_type="linear", cat_roots=False, nominal_pct=0.0,
                             seed=int(RNG.integers(1e9)))
    tests_C = ["fisherz", "cg_lrt", "kci", "fastkci"]
    dfC = evaluate_suite("Chain_alt_unconditional", builder_chain, uncond_alt_from_chain,
                         tests_C, reps=reps, alpha=alpha,
                         filter_fn=lambda cdg,x,y,S: _all_continuous(cdg, [x,y]+S))
    results.append(dfC)

    # D) Collider: X → Z ← Y, conditional alt (dependence when conditioning on Z)
    def builder_collider():
        # 3 nodes, 2 roots (X,Y), 2 edges X->Z, Y->Z
        return build_dataset(n=1000, total_nodes=3, root_nodes=2, edges=2,
                             equation_type="linear", cat_roots=False, nominal_pct=0.0,
                             seed=int(RNG.integers(1e9)))
    tests_D = ["fisherz", "cg_lrt", "kci", "fastkci"]
    dfD = evaluate_suite("Collider_alt_X!⟂Y|Z_linear", builder_collider, find_collider,
                         tests_D, reps=reps, alpha=alpha,
                         filter_fn=lambda cdg,x,y,S: _all_continuous(cdg, [x,y]+S))
    results.append(dfD)

    # E) Discrete: unconditional null and alt (χ² / G²)
    def builder_disc():
        # Make roots categorical and flip non-roots to nominal so X,Y are categorical in practice
        return build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2,
                             equation_type="random", cat_roots=True, nominal_pct=1.0,
                             seed=int(RNG.integers(1e9)))
    tests_E = ["chisq", "gsq"]
    dfE_null = evaluate_suite("Discrete_uncond_null", builder_disc, uncond_null_pair,
                              tests_E, reps=reps, alpha=alpha,
                              filter_fn=lambda cdg,x,y,S: _all_categorical(cdg, [x,y]+S))
    results.append(dfE_null)

    dfE_alt = evaluate_suite("Discrete_uncond_alt", builder_disc, uncond_alt_from_chain,
                             tests_E, reps=reps, alpha=alpha,
                             filter_fn=lambda cdg,x,y,S: _all_categorical(cdg, [x,y]+S))
    results.append(dfE_alt)

    # F) Mixed CLG: fork with categorical Z (null X ⟂ Y | Z). Compare CG-LRT (stratified), Fisher-Z, KCI.
    def builder_mixed():
        # 3 nodes, 1 root (Z), categorical Z, continuous X,Y
        return build_dataset(n=1000, total_nodes=3, root_nodes=1, edges=2,
                             equation_type="linear", cat_roots=True, nominal_pct=0.0,
                             seed=int(RNG.integers(1e9)))
    tests_F = ["cg_lrt", "fisherz", "kci", "fastkci"]
    dfF = evaluate_suite("Mixed_CLG_null_X⟂Y|Z", builder_mixed, find_fork,
                         tests_F, reps=reps, alpha=alpha,
                         filter_fn=lambda cdg,x,y,S: _xy_cont_z_cat(cdg, x, y, S))
    results.append(dfF)

    per_eval = pd.concat(results, ignore_index=True)
    summary = summarize(per_eval)

    # ---- write CSVs ----
    per_eval_csv = f"{output_prefix}_per_eval.csv"
    summary_csv = f"{output_prefix}_summary.csv"
    per_eval.to_csv(per_eval_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote: {per_eval_csv}")
    print(f"Wrote: {summary_csv}")
    return per_eval, summary

if __name__ == "__main__":
    # First-phase settings per your request
    main(output_prefix="ci_benchmark_results", reps=100, alpha=0.05)
