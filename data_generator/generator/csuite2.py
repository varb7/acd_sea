"""
CSuite v2 generator: builds pattern DAGs and generates data using scdg public API
without overwriting the graph. Supports up to 10 nodes per dataset.
"""

from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from scdg import CausalDataGenerator


# ----------------------------- Naming helpers -----------------------------
def idx_to_name(i: int) -> str:
    if i < 26:
        return chr(97 + i)
    return f"{chr(97 + (i // 26) - 1)}{(i % 26) + 1}"


def map_edges(edges_idx: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
    return [(idx_to_name(u), idx_to_name(v)) for (u, v) in edges_idx]


def map_nodes(nodes_idx: List[int]) -> List[str]:
    return [idx_to_name(i) for i in nodes_idx]


# ------------------------------ Patterns ----------------------------------
def pattern_chain(n: int) -> Dict:
    return {
        "edges": [(i, i + 1) for i in range(n - 1)],
        "root_nodes": [0],
        "leaf_nodes": [n - 1],
        "equation_type": "linear",
        "variable_types": {i: "continuous" for i in range(n)},
    }


def pattern_collider(n: int) -> Dict:
    if n < 3:
        raise ValueError("collider requires at least 3 nodes")
    edges = [(0, 1), (2, 1)]
    if n > 3:
        for i in range(3, n):
            edges.append((1, i))
    return {
        "edges": edges,
        "root_nodes": [0, 2] if n == 3 else [0, 2] + list(range(3, n)),
        "leaf_nodes": [1] if n == 3 else list(range(3, n)),
        "equation_type": "linear",
        "variable_types": {i: "continuous" for i in range(n)},
    }


def pattern_backdoor(n: int) -> Dict:
    if n < 3:
        raise ValueError("backdoor requires at least 3 nodes")
    edges = [(0, 1), (2, 1)]
    if n > 3:
        for i in range(3, n):
            edges.extend([(i, 0), (i, 2)])
    return {
        "edges": edges,
        "root_nodes": [2] if n == 3 else [2] + list(range(3, n)),
        "leaf_nodes": [1],
        "equation_type": "linear",
        "variable_types": {i: "continuous" for i in range(n)},
    }


def pattern_mixed_confounding(n: int) -> Dict:
    if n < 4:
        raise ValueError("mixed_confounding requires at least 4 nodes")
    edges = [(0, 1)]
    for i in range(2, n):
        edges.extend([(i, 0), (i, 1)])
    variable_types: Dict[int, str] = {}
    for i in range(n):
        if i in [0, 1]:
            variable_types[i] = "discrete" if i == 0 else "continuous"
        else:
            variable_types[i] = "discrete" if i % 2 == 0 else "continuous"
    return {
        "edges": edges,
        "root_nodes": list(range(2, n)),
        "leaf_nodes": [1],
        "equation_type": "non_linear",
        "variable_types": variable_types,
    }


def pattern_weak_arrow(n: int) -> Dict:
    if n < 3:
        raise ValueError("weak_arrow requires at least 3 nodes")
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 3:
        edges.append((0, n - 1))
    return {
        "edges": edges,
        "root_nodes": [0],
        "leaf_nodes": [n - 1],
        "equation_type": "linear",
        "variable_types": {i: "continuous" for i in range(n)},
    }


def pattern_large_backdoor(n: int) -> Dict:
    if n < 4:
        raise ValueError("large_backdoor requires at least 4 nodes")
    edges = [(0, 1)]
    for i in range(2, n):
        edges.extend([(i, 0), (i, 1)])
    return {
        "edges": edges,
        "root_nodes": list(range(2, n)),
        "leaf_nodes": [1],
        "equation_type": "linear",
        "variable_types": {i: "continuous" for i in range(n)},
    }


PATTERN_BUILDERS = {
    "chain": pattern_chain,
    "collider": pattern_collider,
    "backdoor": pattern_backdoor,
    "mixed_confounding": pattern_mixed_confounding,
    "weak_arrow": pattern_weak_arrow,
    "large_backdoor": pattern_large_backdoor,
}


# -------------------------- Graph + SCDG pipeline --------------------------
def build_graph_from_pattern(pattern: str, num_nodes: int) -> Tuple[nx.DiGraph, Dict]:
    if num_nodes < 2 or num_nodes > 10:
        raise ValueError("num_nodes must be between 2 and 10 for CSuite v2")
    if pattern not in PATTERN_BUILDERS:
        raise ValueError(f"Unknown pattern: {pattern}")

    pat_cfg = PATTERN_BUILDERS[pattern](num_nodes)

    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(idx_to_name(i))
    for u, v in map_edges(pat_cfg["edges"]):
        G.add_edge(u, v)

    metadata = {
        "pattern": pattern,
        "num_nodes": num_nodes,
        "num_edges": len(pat_cfg["edges"]),
        "root_nodes": map_nodes(pat_cfg["root_nodes"]),
        "leaf_nodes": map_nodes(pat_cfg["leaf_nodes"]),
        "equation_type": pat_cfg["equation_type"],
        "variable_types": {idx_to_name(k): v for k, v in pat_cfg["variable_types"].items()},
    }
    return G, metadata


def generate_data_with_scdg(G: nx.DiGraph,
                            root_nodes: List[str],
                            equation_type: str,
                            num_samples: int,
                            seed: int,
                            *,
                            root_categorical: bool = False,
                            root_num_classes: int = 3,
                            root_normal_mean: float = 0.0,
                            root_normal_std: float = 1.0,
                            nonroot_categorical_pct: float = 0.0,
                            nonroot_categorical_num_classes: int = 3) -> pd.DataFrame:
    cdg = CausalDataGenerator(num_samples=num_samples, seed=seed)
    cdg.G = G.copy()
    cdg.root_nodes = set(root_nodes)

    # Build root distributions
    root_distributions: Dict[str, Dict] = {}
    for r in root_nodes:
        if root_categorical:
            root_distributions[r] = {"dist": "categorical", "num_classes": int(root_num_classes)}
        else:
            root_distributions[r] = {"dist": "normal", "mean": float(root_normal_mean), "std": float(root_normal_std)}
    cdg.set_root_distributions(root_distributions)

    # Optionally flip a percentage of non-root nodes to nominal (categorical)
    if nonroot_categorical_pct and nonroot_categorical_pct > 0:
        non_roots = [n for n in G.nodes if n not in root_nodes]
        k = max(1, int(len(non_roots) * float(nonroot_categorical_pct)))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(non_roots, size=min(k, len(non_roots)), replace=False)
        nominal_map: Dict[str, Dict] = {}
        for node in chosen:
            parents = list(G.predecessors(node))
            parent_input = parents[0] if parents else (root_nodes[0] if root_nodes else None)
            if parent_input is None:
                continue
            nominal_map[node] = {"input": parent_input, "num_classes": int(nonroot_categorical_num_classes)}
        if nominal_map:
            cdg.set_nominal_nodes(nominal_map)

    # Assign equations
    cdg.assign_equations_to_graph_nodes(equation_type=equation_type)
    df = cdg.generate_data()
    return df


def assign_simple_stations(G: nx.DiGraph, num_stations: int = 3) -> Dict:
    topo = list(nx.topological_sort(G))
    blocks = np.array_split(topo, num_stations)
    station_blocks = [list(b) for b in blocks if len(b) > 0]
    station_names = [f"Station{i+1}" for i in range(len(station_blocks))]
    station_map: Dict[str, str] = {}
    for sname, nodes in zip(station_names, station_blocks):
        for n in nodes:
            station_map[n] = sname
    return {
        "temporal_order": [n for b in station_blocks for n in b],
        "station_blocks": station_blocks,
        "station_names": station_names,
        "station_map": station_map,
    }


def validate_df(df: pd.DataFrame) -> None:
    if df.isnull().any().any():
        raise ValueError("NaN detected in generated data")
    num = df.select_dtypes(include=[np.number])
    if not num.empty and np.isinf(num).any().any():
        raise ValueError("Inf detected in generated data")


def generate_csuite2_dataset(config: Dict) -> Tuple[pd.DataFrame, Dict, nx.DiGraph]:
    pattern = config.get("pattern")
    num_nodes = int(config.get("num_nodes"))
    num_samples = int(config.get("num_samples", 1000))
    seed = int(config.get("seed", 42))
    num_stations = int(config.get("num_stations", 3))

    G, meta = build_graph_from_pattern(pattern, num_nodes)

    # Allow overriding equation type
    equation_type = str(config.get("equation_type", meta["equation_type"]))

    # Root controls
    root_categorical = bool(config.get("root_categorical", False))
    root_num_classes = int(config.get("root_num_classes", 3))
    root_normal_mean = float(config.get("root_normal_mean", 0.0))
    root_normal_std = float(config.get("root_normal_std", 1.0))

    # Non-root categorical controls
    nonroot_categorical_pct = float(config.get("nonroot_categorical_pct", 0.0))
    nonroot_categorical_num_classes = int(config.get("nonroot_categorical_num_classes", 3))

    df = generate_data_with_scdg(
        G,
        meta["root_nodes"],
        equation_type,
        num_samples,
        seed,
        root_categorical=root_categorical,
        root_num_classes=root_num_classes,
        root_normal_mean=root_normal_mean,
        root_normal_std=root_normal_std,
        nonroot_categorical_pct=nonroot_categorical_pct,
        nonroot_categorical_num_classes=nonroot_categorical_num_classes,
    )
    validate_df(df)

    st = assign_simple_stations(G, num_stations=num_stations)
    df = df[st["temporal_order"]]

    metadata = {**meta, **st, "seed": seed, "num_samples": num_samples}
    return df, metadata, G


