"""
Quick CSuite pattern visualizer using scdg's built-in plot_graph().

Usage examples:
  python visualize_csuite.py --pattern chain --num-nodes 5 --seed 42
  python visualize_csuite.py --pattern collider --num-nodes 4
"""

import argparse
import sys

import networkx as nx

from generator.csuite2 import build_graph_from_pattern
from scdg import CausalDataGenerator


def visualize(pattern: str, num_nodes: int, seed: int) -> None:
    # Build the CSuite graph (G) and metadata (string node names: 'a','b',...)
    G, meta = build_graph_from_pattern(pattern, num_nodes)

    # Prepare scdg and attach graph
    cdg = CausalDataGenerator(num_samples=100, seed=seed)
    cdg.G = G
    cdg.root_nodes = set(meta["root_nodes"])  # show roots in the legend

    # Optionally set simple root distributions so labels look informative
    root_distributions = {r: {"dist": "normal", "mean": 0.0, "std": 1.0} for r in meta["root_nodes"]}
    cdg.set_root_distributions(root_distributions)

    # Plot using scdg's helper
    cdg.plot_graph()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize CSuite DAG patterns")
    parser.add_argument("--pattern", type=str, required=True,
                        choices=[
                            "chain", "collider", "backdoor",
                            "mixed_confounding", "weak_arrow", "large_backdoor"
                        ],
                        help="CSuite pattern to visualize")
    parser.add_argument("--num-nodes", type=int, default=5,
                        help="Number of nodes (2-10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        visualize(args.pattern, args.num_nodes, args.seed)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


