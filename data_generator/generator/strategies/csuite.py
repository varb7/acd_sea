"""
CSuite generation strategy wrapper.

This module exposes a strategy-level function for generating CSuite-style
datasets, delegating to the existing CSuite generator implementation.
"""

from typing import List, Tuple, Optional

from ..csuite_dag_generator import generate_csuite_meta_dataset as _generate


def generate_csuite_meta_dataset(
    patterns: Optional[List[str]] = None,
    num_nodes_range: Tuple[int, int] = (2, 5),
    num_samples: int = 1000,
    output_dir: str = "csuite_benchmark/datasets",
    seed: int = 42,
) -> None:
    """Strategy entrypoint for CSuite-style dataset generation.

    Parameters mirror the underlying generator for compatibility.
    """
    _generate(
        patterns=patterns,
        num_nodes_range=num_nodes_range,
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
    )


