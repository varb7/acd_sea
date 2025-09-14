#!/usr/bin/env python3
"""
Minimal sanity test for Tetrad FGES/RFCI wrappers.

Usage (PowerShell):
  D:\acd_sea\causenv\Scripts\python.exe -m inference_pipeline.tetrad_sanity_test
"""

import os
import numpy as np
import pandas as pd

# Support both `python -m inference_pipeline.tetrad_sanity_test` and direct script run
try:
    from .tetrad_fges import run_fges  # type: ignore
    from .tetrad_rfci import run_rfci  # type: ignore
except ImportError:
    from inference_pipeline.tetrad_fges import run_fges  # type: ignore
    from inference_pipeline.tetrad_rfci import run_rfci  # type: ignore


def main() -> None:
    np.random.seed(0)
    n = 400

    # Simple structure: x -> y, z independent
    x = np.random.normal(size=n)
    y = 2.0 * x + np.random.normal(size=n)
    z = np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    print("TETRAD_JAR=", os.getenv("TETRAD_JAR"))

    print("\n[FGES] Running…")
    try:
        adj = run_fges(df)
        print("FGES edges:", int(np.sum(adj)))
        print(adj)
    except Exception as e:
        print("FGES error:", e)

    print("\n[RFCI] Running…")
    try:
        adj2 = run_rfci(df)
        print("RFCI edges:", int(np.sum(adj2)))
        print(adj2)
    except Exception as e:
        print("RFCI error:", e)


if __name__ == "__main__":
    main()


