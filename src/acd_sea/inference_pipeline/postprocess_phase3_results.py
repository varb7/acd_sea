"""
Augment an existing experiment results CSV with Phase 3 dataset metadata.

Reads each dataset directory's *_meta.pkl to recover root distribution and noise
parameters, computes qualitative intensity tags, and appends the info as new
columns. This avoids rerunning causal discovery experiments.
"""

from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    meta_files = sorted(dataset_dir.glob("*_meta.pkl"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.pkl in {dataset_dir}")
    with open(meta_files[0], "rb") as f:
        return pickle.load(f)


def _parse_params(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _root_variation_tag(meta: Dict[str, Any]) -> Dict[str, Any]:
    dtype = meta.get("root_distribution_type", "unknown")
    params = _parse_params(meta.get("root_distribution_params", {}))

    variation = "unknown"
    mean_bias = "unknown"

    if dtype == "normal":
        std = _to_float(params.get("std"), 1.0)
        mean = _to_float(params.get("mean"), 0.0)
        variation = "low" if std <= 1.0 else "high"
        mean_bias = "positive" if mean > 0.5 else "negative" if mean < -0.5 else "neutral"
    elif dtype == "uniform":
        low = _to_float(params.get("low"), 0.0)
        high = _to_float(params.get("high"), 1.0)
        span = abs(high - low)
        center = (high + low) / 2
        variation = "low" if span <= 6 else "high"
        mean_bias = "positive" if center > 0.5 else "negative" if center < -0.5 else "neutral"
    elif dtype == "exponential":
        scale = _to_float(params.get("scale"), 1.0)
        variation = "low" if scale <= 1.0 else "high"
        mean_bias = "positive"
    elif dtype == "beta":
        a = _to_float(params.get("a"), 1.0)
        b = _to_float(params.get("b"), 1.0)
        variation = "low" if (a + b) >= 4 else "high"
        if abs(a - b) <= 0.5:
            mean_bias = "neutral"
        else:
            mean_bias = "positive" if a > b else "negative"

    return {
        "root_distribution_type": dtype,
        "root_distribution_params": params,
        "root_variation_level": variation,
        "root_mean_bias": mean_bias,
    }


def _noise_variation_tag(meta: Dict[str, Any]) -> Dict[str, Any]:
    dtype = meta.get("noise_type", "unknown")
    params = _parse_params(meta.get("noise_params", {}))
    intensity = "unknown"

    if dtype == "normal":
        std = _to_float(params.get("std"), 1.0)
        intensity = "low" if std <= 1.0 else "high"
    elif dtype == "uniform":
        low = _to_float(params.get("low"), -1.0)
        high = _to_float(params.get("high"), 1.0)
        span = abs(high - low)
        intensity = "low" if span <= 2 else "high"
    elif dtype == "exponential":
        scale = _to_float(params.get("scale"), 1.0)
        intensity = "low" if scale <= 1.0 else "high"
    elif dtype == "beta":
        a = _to_float(params.get("a"), 1.0)
        b = _to_float(params.get("b"), 1.0)
        intensity = "low" if (a + b) >= 7 else "high"

    return {
        "noise_type": dtype,
        "noise_params": params,
        "noise_intensity_level": intensity,
    }


def augment_results(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    if "dataset_dir" not in df.columns:
        raise ValueError("Results file must contain dataset_dir column")

    meta_cache: Dict[str, Dict[str, Any]] = {}
    for ds in sorted(df["dataset_dir"].dropna().unique()):
        dataset_path = Path(ds)
        if not dataset_path.exists():
            dataset_path = Path(input_csv.parent, ds)
        try:
            meta = _load_metadata(dataset_path)
        except Exception as exc:
            print(f"[WARN] Skipping metadata for {ds}: {exc}")
            continue
        meta_cache[ds] = {
            **_root_variation_tag(meta),
            **_noise_variation_tag(meta),
        }

    def _lookup(ds: str, key: str, default: Any = None) -> Any:
        entry = meta_cache.get(ds)
        if not entry:
            return default
        value = entry.get(key, default)
        if isinstance(value, dict):
            return value
        return value

    df["root_distribution_type"] = df["dataset_dir"].apply(
        lambda x: _lookup(x, "root_distribution_type")
    )
    df["root_distribution_params"] = df["dataset_dir"].apply(
        lambda x: _lookup(x, "root_distribution_params", {})
    )
    df["root_variation_level"] = df["dataset_dir"].apply(
        lambda x: _lookup(x, "root_variation_level")
    )
    df["root_mean_bias"] = df["dataset_dir"].apply(
        lambda x: _lookup(x, "root_mean_bias")
    )
    df["noise_type"] = df["dataset_dir"].apply(lambda x: _lookup(x, "noise_type"))
    df["noise_params"] = df["dataset_dir"].apply(lambda x: _lookup(x, "noise_params", {}))
    df["noise_intensity_level"] = df["dataset_dir"].apply(
        lambda x: _lookup(x, "noise_intensity_level")
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote augmented results -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment Phase 3 results with metadata")
    parser.add_argument("--input", required=True, help="Path to experiment_results_phase3.csv")
    parser.add_argument("--output", required=True, help="Path to write augmented CSV")
    args = parser.parse_args()

    augment_results(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

