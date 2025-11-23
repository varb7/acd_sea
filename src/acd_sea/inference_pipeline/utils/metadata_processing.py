"""
Utility module for processing and enriching dataset metadata.
Ported from postprocess_phase3_results.py.
"""

from typing import Any, Dict
import ast

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
        "noise_intensity_level": intensity,
    }


def enrich_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich metadata with derived metrics like variation levels and mean bias.
    """
    enriched = metadata.copy()
    
    # Add root variation tags
    enriched.update(_root_variation_tag(metadata))
    
    # Add noise variation tags
    enriched.update(_noise_variation_tag(metadata))
    
    return enriched
