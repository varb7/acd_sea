#!/usr/bin/env python3
"""
Shared module for adaptive conditional independence test selection in Tetrad algorithms.
Supports intelligent switching between parametric CI tests based on data diagnostics.
"""

import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype

# Optional SciPy for diagnostics
try:
    from scipy.stats import spearmanr, pearsonr, jarque_bera
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class TetradCITestSelector:
    """
    Adaptive conditional independence test selector for Tetrad constraint-based algorithms.
    
    Provides:
    - Data type detection (discrete, continuous, mixed)
    - Global diagnostics (linearity, Gaussianity assessment)
    - Intelligent test selection (Parametric only)
    - Test tracking for experimental comparison
    
    Parameters:
        alpha: float = 0.01              # CI test significance level
        linear_gap_threshold: float = 0.08   # Max |Spearman|-|Pearson| for linearity
        gaussian_p_threshold: float = 0.05   # JB p-value cutoff for Gaussianity
        max_pairs_for_diag: int = 50         # Pairs to sample for linearity check
        max_parents_for_diag: int = 5        # Predictors for normality check
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        linear_gap_threshold: float = 0.08,
        gaussian_p_threshold: float = 0.05,
        max_pairs_for_diag: int = 50,
        max_parents_for_diag: int = 5,
    ):
        self.alpha = alpha
        self.linear_gap_threshold = linear_gap_threshold
        self.gaussian_p_threshold = gaussian_p_threshold
        self.max_pairs_for_diag = max_pairs_for_diag
        self.max_parents_for_diag = max_parents_for_diag
        
        self._global_diagnostics: Dict[str, Any] = {}
        self._selected_test_name: Optional[str] = None
    
    def detect_data_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Return (categorical_cols, continuous_cols)."""
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]):
                cats.append(c)
            elif is_float_dtype(df[c]):
                cont.append(c)
            else:
                try:
                    df[c] = df[c].astype("int64")
                    cats.append(c)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    cont.append(c)
        return cats, cont
    
    def assess_global_diagnostics(
        self, df: pd.DataFrame, cats: List[str], cont: List[str]
    ) -> Dict[str, Any]:
        """Compute diagnostics with EITHER violation required for 'suspect' regime."""
        n_cont = len(cont)
        diagnostics = {
            "regime": None,
            "n_cont": n_cont,
            "n_disc": len(cats),
            "cont_linear_frac": np.nan,
            "cont_gauss_frac": np.nan,
        }
        
        if n_cont == 0:
            diagnostics["regime"] = "discrete"
            return diagnostics
        
        if not SCIPY_AVAILABLE or n_cont == 1:
            diagnostics["regime"] = "continuous_unknown" if not cats else "mixed_clg_unknown"
            return diagnostics
        
        linear_frac = self._assess_linearity(df, cont)
        gauss_frac = self._assess_gaussianity(df, cont)
        diagnostics["cont_linear_frac"] = linear_frac
        diagnostics["cont_gauss_frac"] = gauss_frac
        
        # REQUIRE EITHER VIOLATION for 'suspect' regime
        linear_ok = linear_frac >= 0.80
        gauss_ok = gauss_frac >= 0.80
        either_ok = linear_ok or gauss_ok
        
        if not cats:
            diagnostics["regime"] = "continuous_linear_ok" if either_ok else "continuous_suspect"
        else:
            diagnostics["regime"] = "mixed_clg_ok" if either_ok else "mixed_clg_suspect"
        
        return diagnostics
    
    def _assess_linearity(self, df: pd.DataFrame, cont: List[str]) -> float:
        """Estimate fraction of continuous pairs that are approximately linear using polynomial test."""
        if not SCIPY_AVAILABLE or len(cont) < 2:
            return np.nan
        
        pairs = [(cont[i], cont[j]) for i in range(len(cont)) for j in range(i + 1, len(cont))]
        
        RNG = np.random.default_rng(42)
        if len(pairs) > self.max_pairs_for_diag:
            pairs = list(RNG.choice(pairs, size=self.max_pairs_for_diag, replace=False))
        
        linear_flags = []
        for a, b in pairs:
            x, y = df[a].to_numpy(), df[b].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 20:
                continue
            x, y = x[mask], y[mask]
            
            # Fit linear model
            x_mean, y_mean = np.mean(x), np.mean(y)
            x_std, y_std = np.std(x), np.std(y)
            if x_std < 1e-10 or y_std < 1e-10:
                continue
                
            x_norm = (x - x_mean) / x_std
            y_norm = (y - y_mean) / y_std
            
            # Linear R²
            beta_linear = np.sum(x_norm * y_norm) / len(x_norm)
            y_pred_linear = beta_linear * x_norm
            ss_res_linear = np.sum((y_norm - y_pred_linear)**2)
            ss_tot = np.sum(y_norm**2)  
            r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
            
            # Polynomial (degree 2) R²
            try:
                X_poly = np.column_stack([x_norm, x_norm**2])
                beta_poly = np.linalg.lstsq(X_poly, y_norm, rcond=None)[0]
                y_pred_poly = X_poly @ beta_poly
                ss_res_poly = np.sum((y_norm - y_pred_poly)**2)
                r2_poly = 1 - (ss_res_poly / ss_tot) if ss_tot > 0 else 0
                
                # If polynomial is MUCH better, relationship is nonlinear
                # Threshold: polynomial explains >10% more variance than linear
                improvement = r2_poly - r2_linear
                is_linear = improvement < 0.10
                linear_flags.append(is_linear)
            except Exception:
                continue
        
        return float(np.mean(linear_flags)) if linear_flags else np.nan
    
    def _assess_gaussianity(self, df: pd.DataFrame, cont: List[str]) -> float:
        """Estimate fraction of continuous vars with Gaussian residuals."""
        if not SCIPY_AVAILABLE or len(cont) < 2:
            return np.nan
        
        cont_df = df[cont].astype(float)
        corr = cont_df.corr().abs()
        
        gauss_flags = []
        for col in cont:
            others = corr[col].drop(index=col).sort_values(ascending=False).index.tolist()
            parents = others[: self.max_parents_for_diag]
            if not parents:
                continue
            
            y = cont_df[col].to_numpy()
            X = cont_df[parents].to_numpy()
            X = np.column_stack([np.ones(len(X)), X])
            
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                jb_stat, jb_p = jarque_bera(resid)
                gauss_flags.append(jb_p > self.gaussian_p_threshold)
            except Exception:
                continue
        
        return float(np.mean(gauss_flags)) if gauss_flags else np.nan
    
    def create_independence_test(self, tetrad_data, cats, cont, test_module, ptt_module=None):
        """Create CI test based on data type and diagnostics."""
        regime = self._global_diagnostics.get("regime", None)
        return self._create_parametric_test(tetrad_data, cats, cont, regime, test_module)
    
    def _create_parametric_test(self, tetrad_data, cats, cont, regime, test_module):
        """Traditional parametric test selection with warnings."""
        diag = self._global_diagnostics
        
        if cats and cont:
            if regime == "mixed_clg_suspect":
                warnings.warn(
                    f"[TetradCI] Mixed data with violations (linear={diag.get('cont_linear_frac', np.nan):.2f}, "
                    f"gauss={diag.get('cont_gauss_frac', np.nan):.2f}). Parametric test may be miscalibrated.",
                    RuntimeWarning
                )
            self._selected_test_name = "CG-LRT"
            return test_module.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats:
            self._selected_test_name = "Chi-Square"
            return test_module.IndTestChiSquare(tetrad_data, self.alpha)
        else:
            if regime == "continuous_suspect":
                warnings.warn(
                    f"[TetradCI] Continuous data with violations (linear={diag.get('cont_linear_frac', np.nan):.2f}, "
                    f"gauss={diag.get('cont_gauss_frac', np.nan):.2f}). Parametric test may be miscalibrated.",
                    RuntimeWarning
                )
            self._selected_test_name = "Fisher-Z"
            return test_module.IndTestFisherZ(tetrad_data, self.alpha)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics including selected test."""
        diag = dict(self._global_diagnostics)
        diag["selected_test"] = self._selected_test_name
        return diag
    
    def set_diagnostics(self, diagnostics: Dict[str, Any]):
        """Store diagnostics."""
        self._global_diagnostics = diagnostics
