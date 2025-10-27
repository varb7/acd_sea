"""
Shared utility for building Tetrad Knowledge objects from prior knowledge.
Used by all Tetrad algorithms (RFCI, FGES, GFCI, CFCI, CPC, FCI, FCI-Max).
"""

from typing import Dict, Any, List, Optional
import jpype


def build_tetrad_knowledge(prior: Optional[Dict[str, Any]], columns: List[str]) -> Optional[object]:
    """
    Translate a Python dict into a Tetrad Knowledge object.
    
    Args:
        prior: Prior knowledge dictionary with these keys (all optional):
            - "required_edges" or "required": List of (source, target) tuples for required edges
            - "forbidden_edges" or "forbidden": List of (source, target) tuples for forbidden edges
            - "tier_ordering" or "tiers": List of tiers, each tier is a list of node names
            - "forbid_within_tier": bool, forbid edges within same tier
            - "only_next_tier": bool, only allow edges to next tier
        columns: List of column names in the dataset
    
    Returns:
        edu.cmu.tetrad.data.Knowledge() or None
    
    Example prior dict:
        prior = {
            "forbidden_edges": [("X", "Y"), ("A", "B")],
            "required_edges": [("X", "Z")],
            "tier_ordering": [
                ["T0_var1","T0_var2"],  # tier 0 (earliest)
                ["T1_var1"],            # tier 1
                ["Outcome"]             # tier 2 (latest)
            ],
            "forbid_within_tier": False,
            "only_next_tier": False,
        }
    """
    if prior is None:
        return None
    
    try:
        import edu.cmu.tetrad.data as data
        
        if not jpype.isJVMStarted():
            return None
        
        K = data.Knowledge()
        
        # 1. Required directed edges (must have A -> B)
        # Handle both "required" and "required_edges" keys for compatibility
        required = prior.get("required_edges", prior.get("required", []))
        for (src, dst) in required:
            if str(src) in columns and str(dst) in columns:
                K.setRequired(str(src), str(dst))
        
        # 2. Forbidden directed edges (cannot have A -> B)
        # Handle both "forbidden" and "forbidden_edges" keys for compatibility
        forbidden = prior.get("forbidden_edges", prior.get("forbidden", []))
        for (src, dst) in forbidden:
            if str(src) in columns and str(dst) in columns:
                K.setForbidden(str(src), str(dst))
        
        # 3. Tier / temporal ordering
        # Handle both "tiers" and "tier_ordering" keys for compatibility
        tiers = prior.get("tier_ordering", prior.get("tiers", []))
        for tier_idx, tier_vars in enumerate(tiers):
            for var in tier_vars:
                if str(var) in columns:
                    K.addToTier(tier_idx, str(var))
        
        # 4. Optional stricter tier rules
        if tiers:
            if prior.get("forbid_within_tier", False):
                # forbid any edges among variables in the SAME tier
                for tier_idx in range(len(tiers)):
                    K.setTierForbiddenWithin(tier_idx, True)
            
            if prior.get("only_next_tier", False):
                # allow edges only from tier k -> k+1 (no skipping)
                for tier_idx in range(len(tiers)):
                    K.setOnlyCanCauseNextTier(tier_idx, True)
        
        return K
    
    except Exception as e:
        print(f"[WARNING] Could not build Tetrad Knowledge object: {e}")
        return None

