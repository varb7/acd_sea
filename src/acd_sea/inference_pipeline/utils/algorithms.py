import numpy as np
import pandas as pd
import networkx as nx
import time
from typing import Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import metrics (required by pipeline)
from castle.metrics import MetricsDAG

# Note: We only use PyTetrad algorithms in this pipeline now


def pag_to_directed_adjacency(pag_adj: np.ndarray) -> np.ndarray:
    """
    Convert PAG adjacency {-1, 0, 1, 2} to directed adjacency {0, 1} for metrics comparison.
    
    PAG format:
        -1: Backward edge (a ← b), meaning b → a
         0: No edge
         1: Undirected edge (a — b)
         2: Forward edge (a → b)
    
    Output:
        Binary adjacency where adj[i,j] = 1 means i → j
        
    Conversion rules:
        - Value 2 (forward): Keep as edge (i → j)
        - Value -1 (backward): This is j → i, so DON'T mark i → j
        - Value 1 (undirected): Mark as edge in both directions for skeleton comparison
        - Value 0: No edge
    """
    directed_adj = np.zeros_like(pag_adj, dtype=int)
    
    # Forward edges (value 2) -> mark as directed
    directed_adj[pag_adj == 2] = 1
    
    # Undirected edges (value 1) -> mark symmetrically for skeleton
    directed_adj[pag_adj == 1] = 1
    
    # Backward edges (-1) are NOT marked because they represent the reverse direction
    # The corresponding forward edge (value 2) at the transposed position handles it
    
    return directed_adj


def pag_to_skeleton(pag_adj: np.ndarray) -> np.ndarray:
    """
    Extract skeleton (undirected adjacency) from PAG format.
    Any non-zero value indicates an edge exists.
    
    Returns symmetric binary matrix where adj[i,j] = adj[j,i] = 1 if any edge exists.
    """
    skeleton = (pag_adj != 0).astype(int)
    # Make symmetric for skeleton comparison
    skeleton = np.maximum(skeleton, skeleton.T)
    return skeleton



# ReX algorithm (currently disabled)
REX_AVAILABLE = False

# Python-only alternatives to FGES (currently disabled)

# Tetrad algorithms (robust import for core modules only)
TETRAD_AVAILABLE = False
try:
    from acd_sea.inference_pipeline.tetrad_rfci import TetradRFCI
    from acd_sea.inference_pipeline.tetrad_fges import TetradFGES
    TETRAD_AVAILABLE = True
except ImportError:
    TETRAD_AVAILABLE = False
    print("[WARNING] Tetrad core algorithms not available - tetrad_rfci/ tetrad_fges modules not found")

# FCI Variants (BOSS-FCI, GRaSP-FCI, SP-FCI) - removed per user request
FCI_VARIANTS_AVAILABLE = False

# Optional external algorithms (BOSS, TXGES) removed - not used in current pipeline

# Causal-learn algorithms (GES and FCI baselines for comparison)
CAUSALLEARN_AVAILABLE = False
try:
    from causallearn.search.ScoreBased.GES import ges as causallearn_ges
    from causallearn.search.ConstraintBased.FCI import fci as causallearn_fci
    from causallearn.graph.GraphClass import CausalGraph
    CAUSALLEARN_AVAILABLE = True
except ImportError as e:
    CAUSALLEARN_AVAILABLE = False
    print(f"[INFO] Causal-learn algorithms not available: {e}")


@dataclass
class AlgorithmResult:
    """Container for algorithm results"""
    adjacency_matrix: np.ndarray
    execution_time: float
    preprocessing_time: float
    postprocessing_time: float
    algorithm_name: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAlgorithm(ABC):
    """Base class for all causal discovery algorithms"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.use_prior_knowledge = False
        self.prior_knowledge = None
        
    @abstractmethod
    def _preprocess(self, data: np.ndarray, columns: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for the algorithm"""
        pass
        
    @abstractmethod
    def _run_algorithm(self, preprocessed_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Run the actual algorithm"""
        pass
        
    @abstractmethod
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess algorithm output"""
        pass
        
    def run(self, data: np.ndarray, columns: list) -> AlgorithmResult:
        """Main execution method with timing"""
        start_time = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        preprocessed_data, metadata = self._preprocess(data, columns)
        preprocessing_time = time.time() - preprocess_start
        
        # Algorithm execution
        algo_start = time.time()
        raw_output = self._run_algorithm(preprocessed_data, metadata)
        execution_time = time.time() - algo_start
        
        # Postprocessing
        postprocess_start = time.time()
        final_output = self._postprocess(raw_output, data.shape, metadata)
        postprocessing_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        return AlgorithmResult(
            adjacency_matrix=final_output,
            execution_time=execution_time,
            preprocessing_time=preprocessing_time,
            postprocessing_time=postprocessing_time,
            algorithm_name=self.name,
            metadata=metadata
        )

# Removed Castle (PC, GES), CausalLearn (FCI), and TXGES implementations - not used

class TetradRFCIAlgorithm(BaseAlgorithm):
    """Tetrad RFCI Algorithm implementation using our modular module"""
    
    def __init__(self, **kwargs):
        super().__init__("TetradRFCI", **kwargs)
        # Consistent parameters across all algorithms
        self.alpha = kwargs.get('alpha', 0.05)
        self.depth = kwargs.get('depth', -1)
        
    def _preprocess(self, data: np.ndarray, columns: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for Tetrad RFCI algorithm"""
        # Convert numpy array to pandas DataFrame for our module
        df = pd.DataFrame(data, columns=columns)
        
        metadata = {
            'original_shape': data.shape,
            'columns': columns,
            'alpha': self.alpha,
            'depth': self.depth
        }
        
        return df, metadata
        
    def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
        """Run Tetrad RFCI algorithm"""
        if not TETRAD_AVAILABLE:
            print("[WARNING] Tetrad RFCI not available")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
            
        try:
            print(f"[INFO] Starting Tetrad RFCI with alpha={self.alpha}, depth={self.depth}...")
            print(f"[INFO] Data shape: {preprocessed_data.shape}")
            
            # Use our modular RFCI implementation with optimized parameters
            rfci = TetradRFCI(alpha=self.alpha, depth=self.depth)
            
            # Apply prior knowledge if available
            prior = None
            if self.use_prior_knowledge and self.prior_knowledge:
                prior = self.prior_knowledge
            
            adj_matrix = rfci.run(preprocessed_data, prior=prior)
            
            print(f"[INFO] Tetrad RFCI completed successfully. Matrix shape: {adj_matrix.shape}")
            return adj_matrix
                
        except Exception as e:
            print(f"[ERROR] Tetrad RFCI algorithm failed: {e}")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
        
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess Tetrad RFCI output - preserves PAG format {-1, 0, 1, 2}"""
        if raw_output is None or np.any(np.isnan(raw_output)):
            return np.zeros((original_shape[1], original_shape[1]))
            
        # Ensure correct shape (preserve PAG values, don't convert to binary)
        if raw_output.shape != (original_shape[1], original_shape[1]):
            print(f"[WARNING] Tetrad RFCI output shape mismatch: {raw_output.shape} vs expected {(original_shape[1], original_shape[1])}")
            return np.zeros((original_shape[1], original_shape[1]))
            
        return raw_output.astype(int)


class TetradFGESAlgorithm(BaseAlgorithm):
    """Tetrad FGES Algorithm implementation using our modular module"""
    
    def __init__(self, **kwargs):
        super().__init__("TetradFGES", **kwargs)
        # Consistent parameters across all algorithms (penalty_discount=2.0 matches GFCI, BOSS-FCI, etc.)
        self.penalty_discount = kwargs.get('penalty_discount', 2.0)
        self.max_degree = kwargs.get('max_degree', -1)
        
    def _preprocess(self, data: np.ndarray, columns: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for Tetrad FGES algorithm"""
        # Convert numpy array to pandas DataFrame for our module
        df = pd.DataFrame(data, columns=columns)
        
        metadata = {
            'original_shape': data.shape,
            'columns': columns,
            'penalty_discount': self.penalty_discount,
            'max_degree': self.max_degree
        }
        
        return df, metadata
        
    def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
        """Run Tetrad FGES algorithm"""
        if not TETRAD_AVAILABLE:
            print("[WARNING] Tetrad FGES not available")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
            
        try:
            print(f"[INFO] Starting Tetrad FGES with penalty_discount={self.penalty_discount}, max_degree={self.max_degree}...")
            print(f"[INFO] Data shape: {preprocessed_data.shape}")
            
            # Use our modular FGES implementation with optimized parameters
            fges = TetradFGES(penalty_discount=self.penalty_discount, max_degree=self.max_degree)
            
            # Apply prior knowledge if available
            prior = None
            if self.use_prior_knowledge and self.prior_knowledge:
                prior = self.prior_knowledge
                
            adj_matrix = fges.run(preprocessed_data, prior=prior)
            
            print(f"[INFO] Tetrad FGES completed successfully. Matrix shape: {adj_matrix.shape}")
            return adj_matrix
                
        except Exception as e:
            print(f"[ERROR] Tetrad FGES algorithm failed: {e}")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
        
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess Tetrad FGES output - preserves PAG/CPDAG format {-1, 0, 1, 2}"""
        if raw_output is None or np.any(np.isnan(raw_output)):
            return np.zeros((original_shape[1], original_shape[1]))
            
        # Ensure correct shape (preserve PAG values, don't convert to binary)
        if raw_output.shape != (original_shape[1], original_shape[1]):
            print(f"[WARNING] Tetrad FGES output shape mismatch: {raw_output.shape} vs expected {(original_shape[1], original_shape[1])}")
            return np.zeros((original_shape[1], original_shape[1]))
            
        return raw_output.astype(int)


class AlgorithmRegistry:
    """Registry for managing causal discovery algorithms"""
    
    def __init__(self, enable_fallbacks: bool = False):
        self._algorithms = {}
        self.enable_fallbacks = enable_fallbacks
        self._register_default_algorithms()
        
    def _register_default_algorithms(self):
        """Register only PyTetrad algorithms (no fallbacks)."""
        if not TETRAD_AVAILABLE:
            print("[WARNING] PyTetrad algorithms are not available. Skipping Tetrad registrations.")
            # Intentionally return without raising to allow the pipeline to continue
            # (e.g., for environments without PyTetrad). The caller may see an
            # empty algorithm list and still complete the run without results.
            return
        self.register_algorithm(TetradRFCIAlgorithm())
        self.register_algorithm(TetradFGESAlgorithm())
        # Add GFCI
        try:
            # Simple adapter using convenience function
            class TetradGFCIAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradGFCI", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                    self.penalty_discount = kwargs.get('penalty_discount', 2.0)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_gfci import run_gfci
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    return run_gfci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, penalty_discount=self.penalty_discount, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradGFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradGFCI: {e}")
        # Add CPC
        try:
            class TetradCPCAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradCPC", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_cpc import run_cpc
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    return run_cpc(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradCPCAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradCPC: {e}")
        # Add CFCI
        try:
            class TetradCFCIAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradCFCI", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_cfci import run_cfci
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    return run_cfci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradCFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradCFCI: {e}")
        # Add FCI
        try:
            class TetradFCIAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradFCI", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_fci import run_fci
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    return run_fci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradFCI: {e}")
        # BOSS/SAM/DAGMA related adapters removed per request

        # DAGMA removed per request
        # Add FCI-Max adapter
        try:
            class TetradFCIMaxAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradFCIMax", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_fci_max import run_fci_max
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    return run_fci_max(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradFCIMaxAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradFCIMax: {e}")

        # Add PC algorithm
        try:
            class TetradPCAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradPC", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, {'original_shape': data.shape, 'columns': columns}
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_pc import TetradPC
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    alg = TetradPC(
                        alpha=self.alpha,
                        depth=self.depth
                    )
                    return alg.run(preprocessed_data, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradPCAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradPC: {e}")

        # Add BOSS-FCI
        try:
            class TetradBossFCIAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradBossFCI", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                    self.penalty_discount = kwargs.get('penalty_discount', 2.0)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_boss_fci import TetradBossFCI
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    alg = TetradBossFCI(
                        alpha=self.alpha,
                        depth=self.depth,
                        penalty_discount=self.penalty_discount
                    )
                    return alg.run(preprocessed_data, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradBossFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradBossFCI: {e}")
        
        # Add GRaSP-FCI
        try:
            class TetradGraspFCIAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradGraspFCI", **kwargs)
                    self.alpha = kwargs.get('alpha', 0.05)
                    self.depth = kwargs.get('depth', -1)
                    self.penalty_discount = kwargs.get('penalty_discount', 2.0)
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from acd_sea.inference_pipeline.tetrad_grasp_fci import TetradGraspFCI
                    prior = None
                    if self.use_prior_knowledge and self.prior_knowledge:
                        prior = self.prior_knowledge
                    alg = TetradGraspFCI(
                        alpha=self.alpha,
                        depth=self.depth,
                        penalty_discount=self.penalty_discount
                    )
                    return alg.run(preprocessed_data, prior=prior)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradGraspFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradGraspFCI: {e}")
        
        # =====================================================================
        # Causal-Learn Baseline Algorithms (GES and FCI)
        # These are included for comparison against PyTetrad variants
        # =====================================================================
        if CAUSALLEARN_AVAILABLE:
            # Causal-Learn GES (baseline for comparison with TetradFGES)
            try:
                class CausalLearnGESAlgorithm(BaseAlgorithm):
                    """Causal-learn GES implementation for baseline comparison."""
                    def __init__(self, **kwargs):
                        super().__init__("CausalLearnGES", **kwargs)
                        self.score_func = kwargs.get('score_func', 'local_score_BIC')
                        self.maxP = kwargs.get('maxP', None)
                    
                    def _preprocess(self, data: np.ndarray, columns: list):
                        return data, {'original_shape': data.shape, 'columns': columns}
                    
                    def _run_algorithm(self, preprocessed_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
                        try:
                            print(f"[INFO] Starting CausalLearn GES with score_func={self.score_func}...")
                            result = causallearn_ges(
                                preprocessed_data,
                                score_func=self.score_func,
                                maxP=self.maxP,
                                node_names=metadata.get('columns')
                            )
                            # Extract adjacency matrix from result
                            # GES returns a dict with 'G' key containing the graph
                            G = result.get('G', None)
                            if G is None:
                                print("[WARNING] CausalLearn GES returned no graph")
                                return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                            
                            # Convert GeneralGraph to adjacency matrix
                            adj_matrix = np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                            nodes = G.get_nodes()
                            node_map = {node.get_name(): i for i, node in enumerate(nodes)}
                            
                            for edge in G.get_graph_edges():
                                node1 = edge.get_node1().get_name()
                                node2 = edge.get_node2().get_name()
                                i, j = node_map.get(node1), node_map.get(node2)
                                if i is not None and j is not None:
                                    # Check edge type for direction
                                    endpoint1 = str(edge.get_endpoint1())
                                    endpoint2 = str(edge.get_endpoint2())
                                    # Arrow means directed: node1 -> node2 if endpoint2 is ARROW
                                    if endpoint2 == 'ARROW':
                                        adj_matrix[i, j] = 1
                                    if endpoint1 == 'ARROW':
                                        adj_matrix[j, i] = 1
                                    # If both TAIL, treat as undirected (add both directions)
                                    if endpoint1 == 'TAIL' and endpoint2 == 'TAIL':
                                        adj_matrix[i, j] = 1
                                        adj_matrix[j, i] = 1
                            
                            print(f"[INFO] CausalLearn GES completed. Edges found: {int(adj_matrix.sum())}")
                            return adj_matrix
                        except Exception as e:
                            print(f"[ERROR] CausalLearn GES failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                    
                    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                        if raw_output is None or np.any(np.isnan(raw_output)):
                            return np.zeros((original_shape[1], original_shape[1]))
                        out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                        return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
                
                self.register_algorithm(CausalLearnGESAlgorithm())
            except Exception as e:
                print(f"[WARNING] Could not register CausalLearnGES: {e}")

            # Causal-Learn FCI (baseline for comparison with TetradFCI variants)
            try:
                class CausalLearnFCIAlgorithm(BaseAlgorithm):
                    """Causal-learn FCI implementation for baseline comparison."""
                    def __init__(self, **kwargs):
                        super().__init__("CausalLearnFCI", **kwargs)
                        self.alpha = kwargs.get('alpha', 0.05)
                        self.depth = kwargs.get('depth', -1)
                        self.independence_test = kwargs.get('independence_test', 'fisherz')
                    
                    def _preprocess(self, data: np.ndarray, columns: list):
                        return data, {'original_shape': data.shape, 'columns': columns}
                    
                    def _run_algorithm(self, preprocessed_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
                        try:
                            print(f"[INFO] Starting CausalLearn FCI with alpha={self.alpha}, depth={self.depth}...")
                            G, edges = causallearn_fci(
                                preprocessed_data,
                                independence_test_method=self.independence_test,
                                alpha=self.alpha,
                                depth=self.depth,
                                verbose=False,
                                show_progress=False,
                                node_names=metadata.get('columns')
                            )
                            
                            # Convert Graph to adjacency matrix
                            adj_matrix = np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                            nodes = G.get_nodes()
                            node_map = {node.get_name(): i for i, node in enumerate(nodes)}
                            
                            for edge in G.get_graph_edges():
                                node1 = edge.get_node1().get_name()
                                node2 = edge.get_node2().get_name()
                                i, j = node_map.get(node1), node_map.get(node2)
                                if i is not None and j is not None:
                                    endpoint1 = str(edge.get_endpoint1())
                                    endpoint2 = str(edge.get_endpoint2())
                                    # For FCI: ARROW means directed, CIRCLE means uncertain
                                    # We treat ARROW as directed edge
                                    if endpoint2 == 'ARROW':
                                        adj_matrix[i, j] = 1
                                    if endpoint1 == 'ARROW':
                                        adj_matrix[j, i] = 1
                                    # CIRCLE-CIRCLE or TAIL-TAIL: treat as undirected
                                    if (endpoint1 in ['TAIL', 'CIRCLE'] and 
                                        endpoint2 in ['TAIL', 'CIRCLE']):
                                        adj_matrix[i, j] = 1
                                        adj_matrix[j, i] = 1
                            
                            print(f"[INFO] CausalLearn FCI completed. Edges found: {int(adj_matrix.sum())}")
                            return adj_matrix
                        except Exception as e:
                            print(f"[ERROR] CausalLearn FCI failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                    
                    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                        if raw_output is None or np.any(np.isnan(raw_output)):
                            return np.zeros((original_shape[1], original_shape[1]))
                        out = raw_output.astype(int)  # Preserve PAG format {-1, 0, 1, 2}
                        return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
                
                self.register_algorithm(CausalLearnFCIAlgorithm())
            except Exception as e:
                print(f"[WARNING] Could not register CausalLearnFCI: {e}")
            
    def register_algorithm(self, algorithm: BaseAlgorithm):
        """Register a new algorithm"""
        self._algorithms[algorithm.name] = algorithm
        
    def get_algorithm(self, name: str) -> Optional[BaseAlgorithm]:
        """Get algorithm by name"""
        return self._algorithms.get(name)
        
    def list_algorithms(self) -> list:
        """List all available algorithms"""
        return list(self._algorithms.keys())
        
    def run_algorithm(self, name: str, data: np.ndarray, columns: list, 
                     use_prior_knowledge: bool = False, prior_knowledge: Optional[Dict] = None) -> Optional[AlgorithmResult]:
        """Run a specific algorithm with optional prior knowledge"""
        algorithm = self.get_algorithm(name)
        if algorithm is None:
            print(f"[ERROR] Algorithm '{name}' not found")
            return None
        
        # Pass prior knowledge to algorithm if supported
        if hasattr(algorithm, 'use_prior_knowledge'):
            algorithm.use_prior_knowledge = use_prior_knowledge
        if hasattr(algorithm, 'prior_knowledge'):
            algorithm.prior_knowledge = prior_knowledge
            
        return algorithm.run(data, columns)

def compute_metrics(pred_adj: np.ndarray, true_adj: np.ndarray, max_edges: int) -> Dict[str, Any]:
    """
    Compute evaluation metrics for causal discovery results.
    
    Handles PAG format {-1, 0, 1, 2} from algorithm output:
        -1: Backward edge (a ← b)
         0: No edge
         1: Undirected edge (a — b)
         2: Forward edge (a → b)
    
    Computes both skeleton metrics (any edge) and directed metrics (correct direction).
    """
    try:
        # Handle edge cases
        if pred_adj is None or np.any(np.isnan(pred_adj)) or np.any(np.isinf(pred_adj)):
            return default_metrics()
            
        # Ensure matrices are the same size
        if pred_adj.shape != true_adj.shape:
            print(f"[WARNING] Shape mismatch: pred={pred_adj.shape}, true={true_adj.shape}")
            return default_metrics()
        
        # Convert PAG to directed adjacency for metrics comparison
        # This extracts forward edges (value 2) and undirected (value 1)
        pred_directed = pag_to_directed_adjacency(pred_adj)
        
        # Ground truth is typically already in directed format {0, 1}
        # but we apply the same conversion for consistency
        true_directed = (true_adj != 0).astype(int)
        
        # Also compute skeleton metrics (ignore direction, just check if edge exists)
        pred_skeleton = pag_to_skeleton(pred_adj)
        true_skeleton = (true_adj != 0).astype(int)
        true_skeleton = np.maximum(true_skeleton, true_skeleton.T)  # Make symmetric
        
        # Primary metrics: directed comparison (what the algorithm predicted vs ground truth)
        metrics = MetricsDAG(pred_directed, true_directed).metrics
        
        # Extract basic metrics
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        shd = metrics.get('shd', None)
        
        # Extract SID and gscore (available in gCastle's MetricsDAG)
        sid = metrics.get('sid', None)
        gscore = metrics.get('gscore', None)
        
        # Compute skeleton metrics separately
        skeleton_metrics = MetricsDAG(pred_skeleton, true_skeleton).metrics
        skeleton_precision = skeleton_metrics.get('precision', 0.0)
        skeleton_recall = skeleton_metrics.get('recall', 0.0)
        skeleton_shd = skeleton_metrics.get('shd', None)
        
        # Convert numpy types to Python types to avoid serialization issues
        if isinstance(precision, np.floating):
            precision = float(precision)
        if isinstance(recall, np.floating):
            recall = float(recall)
        if isinstance(shd, (np.integer, np.floating)):
            shd = int(shd) if shd == int(shd) else float(shd)
        if isinstance(sid, (np.integer, np.floating)):
            sid = int(sid) if sid == int(sid) else float(sid)
        if isinstance(gscore, np.floating):
            gscore = float(gscore)
        if isinstance(skeleton_precision, np.floating):
            skeleton_precision = float(skeleton_precision)
        if isinstance(skeleton_recall, np.floating):
            skeleton_recall = float(skeleton_recall)
        if isinstance(skeleton_shd, (np.integer, np.floating)):
            skeleton_shd = int(skeleton_shd) if skeleton_shd == int(skeleton_shd) else float(skeleton_shd)
            
        # Handle NaN values
        if np.isnan(precision) or np.isnan(recall):
            precision = 0.0
            recall = 0.0
        if np.isnan(skeleton_precision) or np.isnan(skeleton_recall):
            skeleton_precision = 0.0
            skeleton_recall = 0.0
        if sid is not None and np.isnan(sid):
            sid = None
        if gscore is not None and np.isnan(gscore):
            gscore = None
            
        # Calculate F1 scores safely
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        if skeleton_precision + skeleton_recall > 0:
            skeleton_f1 = 2 * (skeleton_precision * skeleton_recall) / (skeleton_precision + skeleton_recall)
        else:
            skeleton_f1 = 0.0
            
        # Calculate normalized SHD safely
        if shd is not None and max_edges > 0:
            normalized_shd = 1 - (shd / max_edges)
        else:
            normalized_shd = 0.0
            
        # Count edges by type in predicted adjacency
        pred_forward_edges = int(np.sum(pred_adj == 2))
        pred_undirected_edges = int(np.sum(pred_adj == 1))
        pred_backward_edges = int(np.sum(pred_adj == -1))
        pred_total_edges = int(np.sum(pred_directed != 0))
            
        # Debug print for zero precision and recall
        if precision == 0.0 and recall == 0.0:
            print(f"[DEBUG] Both precision and recall are zero! pred_edge_count={pred_total_edges} true_edge_count={np.sum(true_directed != 0)}")

        return {
            'shd': shd,
            'normalized_shd': normalized_shd,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'pred_edge_count': pred_total_edges,
            'pred_forward_edges': pred_forward_edges,
            'pred_undirected_edges': pred_undirected_edges,
            'pred_backward_edges': pred_backward_edges,
            'skeleton_f1': skeleton_f1,
            'skeleton_precision': skeleton_precision,
            'skeleton_recall': skeleton_recall,
            'skeleton_shd': skeleton_shd,
            'sid': sid,
            'gscore': gscore
        }
    except Exception as e:
        print(f"[ERROR] Exception in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return default_metrics()

def default_metrics() -> Dict[str, Any]:
    """Return default metrics for failed algorithms"""
    return {
        'shd': None,
        'normalized_shd': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'pred_edge_count': 0,
        'pred_forward_edges': 0,
        'pred_undirected_edges': 0,
        'pred_backward_edges': 0,
        'skeleton_f1': 0.0,
        'skeleton_precision': 0.0,
        'skeleton_recall': 0.0,
        'skeleton_shd': None,
        'sid': None,
        'gscore': None
    }

def apply_causal_discovery_algorithms(data: pd.DataFrame, true_adj_matrix: np.ndarray, 
                                    algorithms: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple causal discovery algorithms to the data
    
    Args:
        data: Input data as pandas DataFrame
        true_adj_matrix: True adjacency matrix for evaluation
        algorithms: List of algorithm names to run (None for all)
    
    Returns:
        Dictionary of results for each algorithm
    """
    registry = AlgorithmRegistry()
    
    if algorithms is None:
        algorithms = registry.list_algorithms()
    
    results = {}
    max_possible_edges = true_adj_matrix.size
    
    for algo_name in algorithms:
        print(f"Running {algo_name}...")
        
        # Run algorithm
        result = registry.run_algorithm(algo_name, data.values, list(data.columns))
        
        if result is None:
            results[algo_name] = {**default_metrics(), 'execution_time': 0.0}
            continue
            
        # Compute metrics for raw result
        raw_metrics = compute_metrics(result.adjacency_matrix, true_adj_matrix, max_possible_edges)
        results[algo_name] = {
            **raw_metrics,
            'execution_time': result.execution_time,
            'preprocessing_time': result.preprocessing_time,
            'postprocessing_time': result.postprocessing_time
        }
        
    
    return results

# Legacy function for backward compatibility
def get_algorithm_registry():
    """Legacy function - returns a simple function registry for backward compatibility"""
    registry = AlgorithmRegistry()
    
    def create_legacy_function(algo_name):
        def legacy_func(scaled_data, data_columns):
            result = registry.run_algorithm(algo_name, scaled_data, data_columns)
            return result.adjacency_matrix if result else None
        return legacy_func
    
    return {name: create_legacy_function(name) for name in registry.list_algorithms()}
