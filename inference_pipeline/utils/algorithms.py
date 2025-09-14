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



# ReX algorithm (currently disabled)
REX_AVAILABLE = False

# Python-only alternatives to FGES (currently disabled)

# Tetrad algorithms
try:
    from tetrad_rfci import TetradRFCI
    from tetrad_fges import TetradFGES
    from tetrad_gfci import TetradGFCI
    from tetrad_fci_max import TetradFCIMax
    from tetrad_fci import TetradFCI
    from tetrad_cpc import TetradCPC
    from tetrad_cfci import TetradCFCI
    from tetrad_boss import TetradBOSS
    TETRAD_AVAILABLE = True
    print("[INFO] Tetrad algorithms (RFCI, FGES, GFCI, FCI, FCI-Max, CPC, CFCI, BOSS) available")
except ImportError:
    TETRAD_AVAILABLE = False
    print("[WARNING] Tetrad algorithms not available - tetrad_* modules not found")

# Try to import TXGES (new version)
try:
    import txges
    TXGES_AVAILABLE = True
    print("[INFO] TXGES algorithm available from txges library")
except ImportError:
    TXGES_AVAILABLE = False
    print("[INFO] TXGES algorithm not available - txges not installed")

# Optional external algorithms (BOSS)
try:
    from .boss_adapter import run_boss
    BOSS_AVAILABLE = True
    print("[INFO] BOSS adapter available")
except Exception:
    BOSS_AVAILABLE = False
    print("[INFO] BOSS not available")



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

# Removed Castle (PC, GES) and CausalLearn (FCI) fallback implementations


class TXGESAlgorithm(BaseAlgorithm):
    """TXGES Algorithm implementation using txges library"""
    
    def __init__(self, **kwargs):
        super().__init__("TXGES", **kwargs)
        
    def _preprocess(self, data: np.ndarray, columns: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for TXGES algorithm"""
        # TXGES works with continuous data, no scaling needed
        metadata = {
            'original_shape': data.shape,
            'columns': columns
        }
        return data, metadata
        
    def _run_algorithm(self, preprocessed_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Run TXGES algorithm"""
        if not TXGES_AVAILABLE:
            print("[WARNING] TXGES not available - txges not installed")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
            
        try:
            print(f"[INFO] Starting TXGES with {preprocessed_data.shape[1]} variables...")
            print(f"[INFO] Data shape: {preprocessed_data.shape}")
            
            # Initialize and run TXGES using the new usage pattern
            model = txges.XGES()
            model.fit(preprocessed_data)
            
            print("[INFO] Extracting adjacency matrix...")
            # Get the adjacency matrix from the PDAG
            pdag = model.get_pdag()
            adj_matrix = pdag.to_adjacency_matrix()
            
            # Ensure it's the right shape
            if adj_matrix.shape == (preprocessed_data.shape[1], preprocessed_data.shape[1]):
                print(f"[INFO] TXGES completed successfully. Matrix shape: {adj_matrix.shape}")
                return adj_matrix
            else:
                print(f"[WARNING] TXGES adjacency matrix shape mismatch: {adj_matrix.shape}")
                return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                
        except Exception as e:
            print(f"[ERROR] TXGES algorithm failed: {e}")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
        
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess TXGES output"""
        # Ensure output is binary and correct shape
        if raw_output is None or np.any(np.isnan(raw_output)):
            return np.zeros((original_shape[1], original_shape[1]))
            
        # Convert to binary
        binary_output = (raw_output != 0).astype(int)
        
        # Ensure correct shape
        if binary_output.shape != (original_shape[1], original_shape[1]):
            print(f"[WARNING] TXGES output shape mismatch: {binary_output.shape} vs expected {(original_shape[1], original_shape[1])}")
            return np.zeros((original_shape[1], original_shape[1]))
            
        return binary_output


class TetradRFCIAlgorithm(BaseAlgorithm):
    """Tetrad RFCI Algorithm implementation using our modular module"""
    
    def __init__(self, **kwargs):
        super().__init__("TetradRFCI", **kwargs)
        # Use optimized parameters from our analysis
        self.alpha = kwargs.get('alpha', 0.1)
        self.depth = kwargs.get('depth', 2)
        
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
            adj_matrix = rfci.run(preprocessed_data)
            
            print(f"[INFO] Tetrad RFCI completed successfully. Matrix shape: {adj_matrix.shape}")
            return adj_matrix
                
        except Exception as e:
            print(f"[ERROR] Tetrad RFCI algorithm failed: {e}")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
        
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess Tetrad RFCI output"""
        # Ensure output is binary and correct shape
        if raw_output is None or np.any(np.isnan(raw_output)):
            return np.zeros((original_shape[1], original_shape[1]))
            
        # Convert to binary
        binary_output = (raw_output != 0).astype(int)
        
        # Ensure correct shape
        if binary_output.shape != (original_shape[1], original_shape[1]):
            print(f"[WARNING] Tetrad RFCI output shape mismatch: {binary_output.shape} vs expected {(original_shape[1], original_shape[1])}")
            return np.zeros((original_shape[1], original_shape[1]))
            
        return binary_output


class TetradFGESAlgorithm(BaseAlgorithm):
    """Tetrad FGES Algorithm implementation using our modular module"""
    
    def __init__(self, **kwargs):
        super().__init__("TetradFGES", **kwargs)
        # Use optimized parameters from our analysis
        self.penalty_discount = kwargs.get('penalty_discount', 0.5)
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
            adj_matrix = fges.run(preprocessed_data)
            
            print(f"[INFO] Tetrad FGES completed successfully. Matrix shape: {adj_matrix.shape}")
            return adj_matrix
                
        except Exception as e:
            print(f"[ERROR] Tetrad FGES algorithm failed: {e}")
            return np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
        
    def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
        """Postprocess Tetrad FGES output"""
        # Ensure output is binary and correct shape
        if raw_output is None or np.any(np.isnan(raw_output)):
            return np.zeros((original_shape[1], original_shape[1]))
            
        # Convert to binary
        binary_output = (raw_output != 0).astype(int)
        
        # Ensure correct shape
        if binary_output.shape != (original_shape[1], original_shape[1]):
            print(f"[WARNING] Tetrad FGES output shape mismatch: {binary_output.shape} vs expected {(original_shape[1], original_shape[1])}")
            return np.zeros((original_shape[1], original_shape[1]))
            
        return binary_output


class AlgorithmRegistry:
    """Registry for managing causal discovery algorithms"""
    
    def __init__(self, enable_fallbacks: bool = False):
        self._algorithms = {}
        self.enable_fallbacks = enable_fallbacks
        self._register_default_algorithms()
        
    def _register_default_algorithms(self):
        """Register only PyTetrad algorithms (no fallbacks)."""
        if not TETRAD_AVAILABLE:
            raise RuntimeError("PyTetrad algorithms are not available. Ensure tetrad_rfci.py and tetrad_fges.py are present.")
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
                    from tetrad_gfci import run_gfci
                    return run_gfci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth, penalty_discount=self.penalty_discount)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
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
                    from tetrad_cpc import run_cpc
                    return run_cpc(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
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
                    from tetrad_cfci import run_cfci
                    return run_cfci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradCFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradCFCI: {e}")
        # SAM not supported in current Tetrad build; consider external methods like BOSS/DAGMA separately
        # Add Tetrad BOSS (native via pytetrad.tools.search)
        try:
            class TetradBOSSAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("TetradBOSS", **kwargs)
                    self.penalty_discount = kwargs.get('penalty_discount', 2.0)
                    self.use_bes = kwargs.get('use_bes', True)
                    self.num_starts = kwargs.get('num_starts', 10)
                    self.threads = kwargs.get('threads', max(1, (os.cpu_count() or 1)))
                def _preprocess(self, data: np.ndarray, columns: list):
                    df = pd.DataFrame(data, columns=columns)
                    return df, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
                    from tetrad_boss import run_boss_tetrad
                    return run_boss_tetrad(
                        preprocessed_data,
                        list(preprocessed_data.columns),
                        penalty_discount=self.penalty_discount,
                        use_bes=self.use_bes,
                        num_starts=self.num_starts,
                        threads=self.threads,
                    )
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradBOSSAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradBOSS: {e}")
        # Add BOSS if available
        if 'BOSS_AVAILABLE' in globals() and BOSS_AVAILABLE:
            class BOSSAlgorithm(BaseAlgorithm):
                def __init__(self, **kwargs):
                    super().__init__("BOSS", **kwargs)
                def _preprocess(self, data: np.ndarray, columns: list):
                    return data, { 'original_shape': data.shape, 'columns': columns }
                def _run_algorithm(self, preprocessed_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
                    adj = run_boss(preprocessed_data)
                    return adj if adj is not None else np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[1]))
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(BOSSAlgorithm())

        # DAGMA removed per request
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
                    from tetrad_fci import run_fci
                    return run_fci(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradFCIAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradFCI: {e}")
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
                    from tetrad_fci_max import run_fci_max
                    return run_fci_max(preprocessed_data, list(preprocessed_data.columns), alpha=self.alpha, depth=self.depth)
                def _postprocess(self, raw_output: np.ndarray, original_shape: Tuple[int, int], metadata: Dict[str, Any]) -> np.ndarray:
                    if raw_output is None or np.any(np.isnan(raw_output)):
                        return np.zeros((original_shape[1], original_shape[1]))
                    out = (raw_output != 0).astype(int)
                    return out if out.shape == (original_shape[1], original_shape[1]) else np.zeros((original_shape[1], original_shape[1]))
            self.register_algorithm(TetradFCIMaxAlgorithm())
        except Exception as e:
            print(f"[WARNING] Could not register TetradFCIMax: {e}")

            
    def register_algorithm(self, algorithm: BaseAlgorithm):
        """Register a new algorithm"""
        self._algorithms[algorithm.name] = algorithm
        
    def get_algorithm(self, name: str) -> Optional[BaseAlgorithm]:
        """Get algorithm by name"""
        return self._algorithms.get(name)
        
    def list_algorithms(self) -> list:
        """List all available algorithms"""
        return list(self._algorithms.keys())
        
    def run_algorithm(self, name: str, data: np.ndarray, columns: list) -> Optional[AlgorithmResult]:
        """Run a specific algorithm"""
        algorithm = self.get_algorithm(name)
        if algorithm is None:
            print(f"[ERROR] Algorithm '{name}' not found")
            return None
            
        return algorithm.run(data, columns)

def compute_metrics(pred_adj: np.ndarray, true_adj: np.ndarray, max_edges: int) -> Dict[str, Any]:
    """Compute evaluation metrics for causal discovery results"""
    try:
        # Handle edge cases
        if pred_adj is None or np.any(np.isnan(pred_adj)) or np.any(np.isinf(pred_adj)):
            return default_metrics()
            
        # Ensure matrices are the same size
        if pred_adj.shape != true_adj.shape:
            print(f"[WARNING] Shape mismatch: pred={pred_adj.shape}, true={true_adj.shape}")
            return default_metrics()
            
        # Ensure binary matrices
        pred_binary = (pred_adj != 0).astype(int)
        true_binary = (true_adj != 0).astype(int)
        
        metrics = MetricsDAG(pred_binary, true_binary).metrics
        
        # Handle NaN values in metrics
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        shd = metrics.get('shd', None)
        
        # Convert numpy types to Python types to avoid serialization issues
        if isinstance(precision, np.floating):
            precision = float(precision)
        if isinstance(recall, np.floating):
            recall = float(recall)
        if isinstance(shd, (np.integer, np.floating)):
            shd = int(shd) if shd == int(shd) else float(shd)
            
        # Handle NaN values
        if np.isnan(precision) or np.isnan(recall):
            precision = 0.0
            recall = 0.0
            
        # Calculate F1 score safely
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        # Calculate normalized SHD safely
        if shd is not None and max_edges > 0:
            normalized_shd = 1 - (shd / max_edges)
        else:
            normalized_shd = 0.0
            
        # Debug print for zero precision and recall
        if precision == 0.0 and recall == 0.0:
            print(f"[DEBUG] Both precision and recall are zero! pred_edge_count={np.sum(pred_binary != 0)} true_edge_count={np.sum(true_binary != 0)}")

        return {
            'shd': shd,
            'normalized_shd': normalized_shd,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'pred_edge_count': int(np.sum(pred_binary != 0))
        }
    except Exception as e:
        print(f"[ERROR] Exception in compute_metrics: {e}")
        return default_metrics()

def default_metrics() -> Dict[str, Any]:
    """Return default metrics for failed algorithms"""
    return {
        'shd': None,
        'normalized_shd': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'pred_edge_count': 0
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
