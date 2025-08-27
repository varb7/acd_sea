from castle.algorithms import PC, GES
import numpy as np

# Try to import TXGES (new version)
try:
    import txges
    TXGES_AVAILABLE = True
    print("[INFO] TXGES algorithm available from txges library")
except ImportError:
    TXGES_AVAILABLE = False
    print("[INFO] TXGES algorithm not available - txges not installed")



# Try to import ReX from causalexplain
try:
    from causalexplain import GraphDiscovery
    REX_AVAILABLE = True
except ImportError:
    REX_AVAILABLE = False
    print("[INFO] ReX algorithm not available in algorithm registry - causalexplain not installed (optional)")

# Try to import ICALiNGAM and DirectLiNGAM from gcastle
try:
    from castle.algorithms import ICALiNGAM, DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False
    print("[INFO] LiNGAM algorithms not available in algorithm registry - gcastle version may not support them")

# Each function takes (scaled_data, data_columns) and returns an adjacency matrix

def pc_algorithm(scaled_data, data_columns):
    try:
        algo = PC()
        algo.learn(scaled_data)
        
        # Get the causal adjacency matrix
        causal_matrix = algo.causal_matrix
        
        # Ensure it's the correct shape (n_vars x n_vars)
        if causal_matrix.shape != (scaled_data.shape[1], scaled_data.shape[1]):
            print(f"[WARNING] PC returned wrong shape: {causal_matrix.shape}, expected {(scaled_data.shape[1], scaled_data.shape[1])}")
            # If it's the wrong shape, try to get the adjacency matrix differently
            try:
                # Try to access the adjacency matrix directly
                if hasattr(algo, 'adjacency_matrix'):
                    causal_matrix = algo.adjacency_matrix
                elif hasattr(algo, 'get_adjacency_matrix'):
                    causal_matrix = algo.get_adjacency_matrix()
                else:
                    # Create a zero matrix of the correct shape
                    causal_matrix = np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
            except:
                causal_matrix = np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
        
        return causal_matrix
    except Exception as e:
        print(f"[WARNING] PC algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))

def ges_algorithm(scaled_data, data_columns):
    try:
        algo = GES()
        algo.learn(scaled_data)
        
        # Get the causal adjacency matrix
        causal_matrix = algo.causal_matrix
        
        # Ensure it's the correct shape (n_vars x n_vars)
        if causal_matrix.shape != (scaled_data.shape[1], scaled_data.shape[1]):
            print(f"[WARNING] GES returned wrong shape: {causal_matrix.shape}, expected {(scaled_data.shape[1], scaled_data.shape[1])}")
            # If it's the wrong shape, try to get the adjacency matrix differently
            try:
                # Try to access the adjacency matrix directly
                if hasattr(algo, 'adjacency_matrix'):
                    causal_matrix = algo.adjacency_matrix
                elif hasattr(algo, 'get_adjacency_matrix'):
                    causal_matrix = algo.get_adjacency_matrix()
                else:
                    # Create a zero matrix of the correct shape
                    causal_matrix = np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
            except:
                causal_matrix = np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
        
        return causal_matrix
    except Exception as e:
        print(f"[WARNING] GES algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))

def fci_algorithm(scaled_data, data_columns):
    try:
        from causallearn.search.ConstraintBased.FCI import fci as fci_algorithm
        result = fci_algorithm(scaled_data, alpha=0.05)
        
        # FCI returns a tuple: (GeneralGraph, [Edge objects])
        if isinstance(result, tuple) and len(result) > 0:
            general_graph = result[0]  # First element is the GeneralGraph
            
            # The GeneralGraph has a .graph attribute that is a numpy adjacency matrix
            if hasattr(general_graph, 'graph'):
                adj_matrix = general_graph.graph
                
                # Ensure it's the right shape
                if adj_matrix.shape == (scaled_data.shape[1], scaled_data.shape[1]):
                    # Convert to binary (0 or 1) - FCI can return fractional values
                    binary_adj = (adj_matrix != 0).astype(int)
                    return binary_adj
                else:
                    print(f"[WARNING] FCI adjacency matrix shape mismatch: {adj_matrix.shape} vs expected {(scaled_data.shape[1], scaled_data.shape[1])}")
                    return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
            else:
                print(f"[WARNING] FCI GeneralGraph has no .graph attribute")
                return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
                
        else:
            print(f"[WARNING] FCI returned unexpected result type: {type(result)}")
            return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
            
    except Exception as e:
        print(f"[WARNING] FCI algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))

def icalingam_algorithm(scaled_data, data_columns):
    if not LINGAM_AVAILABLE:
        print("[WARNING] ICALiNGAM algorithm not available - gcastle version may not support it")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
    
    try:
        algo = ICALiNGAM()
        algo.learn(scaled_data)
        
        # Check what we got back
        result = algo.causal_matrix
        
        # If it's the wrong shape (data matrix instead of adjacency matrix)
        if result.shape != (scaled_data.shape[1], scaled_data.shape[1]):
            print(f"[WARNING] ICALiNGAM returned data matrix shape: {result.shape}, expected adjacency matrix shape: {(scaled_data.shape[1], scaled_data.shape[1])}")
            
            # Try to get the adjacency matrix using different methods
            if hasattr(algo, 'adjacency_matrix'):
                result = algo.adjacency_matrix
            elif hasattr(algo, 'get_adjacency_matrix'):
                result = algo.get_adjacency_matrix()
            elif hasattr(algo, 'adj_matrix'):
                result = algo.adj_matrix
            elif hasattr(algo, 'get_adj_matrix'):
                result = algo.get_adj_matrix()
            else:
                # If we can't get the adjacency matrix, return zeros
                print("[WARNING] Could not find adjacency matrix attribute for ICALiNGAM")
                return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
        
        return result
    except Exception as e:
        print(f"[WARNING] ICALiNGAM algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))

def directlingam_algorithm(scaled_data, data_columns):
    if not LINGAM_AVAILABLE:
        print("[WARNING] DirectLiNGAM algorithm not available - gcastle version may not support it")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
    
    try:
        algo = DirectLiNGAM()
        algo.learn(scaled_data)
        
        # Check what we got back
        result = algo.causal_matrix
        
        # If it's the wrong shape (data matrix instead of adjacency matrix)
        if result.shape != (scaled_data.shape[1], scaled_data.shape[1]):
            print(f"[WARNING] DirectLiNGAM returned data matrix shape: {result.shape}, expected adjacency matrix shape: {(scaled_data.shape[1], scaled_data.shape[1])}")
            
            # Try to get the adjacency matrix using different methods
            if hasattr(algo, 'adjacency_matrix'):
                result = algo.adjacency_matrix
            elif hasattr(algo, 'get_adjacency_matrix'):
                result = algo.get_adjacency_matrix()
            elif hasattr(algo, 'adj_matrix'):
                result = algo.adj_matrix
            elif hasattr(algo, 'get_adj_matrix'):
                result = algo.get_adj_matrix()
            else:
                # If we can't get the adjacency matrix, return zeros
                print("[WARNING] Could not find adjacency matrix attribute for DirectLiNGAM")
                return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
        
        return result
    except Exception as e:
        print(f"[WARNING] DirectLiNGAM algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))


def txges_algorithm(scaled_data, data_columns):
    if not TXGES_AVAILABLE:
        print("[WARNING] TXGES algorithm not available - txges not installed")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
    
    try:
        print(f"[INFO] Starting TXGES with {scaled_data.shape[1]} variables...")
        print(f"[INFO] Data shape: {scaled_data.shape}")
        
        # Initialize and run TXGES using the new usage pattern
        model = txges.XGES()
        model.fit(scaled_data)
        
        print("[INFO] Extracting adjacency matrix...")
        # Get the adjacency matrix from the PDAG
        pdag = model.get_pdag()
        adj_matrix = pdag.to_adjacency_matrix()
        
        # Ensure it's the right shape
        if adj_matrix.shape == (scaled_data.shape[1], scaled_data.shape[1]):
            print(f"[INFO] TXGES completed successfully. Matrix shape: {adj_matrix.shape}")
            return adj_matrix
        else:
            print(f"[WARNING] TXGES adjacency matrix shape mismatch: {adj_matrix.shape}")
            return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
            
    except Exception as e:
        print(f"[WARNING] TXGES algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))


def rex_algorithm(scaled_data, data_columns):
    if not REX_AVAILABLE:
        print("[WARNING] ReX algorithm not available - causalexplain not installed")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
    
    try:
        # Create a temporary CSV file for ReX
        import pandas as pd
        import tempfile
        import os
        
        # Convert numpy array to DataFrame with column names
        df = pd.DataFrame(scaled_data, columns=data_columns)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_csv:
            df.to_csv(temp_csv.name, index=False)
            csv_path = temp_csv.name
        
        # Initialize ReX experiment
        experiment = GraphDiscovery(
            experiment_name='rex_inference',
            model_type='rex',
            csv_filename=csv_path,
            threshold=0.1,  # Default threshold for edge detection
            iterations=100,  # Number of iterations
            bootstrap=False,  # Disable bootstrap for speed
            regressor='rf'  # Random forest regressor
        )
        
        # Run the experiment
        experiment.run()
        
        # Get the adjacency matrix from the experiment
        # ReX returns a graph object, we need to convert it to adjacency matrix
        try:
            # Try to get the adjacency matrix directly
            adj_matrix = experiment.get_adjacency_matrix()
        except:
            # Fallback: create adjacency matrix from graph edges
            graph = experiment.get_graph()
            n_vars = len(data_columns)
            adj_matrix = np.zeros((n_vars, n_vars))
            
            # Convert graph edges to adjacency matrix
            for edge in graph.edges():
                source_idx = data_columns.index(edge[0])
                target_idx = data_columns.index(edge[1])
                adj_matrix[source_idx, target_idx] = 1
        
        # Clean up temporary file
        os.unlink(csv_path)
        
        return adj_matrix
        
    except Exception as e:
        print(f"[WARNING] ReX algorithm failed: {e}")
        return np.zeros((scaled_data.shape[1], scaled_data.shape[1]))

def get_algorithm_registry():
    registry = {
        'PC': pc_algorithm,
        'GES': ges_algorithm,
        'FCI': fci_algorithm,
    }
    
    # Add LiNGAM algorithms if available
    if LINGAM_AVAILABLE:
        registry['ICALiNGAM'] = icalingam_algorithm
        registry['DirectLiNGAM'] = directlingam_algorithm
    
    # Add TXGES if available
    if TXGES_AVAILABLE:
        registry['TXGES'] = txges_algorithm
    

    
    # Add ReX if available
    if REX_AVAILABLE:
        registry['ReX'] = rex_algorithm
    
    return registry 