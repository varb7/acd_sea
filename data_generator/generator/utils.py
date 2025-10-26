import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np

def append_to_csv(index_fp, row):
    """Append one row to a CSV index file, creating it if necessary."""
    if os.path.exists(index_fp):
        try:
            df = pd.read_csv(index_fp)
        except Exception:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    # Keep a stable column order
    cols = ["fp_data", "fp_graph", "fp_regime", "fp_intervention", "split", "n_samples", "n_variables"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(index_fp), exist_ok=True)

    df.to_csv(index_fp, index=False)

def save_dcdi_bundle(folder, data_arr, dag_arr):
    """
    Write DCDI format files: data.npy, regimes.csv, interventions.csv, graph.npy.
    
    For observational data:
    - regimes.csv: all zeros (observational regime)
    - interventions.csv: all -1 (no interventions)
    """
    # Save data and graph in DCDI format
    np.save(os.path.join(folder, "data.npy"), data_arr.astype(np.float32))
    np.save(os.path.join(folder, "graph.npy"), dag_arr.astype(np.int8))

    N = data_arr.shape[0]
    
    # For observational data: all samples are in regime 0 (observational)
    pd.Series(np.zeros(N, dtype=int)).to_csv(
        os.path.join(folder, "regimes.csv"), index=False, header=False
    )
    
    # For observational data: no interventions (all -1)
    pd.Series(-np.ones(N, dtype=int)).to_csv(
        os.path.join(folder, "interventions.csv"), index=False, header=False
    )

def save_dataset_with_splits(dataframe, adjacency_matrix, metadata, dataset_dir, base_name, index_file=None, train_ratio=0.8):
    """
    Save dataset with automatic train/test split generation.
    
    Args:
        dataframe: The dataset dataframe
        adjacency_matrix: The adjacency matrix
        metadata: Dataset metadata
        dataset_dir: Directory to save dataset
        base_name: Base name for files
        index_file: Index file path
        train_ratio: Ratio of data to use for training (default 0.8)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    # Align data columns to temporal order if provided
    if isinstance(metadata, dict) and "temporal_order" in metadata:
        try:
            ordered_cols = list(metadata["temporal_order"])  # ensure list
            if set(ordered_cols) == set(dataframe.columns):
                dataframe = dataframe[ordered_cols]
        except Exception:
            pass

    # Create train/test split
    n_samples = len(dataframe)
    n_train = int(n_samples * train_ratio)

    # Shuffle indices for random split (seeded if metadata contains 'seed')
    try:
        seed = metadata.get('seed') if isinstance(metadata, dict) else None
    except Exception:
        seed = None
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_df = dataframe.iloc[train_indices].reset_index(drop=True)
    test_df = dataframe.iloc[test_indices].reset_index(drop=True)

    # Save full dataset
    dataframe.to_csv(os.path.join(dataset_dir, f"{base_name}.csv"), index=False)
    
    # Save train/test splits
    train_df.to_csv(os.path.join(dataset_dir, f"{base_name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(dataset_dir, f"{base_name}_test.csv"), index=False)

    # Save adjacency matrix
    pd.DataFrame(adjacency_matrix,
                 index=metadata['temporal_order'],
                 columns=metadata['temporal_order']).to_csv(
        os.path.join(dataset_dir, f"{base_name}_adj_matrix.csv")
    )

    # Save metadata
    with open(os.path.join(dataset_dir, f"{base_name}_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Optionally save human-readable version
    with open(os.path.join(dataset_dir, f"{base_name}_meta.txt"), "w") as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    # Save a simple results file (removed root node information)
    with open(os.path.join(dataset_dir, f"{base_name}_results.txt"), "w") as f:
        f.write("Dataset generated successfully\n")

    # Save DCDI format files alongside current format
    data_arr = dataframe.to_numpy()
    dag_arr = adjacency_matrix
    save_dcdi_bundle(dataset_dir, data_arr, dag_arr)

    # Update CSV index with both train and test entries if provided
    if index_file:
        # Determine how to construct paths for the index
        use_absolute = os.getenv("SEA_INDEX_ABSOLUTE", "0") in ("1", "true", "True")
        base_dir_override = os.getenv("SEA_INDEX_BASE_DIR")

        # Compute base directory for relative paths
        if not use_absolute:
            if base_dir_override and base_dir_override.strip():
                base_root = os.path.abspath(base_dir_override.strip())
            else:
                # Default: make paths relative to the parent of the index directory
                # If index is at .../data/.../index.csv, this yields paths starting with 'data/...'
                index_dir = os.path.abspath(os.path.dirname(index_file))
                base_root = os.path.abspath(os.path.dirname(index_dir))

            rel_dir = os.path.relpath(dataset_dir, start=base_root)
            rel_dir = rel_dir.replace("\\", "/")
            fp_data = f"{rel_dir}/data.npy"
            fp_graph = f"{rel_dir}/graph.npy"
            fp_regime_path = f"{rel_dir}/regimes.csv"
            fp_intervention_path = f"{rel_dir}/interventions.csv"
        else:
            # Absolute paths
            fp_data = os.path.abspath(os.path.join(dataset_dir, "data.npy")).replace("\\", "/")
            fp_graph = os.path.abspath(os.path.join(dataset_dir, "graph.npy")).replace("\\", "/")
            fp_regime_path = os.path.abspath(os.path.join(dataset_dir, "regimes.csv")).replace("\\", "/")
            fp_intervention_path = os.path.abspath(os.path.join(dataset_dir, "interventions.csv")).replace("\\", "/")

        # Allow blank regime/intervention for baseline compatibility via env toggle
        blank_regime = os.getenv("SEA_INDEX_BLANK_REGIME", "0") in ("1", "true", "True")

        # Add train entry
        train_row = dict(
            fp_data=fp_data,
            fp_graph=fp_graph,
            fp_regime=("" if blank_regime else fp_regime_path),
            fp_intervention=("" if blank_regime else fp_intervention_path),
            split="train",
            n_samples=len(train_df),
            n_variables=train_df.shape[1]
        )
        append_to_csv(index_file, train_row)
        
        # Add test entry
        test_row = dict(
            fp_data=fp_data,
            fp_graph=fp_graph,
            fp_regime=("" if blank_regime else fp_regime_path),
            fp_intervention=("" if blank_regime else fp_intervention_path),
            split="test",
            n_samples=len(test_df),
            n_variables=test_df.shape[1]
        )
        append_to_csv(index_file, test_row)


def save_dataset(dataframe, adjacency_matrix, metadata, dataset_dir, base_name, index_file=None, split="test"):
    """
    Legacy function for backward compatibility. Now uses save_dataset_with_splits.
    """
    save_dataset_with_splits(dataframe, adjacency_matrix, metadata, dataset_dir, base_name, index_file)

def get_equation_type(complexity):
    return {
        "low": "linear",
        "medium": "non_linear",
        "high": "random"
    }[complexity]
