import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np

def append_to_excel(index_fp, row):
    """Append one row to an Excel sheet, creating it if necessary."""
    if os.path.exists(index_fp):
        df = pd.read_excel(index_fp)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    # Keep a stable column order
    cols = ["fp_data", "fp_graph", "fp_regime", "fp_intervention", "split", "n_samples", "n_variables"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(index_fp), exist_ok=True)
    
    with pd.ExcelWriter(index_fp, engine="openpyxl", mode="w") as xls:
        df.to_excel(xls, index=False)

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

def save_dataset(dataframe, adjacency_matrix, metadata, dataset_dir, base_name, index_file=None, split="train"):
    os.makedirs(dataset_dir, exist_ok=True)

    # Align data columns to temporal order if provided
    if isinstance(metadata, dict) and "temporal_order" in metadata:
        try:
            ordered_cols = list(metadata["temporal_order"])  # ensure list
            if set(ordered_cols) == set(dataframe.columns):
                dataframe = dataframe[ordered_cols]
        except Exception:
            pass

    # Save data
    dataframe.to_csv(os.path.join(dataset_dir, f"{base_name}.csv"), index=False)

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

    # Update Excel index if provided
    if index_file:
        row = dict(
            fp_data=os.path.join(dataset_dir, "data.npy"),
            fp_graph=os.path.join(dataset_dir, "graph.npy"),
            fp_regime=os.path.join(dataset_dir, "regimes.csv"),
            fp_intervention=os.path.join(dataset_dir, "interventions.csv"),
            split=split,
            n_samples=dataframe.shape[0],
            n_variables=dataframe.shape[1]
        )
        append_to_excel(index_file, row)

def get_equation_type(complexity):
    return {
        "low": "linear",
        "medium": "non_linear",
        "high": "random"
    }[complexity]
