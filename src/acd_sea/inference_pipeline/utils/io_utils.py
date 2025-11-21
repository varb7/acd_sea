import os
import pandas as pd
import pickle

def load_datasets(base_dir):
    datasets = []
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_path):
            continue

        csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv') and not f.endswith('_adj_matrix.csv')]
        meta_files = [f for f in os.listdir(full_path) if f.endswith('_meta.pkl')]
        adj_matrix_files = [f for f in os.listdir(full_path) if f.endswith('_adj_matrix.csv')]

        if csv_files and meta_files and adj_matrix_files:
            data_path = os.path.join(full_path, csv_files[0])
            data = pd.read_csv(data_path)

            with open(os.path.join(full_path, meta_files[0]), 'rb') as f:
                metadata = pickle.load(f)

            adj_matrix_path = os.path.join(full_path, adj_matrix_files[0])
            true_adj_matrix = pd.read_csv(adj_matrix_path, index_col=0).values

            datasets.append({
                'data': data,
                'metadata': metadata,
                'true_adj_matrix': true_adj_matrix
            })
    return datasets

def save_results(results_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving results to {output_path}")
    results_df.to_csv(output_path, index=False)
