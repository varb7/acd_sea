import numpy as np
import pandas as pd
from scipy import stats

def compute_data_properties(data, true_adj_matrix):
    properties = {
        'dataset_mean': np.mean(data.values),
        'dataset_std': np.std(data.values),
        'dataset_kurtosis': stats.kurtosis(data.values.ravel()),
        'dataset_skewness': stats.skew(data.values.ravel()),
        'edge_density': np.sum(true_adj_matrix) / true_adj_matrix.size,
        'average_degree': np.sum(true_adj_matrix) / true_adj_matrix.shape[0],
        'sparsity': 1 - (np.sum(true_adj_matrix) / true_adj_matrix.size)
    }
    return properties
