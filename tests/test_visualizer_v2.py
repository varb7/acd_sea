import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Mock tkinter and matplotlib BEFORE importing the module
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = MagicMock()

# Now import the visualizer
import visualize_experiment_v2 as viz

class TestVisualizerV2(unittest.TestCase):
    def setUp(self):
        # Create dummy dataframe
        self.df = pd.DataFrame({
            'algorithm': ['AlgoA', 'AlgoA', 'AlgoB', 'AlgoB'],
            'use_prior': [True, False, True, False],
            'num_samples': [100, 100, 100, 100],
            'edge_density': [0.1, 0.1, 0.1, 0.1],
            'shd': [1.0, 2.0, 1.5, 2.5],
            'recall': [0.8, 0.7, 0.9, 0.6],
            'equation_type': ['linear', 'linear', 'linear', 'linear'],
            'var_type_tag': ['continuous', 'continuous', 'continuous', 'continuous'],
            'root_distribution_type': ['gaussian', 'gaussian', 'gaussian', 'gaussian'],
            'root_variation_level': [0, 0, 0, 0],
            'root_mean_bias': [0, 0, 0, 0],
            'noise_type': ['gaussian', 'gaussian', 'gaussian', 'gaussian'],
            'noise_intensity_level': [0, 0, 0, 0]
        })
        
        # Patch the global df in the module
        viz.df = self.df
        
        # Configure StringVar mock to return 'Line' by default to avoid crash in __init__
        sys.modules['tkinter'].StringVar.return_value.get.return_value = 'Line'
        
        # Mock root
        self.root = MagicMock()
        self.app = viz.InteractiveViz(self.root)

    def test_plot_line(self):
        print("Testing Line Plot...")
        self.app.plot_type_var.get.return_value = 'Line'
        self.app.metric_var.get.return_value = 'shd'
        self.app.x_var.get.return_value = 'num_samples'
        self.app.style_var.get.return_value = 'Default'
        self.app.facet_row_var.get.return_value = 'None'
        self.app.facet_col_var.get.return_value = 'None'
        self.app.compare_prior_var.get.return_value = False
        self.app.show_ci_var.get.return_value = True
        
        # Also need to mock filter vars to return 'All'
        self.app.algo_var.get.return_value = 'All'
        self.app.prior_var.get.return_value = 'All'
        self.app.eq_type_var.get.return_value = 'All'
        self.app.var_type_var.get.return_value = 'All'
        self.app.root_dist_var.get.return_value = 'All'
        self.app.root_var_level.get.return_value = 'All'
        self.app.root_mean_bias.get.return_value = 'All'
        self.app.noise_type_var.get.return_value = 'All'
        self.app.noise_intensity_var.get.return_value = 'All'
        
        self.app.plot_graph()
        print("Line Plot OK")

    def test_plot_heatmap(self):
        print("Testing Heatmap Plot...")
        self.app.plot_type_var.get.return_value = 'Heatmap'
        self.app.metric_var.get.return_value = 'shd'
        self.app.x_var.get.return_value = 'num_samples'
        self.app.y_var.get.return_value = 'algorithm'
        self.app.style_var.get.return_value = 'Default'
        self.app.facet_row_var.get.return_value = 'None'
        self.app.facet_col_var.get.return_value = 'None'
        self.app.compare_prior_var.get.return_value = False
        
        # Mock filters
        self.app.algo_var.get.return_value = 'All'
        self.app.prior_var.get.return_value = 'All'
        self.app.eq_type_var.get.return_value = 'All'
        self.app.var_type_var.get.return_value = 'All'
        self.app.root_dist_var.get.return_value = 'All'
        self.app.root_var_level.get.return_value = 'All'
        self.app.root_mean_bias.get.return_value = 'All'
        self.app.noise_type_var.get.return_value = 'All'
        self.app.noise_intensity_var.get.return_value = 'All'

        self.app.plot_graph()
        print("Heatmap Plot OK")

    def test_plot_scatter(self):
        print("Testing Scatter Plot...")
        self.app.plot_type_var.get.return_value = 'Scatter'
        self.app.metric_var.get.return_value = 'shd' 
        self.app.x_var.get.return_value = 'shd'
        self.app.y_var.get.return_value = 'recall'
        self.app.style_var.get.return_value = 'Default'
        self.app.facet_row_var.get.return_value = 'None'
        self.app.facet_col_var.get.return_value = 'None'
        self.app.compare_prior_var.get.return_value = False

        # Mock filters
        self.app.algo_var.get.return_value = 'All'
        self.app.prior_var.get.return_value = 'All'
        self.app.eq_type_var.get.return_value = 'All'
        self.app.var_type_var.get.return_value = 'All'
        self.app.root_dist_var.get.return_value = 'All'
        self.app.root_var_level.get.return_value = 'All'
        self.app.root_mean_bias.get.return_value = 'All'
        self.app.noise_type_var.get.return_value = 'All'
        self.app.noise_intensity_var.get.return_value = 'All'

        self.app.plot_graph()
        print("Scatter Plot OK")

if __name__ == '__main__':
    unittest.main()
