import numpy as np
from scipy.stats import truncnorm
import random

class ManufacturingDistributionManager:
    """
    Manages manufacturing-specific distributions for root nodes.
    Works with the existing SCDG framework.
    """
    
    def __init__(self, config, seed=None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
    def assign_manufacturing_distributions(self, root_nodes):
        """
        Assign manufacturing-specific distributions to root nodes.
        Returns SCDG-compatible distribution configurations.
        
        Args:
            root_nodes: List of root node names
            
        Returns:
            dict: SCDG-compatible distribution configurations
        """
        root_ranges = {}
        num_roots = len(root_nodes)
        
        # Calculate number of categorical vs continuous nodes
        num_categorical = max(1, int(num_roots * self.config["categorical_percentage"]))
        num_continuous = num_roots - num_categorical
        
        # Shuffle root nodes for random assignment
        shuffled_roots = list(root_nodes)
        self.rng.shuffle(shuffled_roots)
        
        # Assign categorical distributions
        categorical_roots = shuffled_roots[:num_categorical]
        for node in categorical_roots:
            root_ranges[node] = self._generate_categorical_distribution(node)
        
        # Assign continuous distributions
        continuous_roots = shuffled_roots[num_categorical:]
        for node in continuous_roots:
            root_ranges[node] = self._generate_continuous_distribution(node)
            
        return root_ranges
    
    def _generate_categorical_distribution(self, node):
        """Generate categorical distribution (uniform or non-uniform)"""
        dist_type = self.rng.choice(
            ["uniform", "non_uniform"], 
            p=[self.config["categorical_distributions"]["uniform"], 
               self.config["categorical_distributions"]["non_uniform"]]
        )
        
        num_classes = self.rng.integers(2, 6)  # 2-5 categories
        
        if dist_type == "uniform":
            return {
                "dist": "categorical",
                "num_classes": num_classes
            }
        else:
            # Non-uniform: generate skewed probabilities
            skew_factor = self.rng.uniform(0.1, 0.9)
            probabilities = self._generate_skewed_probabilities(num_classes, skew_factor)
            return {
                "dist": "categorical_non_uniform",
                "num_classes": num_classes,
                "probabilities": probabilities
            }
    
    def _generate_continuous_distribution(self, node):
        """Generate continuous distribution (normal, truncated normal, or lognormal)"""
        dist_type = self.rng.choice(
            ["normal", "truncated_normal", "lognormal"],
            p=[self.config["continuous_distributions"]["normal"],
               self.config["continuous_distributions"]["truncated_normal"],
               self.config["continuous_distributions"]["lognormal"]]
        )
        
        if dist_type == "normal":
            return self._generate_normal_distribution()
        elif dist_type == "truncated_normal":
            return self._generate_truncated_normal_distribution()
        elif dist_type == "lognormal":
            return self._generate_lognormal_distribution()
    
    def _generate_normal_distribution(self):
        """Generate standard normal distribution parameters"""
        mean = self.rng.uniform(0, 10)
        std = self.rng.uniform(0.1, 5.0)
        
        return {
            "dist": "normal",
            "mean": mean,
            "std": std
        }
    
    def _generate_truncated_normal_distribution(self):
        """Generate truncated normal distribution parameters"""
        mean = self.rng.uniform(0, 10)
        std = self.rng.uniform(0.1, 5.0)
        low = mean - 2 * std  # Truncate at 2 std devs
        high = mean + 2 * std
        
        return {
            "dist": "truncated_normal",
            "mean": mean,
            "std": std,
            "low": low,
            "high": high
        }
    
    def _generate_lognormal_distribution(self):
        """Generate lognormal distribution parameters"""
        mean = self.rng.uniform(0, 2)
        sigma = self.rng.uniform(0.1, 1.0)
        
        return {
            "dist": "lognormal",
            "mean": mean,
            "sigma": sigma
        }
    
    def _generate_skewed_probabilities(self, num_classes, skew_factor):
        """Generate skewed probabilities for non-uniform categorical distribution"""
        # Create probabilities that favor lower classes
        base_probs = np.array([(1 - skew_factor) ** i for i in range(num_classes)])
        # Normalize to sum to 1
        return (base_probs / base_probs.sum()).tolist() 