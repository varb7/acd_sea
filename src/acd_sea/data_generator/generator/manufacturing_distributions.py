import numpy as np
from scipy.stats import truncnorm
import random

class ManufacturingDistributionManager:
    """
    Manages manufacturing-specific distributions for root nodes.
    Works with the existing SCDG framework.
    
    Supports two modes:
    1. Static mode: Uses fixed values from 'manufacturing' config section
    2. Range mode: Samples values from 'generation_ranges' for dataset diversity
    """
    
    def __init__(self, config, seed=None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.generation_ranges = config.get("generation_ranges", {})
        
        # Sample or use static values for this dataset instance
        self._sample_dataset_parameters()
        
    def _sample_dataset_parameters(self):
        """
        Sample parameters for this dataset from generation_ranges if available,
        otherwise use static values from the config.
        """
        # Categorical percentage
        if "categorical_percentage" in self.generation_ranges:
            range_val = self.generation_ranges["categorical_percentage"]
            self.categorical_percentage = self.rng.uniform(range_val[0], range_val[1])
        else:
            self.categorical_percentage = self.config.get("categorical_percentage", 0.10)
        
        # Continuous distribution probabilities
        if all(k in self.generation_ranges for k in ["normal_percentage", "truncated_normal_percentage", "lognormal_percentage"]):
            # Sample from ranges and normalize to sum to 1
            normal_pct = self.rng.uniform(*self.generation_ranges["normal_percentage"])
            truncated_pct = self.rng.uniform(*self.generation_ranges["truncated_normal_percentage"])
            lognormal_pct = self.rng.uniform(*self.generation_ranges["lognormal_percentage"])
            total = normal_pct + truncated_pct + lognormal_pct
            self.continuous_distributions = {
                "normal": normal_pct / total,
                "truncated_normal": truncated_pct / total,
                "lognormal": lognormal_pct / total
            }
        else:
            self.continuous_distributions = self.config.get("continuous_distributions", {
                "normal": 0.65, "truncated_normal": 0.25, "lognormal": 0.10
            })
        
        # Categorical distribution probabilities (uniform vs non-uniform)
        if all(k in self.generation_ranges for k in ["uniform_categorical_percentage", "non_uniform_categorical_percentage"]):
            # Sample from ranges and normalize to sum to 1
            uniform_pct = self.rng.uniform(*self.generation_ranges["uniform_categorical_percentage"])
            non_uniform_pct = self.rng.uniform(*self.generation_ranges["non_uniform_categorical_percentage"])
            total = uniform_pct + non_uniform_pct
            self.categorical_distributions = {
                "uniform": uniform_pct / total,
                "non_uniform": non_uniform_pct / total
            }
        else:
            self.categorical_distributions = self.config.get("categorical_distributions", {
                "uniform": 0.50, "non_uniform": 0.50
            })
        
        # Noise level
        if "noise_level" in self.generation_ranges:
            range_val = self.generation_ranges["noise_level"]
            self.noise_level = self.rng.uniform(range_val[0], range_val[1])
        else:
            self.noise_level = self.config.get("noise_level", 0.005)
        
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
        
        # Calculate number of categorical vs continuous nodes using sampled percentage
        num_categorical = max(1, int(num_roots * self.categorical_percentage))
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
            p=[self.categorical_distributions["uniform"], 
               self.categorical_distributions["non_uniform"]]
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
            p=[self.continuous_distributions["normal"],
               self.continuous_distributions["truncated_normal"],
               self.continuous_distributions["lognormal"]]
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