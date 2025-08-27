#!/usr/bin/env python3
"""
Dynamic Configuration Generator for Diverse Dataset Generation

This module provides tools to generate varied configurations for creating
diverse synthetic datasets while maintaining the base structure.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple

class DynamicConfigGenerator:
    """
    Generates dynamic configurations for diverse dataset generation.
    """
    
    def __init__(self, base_config: Dict, parameter_ranges: Dict, seed: Optional[int] = None):
        """
        Initialize the dynamic config generator.
        
        Args:
            base_config: Base configuration template
            parameter_ranges: Valid ranges for each parameter
            seed: Random seed for reproducibility
        """
        self.base_config = base_config.copy()
        self.parameter_ranges = parameter_ranges
        self.rng = np.random.default_rng(seed)
        
    def generate_random_config(self) -> Dict:
        """
        Generate a random configuration within the parameter ranges.
        
        Returns:
            dict: Random configuration
        """
        config = self.base_config.copy()
        
        # Generate random categorical vs continuous split
        cat_min, cat_max = self.parameter_ranges["categorical_percentage"]
        categorical_percentage = self.rng.uniform(cat_min, cat_max)
        continuous_percentage = 1.0 - categorical_percentage
        
        config["categorical_percentage"] = categorical_percentage
        config["continuous_percentage"] = continuous_percentage
        
        # Generate random continuous distribution breakdown
        normal_min, normal_max = self.parameter_ranges["normal_percentage"]
        truncated_min, truncated_max = self.parameter_ranges["truncated_normal_percentage"]
        lognormal_min, lognormal_max = self.parameter_ranges["lognormal_percentage"]
        
        # Ensure they sum to 1.0
        normal_pct = self.rng.uniform(normal_min, normal_max)
        truncated_pct = self.rng.uniform(truncated_min, truncated_max)
        lognormal_pct = self.rng.uniform(lognormal_min, lognormal_max)
        
        # Normalize to sum to 1.0
        total = normal_pct + truncated_pct + lognormal_pct
        config["continuous_distributions"] = {
            "normal": normal_pct / total,
            "truncated_normal": truncated_pct / total,
            "lognormal": lognormal_pct / total
        }
        
        # Generate random categorical distribution breakdown
        uniform_min, uniform_max = self.parameter_ranges["uniform_categorical_percentage"]
        non_uniform_min, non_uniform_max = self.parameter_ranges["non_uniform_categorical_percentage"]
        
        uniform_pct = self.rng.uniform(uniform_min, uniform_max)
        non_uniform_pct = self.rng.uniform(non_uniform_min, non_uniform_max)
        
        # Normalize to sum to 1.0
        total_cat = uniform_pct + non_uniform_pct
        config["categorical_distributions"] = {
            "uniform": uniform_pct / total_cat,
            "non_uniform": non_uniform_pct / total_cat
        }
        
        # Generate random noise level
        noise_min, noise_max = self.parameter_ranges["noise_level"]
        noise_level = self.rng.uniform(noise_min, noise_max)
        
        config["noise_level"] = noise_level
        config["noise_params"]["std"] = noise_level
        
        return config
    
    def generate_config_batch(self, num_configs: int) -> List[Dict]:
        """
        Generate a batch of random configurations.
        
        Args:
            num_configs: Number of configurations to generate
            
        Returns:
            list: List of random configurations
        """
        configs = []
        for i in range(num_configs):
            config = self.generate_random_config()
            config["config_id"] = f"dynamic_{i:03d}"
            configs.append(config)
        return configs
    
    def generate_targeted_config(self, 
                                target_categorical_percentage: Optional[float] = None,
                                target_normal_percentage: Optional[float] = None,
                                target_noise_level: Optional[float] = None) -> Dict:
        """
        Generate a configuration targeting specific parameter values.
        
        Args:
            target_categorical_percentage: Target categorical percentage
            target_normal_percentage: Target normal distribution percentage
            target_noise_level: Target noise level
            
        Returns:
            dict: Targeted configuration
        """
        config = self.base_config.copy()
        
        # Set targeted values if provided
        if target_categorical_percentage is not None:
            config["categorical_percentage"] = target_categorical_percentage
            config["continuous_percentage"] = 1.0 - target_categorical_percentage
        
        if target_normal_percentage is not None:
            # Adjust continuous distributions to target normal percentage
            remaining = 1.0 - target_normal_percentage
            truncated_pct = remaining * 0.75  # 75% of remaining to truncated
            lognormal_pct = remaining * 0.25  # 25% of remaining to lognormal
            
            config["continuous_distributions"] = {
                "normal": target_normal_percentage,
                "truncated_normal": truncated_pct,
                "lognormal": lognormal_pct
            }
        
        if target_noise_level is not None:
            config["noise_level"] = target_noise_level
            config["noise_params"]["std"] = target_noise_level
        
        return config
    
    def generate_gradient_configs(self, 
                                 param_name: str, 
                                 start_value: float, 
                                 end_value: float, 
                                 num_steps: int) -> List[Dict]:
        """
        Generate configurations with a gradient of a specific parameter.
        
        Args:
            param_name: Parameter to vary ('categorical_percentage', 'normal_percentage', 'noise_level')
            start_value: Starting value
            end_value: Ending value
            num_steps: Number of steps in the gradient
            
        Returns:
            list: List of configurations with gradient values
        """
        configs = []
        values = np.linspace(start_value, end_value, num_steps)
        
        for i, value in enumerate(values):
            if param_name == "categorical_percentage":
                config = self.generate_targeted_config(target_categorical_percentage=value)
            elif param_name == "normal_percentage":
                config = self.generate_targeted_config(target_normal_percentage=value)
            elif param_name == "noise_level":
                config = self.generate_targeted_config(target_noise_level=value)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            config["config_id"] = f"gradient_{param_name}_{i:03d}"
            configs.append(config)
        
        return configs

def create_diverse_configs_from_presets(preset_configs: Dict, 
                                       num_variations: int = 3,
                                       seed: Optional[int] = None) -> List[Dict]:
    """
    Create diverse configurations by adding random variations to preset configs.
    
    Args:
        preset_configs: Dictionary of preset configurations
        num_variations: Number of variations per preset
        seed: Random seed
        
    Returns:
        list: List of diverse configurations
    """
    rng = np.random.default_rng(seed)
    diverse_configs = []
    
    for preset_name, preset_config in preset_configs.items():
        # Add the original preset
        preset_config["config_id"] = f"preset_{preset_name}"
        diverse_configs.append(preset_config.copy())
        
        # Create variations
        for i in range(num_variations):
            variation = preset_config.copy()
            
            # Add small random variations (±10% of original values)
            variation["categorical_percentage"] *= rng.uniform(0.9, 1.1)
            variation["categorical_percentage"] = np.clip(variation["categorical_percentage"], 0.01, 0.5)
            
            variation["continuous_distributions"]["normal"] *= rng.uniform(0.9, 1.1)
            variation["continuous_distributions"]["truncated_normal"] *= rng.uniform(0.9, 1.1)
            variation["continuous_distributions"]["lognormal"] *= rng.uniform(0.9, 1.1)
            
            # Normalize continuous distributions
            total = sum(variation["continuous_distributions"].values())
            for key in variation["continuous_distributions"]:
                variation["continuous_distributions"][key] /= total
            
            variation["noise_level"] *= rng.uniform(0.8, 1.2)
            variation["noise_params"]["std"] = variation["noise_level"]
            
            variation["config_id"] = f"variation_{preset_name}_{i:02d}"
            diverse_configs.append(variation)
    
    return diverse_configs

def analyze_config_diversity(configs: List[Dict]) -> Dict:
    """
    Analyze the diversity of a set of configurations.
    
    Args:
        configs: List of configurations to analyze
        
    Returns:
        dict: Diversity analysis results
    """
    if not configs:
        return {}
    
    analysis = {
        "num_configs": len(configs),
        "categorical_percentages": [],
        "normal_percentages": [],
        "truncated_percentages": [],
        "lognormal_percentages": [],
        "noise_levels": [],
        "uniform_categorical_percentages": []
    }
    
    for config in configs:
        analysis["categorical_percentages"].append(config["categorical_percentage"])
        analysis["normal_percentages"].append(config["continuous_distributions"]["normal"])
        analysis["truncated_percentages"].append(config["continuous_distributions"]["truncated_normal"])
        analysis["lognormal_percentages"].append(config["continuous_distributions"]["lognormal"])
        analysis["noise_levels"].append(config["noise_level"])
        analysis["uniform_categorical_percentages"].append(config["categorical_distributions"]["uniform"])
    
    # Calculate statistics
    keys_to_process = [key for key in analysis if key != "num_configs" and analysis[key]]
    for key in keys_to_process:
        values = np.array(analysis[key])
        analysis[f"{key}_stats"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values))
        }
    
    return analysis 