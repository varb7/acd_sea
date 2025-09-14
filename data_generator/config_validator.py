"""
Configuration validation utilities for the data generation pipeline.

This module provides comprehensive validation for configuration files
and parameters to ensure they are correct before data generation begins.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import json

from .config_schema import DataGeneratorConfig, GraphStructureConfig, DataGenerationConfig, ManufacturingConfig, GenerationRangesConfig, OutputConfig, GenerationStrategyConfig


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Comprehensive configuration validator."""
    
    def __init__(self):
        """Initialize the validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: DataGeneratorConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate core settings
        self._validate_core_settings(config)
        
        # Validate component configurations
        self._validate_graph_structure(config.graph_structure)
        self._validate_data_generation(config.data_generation)
        self._validate_manufacturing(config.manufacturing)
        self._validate_generation_ranges(config.generation_ranges)
        self._validate_output(config.output)
        self._validate_strategy(config.strategy)
        
        # Validate cross-component consistency
        self._validate_cross_component_consistency(config)
        
        # Validate file system requirements
        self._validate_file_system(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_core_settings(self, config: DataGeneratorConfig):
        """Validate core configuration settings."""
        if config.total_datasets <= 0:
            self.errors.append("total_datasets must be positive")
        elif config.total_datasets > 10000:
            self.warnings.append("total_datasets is very large, generation may take a long time")
        
        if not isinstance(config.output_dir, str) or not config.output_dir.strip():
            self.errors.append("output_dir must be a non-empty string")
        
        if config.random_seed is not None and (not isinstance(config.random_seed, int) or config.random_seed < 0):
            self.errors.append("random_seed must be a non-negative integer")
        
        if config.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.errors.append("log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    
    def _validate_graph_structure(self, graph_config: GraphStructureConfig):
        """Validate graph structure configuration."""
        # Validate node range
        if len(graph_config.num_nodes_range) != 2:
            self.errors.append("num_nodes_range must have exactly 2 elements [min, max]")
        elif graph_config.num_nodes_range[0] >= graph_config.num_nodes_range[1]:
            self.errors.append("num_nodes_range min must be less than max")
        elif graph_config.num_nodes_range[0] < 2:
            self.errors.append("num_nodes_range min must be at least 2")
        elif graph_config.num_nodes_range[1] > 1000:
            self.warnings.append("num_nodes_range max is very large, generation may be slow")
        
        # Validate root percentage range
        if len(graph_config.root_nodes_percentage_range) != 2:
            self.errors.append("root_nodes_percentage_range must have exactly 2 elements [min, max]")
        elif not (0 <= graph_config.root_nodes_percentage_range[0] <= 1 and 
                 0 <= graph_config.root_nodes_percentage_range[1] <= 1):
            self.errors.append("root_nodes_percentage_range values must be between 0 and 1")
        elif graph_config.root_nodes_percentage_range[0] >= graph_config.root_nodes_percentage_range[1]:
            self.errors.append("root_nodes_percentage_range min must be less than max")
        
        # Validate edge density range
        if len(graph_config.edges_density_range) != 2:
            self.errors.append("edges_density_range must have exactly 2 elements [min, max]")
        elif not (0 <= graph_config.edges_density_range[0] <= 1 and 
                 0 <= graph_config.edges_density_range[1] <= 1):
            self.errors.append("edges_density_range values must be between 0 and 1")
        elif graph_config.edges_density_range[0] >= graph_config.edges_density_range[1]:
            self.errors.append("edges_density_range min must be less than max")
    
    def _validate_data_generation(self, data_config: DataGenerationConfig):
        """Validate data generation configuration."""
        # Validate sample range
        if len(data_config.num_samples_range) != 2:
            self.errors.append("num_samples_range must have exactly 2 elements [min, max]")
        elif data_config.num_samples_range[0] >= data_config.num_samples_range[1]:
            self.errors.append("num_samples_range min must be less than max")
        elif data_config.num_samples_range[0] < 10:
            self.errors.append("num_samples_range min must be at least 10")
        elif data_config.num_samples_range[1] > 1000000:
            self.warnings.append("num_samples_range max is very large, generation may be slow")
        
        # Validate default samples
        if (data_config.default_num_samples < data_config.num_samples_range[0] or 
            data_config.default_num_samples > data_config.num_samples_range[1]):
            self.errors.append("default_num_samples must be within num_samples_range")
    
    def _validate_manufacturing(self, mfg_config: ManufacturingConfig):
        """Validate manufacturing configuration."""
        # Validate percentages
        if not (0 <= mfg_config.categorical_percentage <= 1):
            self.errors.append("manufacturing.categorical_percentage must be between 0 and 1")
        
        if abs(mfg_config.categorical_percentage + mfg_config.continuous_percentage - 1.0) > 1e-6:
            self.errors.append("manufacturing categorical + continuous percentages must sum to 1.0")
        
        # Validate continuous distributions
        cont_sum = sum(mfg_config.continuous_distributions.values())
        if abs(cont_sum - 1.0) > 1e-6:
            self.errors.append("manufacturing.continuous_distributions must sum to 1.0")
        
        for dist_name, weight in mfg_config.continuous_distributions.items():
            if not (0 <= weight <= 1):
                self.errors.append(f"manufacturing.continuous_distributions.{dist_name} must be between 0 and 1")
        
        # Validate categorical distributions
        cat_sum = sum(mfg_config.categorical_distributions.values())
        if abs(cat_sum - 1.0) > 1e-6:
            self.errors.append("manufacturing.categorical_distributions must sum to 1.0")
        
        for dist_name, weight in mfg_config.categorical_distributions.items():
            if not (0 <= weight <= 1):
                self.errors.append(f"manufacturing.categorical_distributions.{dist_name} must be between 0 and 1")
        
        # Validate noise parameters
        if mfg_config.noise_level < 0:
            self.errors.append("manufacturing.noise_level must be non-negative")
        
        if mfg_config.noise_type not in ['normal', 'uniform', 'exponential']:
            self.errors.append("manufacturing.noise_type must be one of: normal, uniform, exponential")
    
    def _validate_generation_ranges(self, ranges_config: GenerationRangesConfig):
        """Validate generation ranges configuration."""
        # Validate all percentage ranges
        range_attrs = [
            'categorical_percentage', 'normal_percentage', 'truncated_normal_percentage',
            'lognormal_percentage', 'uniform_categorical_percentage', 'non_uniform_categorical_percentage'
        ]
        
        for attr_name in range_attrs:
            value = getattr(ranges_config, attr_name)
            if not isinstance(value, tuple) or len(value) != 2:
                self.errors.append(f"generation_ranges.{attr_name} must be a tuple of (min, max)")
            elif not (0 <= value[0] <= 1 and 0 <= value[1] <= 1):
                self.errors.append(f"generation_ranges.{attr_name} values must be between 0 and 1")
            elif value[0] >= value[1]:
                self.errors.append(f"generation_ranges.{attr_name} min must be less than max")
        
        # Validate noise level range
        if not isinstance(ranges_config.noise_level, tuple) or len(ranges_config.noise_level) != 2:
            self.errors.append("generation_ranges.noise_level must be a tuple of (min, max)")
        elif ranges_config.noise_level[0] < 0 or ranges_config.noise_level[1] < 0:
            self.errors.append("generation_ranges.noise_level values must be non-negative")
        elif ranges_config.noise_level[0] >= ranges_config.noise_level[1]:
            self.errors.append("generation_ranges.noise_level min must be less than max")
    
    def _validate_output(self, output_config: OutputConfig):
        """Validate output configuration."""
        if not isinstance(output_config.output_dir, str) or not output_config.output_dir.strip():
            self.errors.append("output.output_dir must be a non-empty string")
        
        if output_config.metadata_format not in ['json', 'yaml', 'pickle']:
            self.errors.append("output.metadata_format must be one of: json, yaml, pickle")
        
        if output_config.graph_format not in ['npy', 'pickle', 'json']:
            self.errors.append("output.graph_format must be one of: npy, pickle, json")
        
        if output_config.adjacency_format not in ['csv', 'json', 'npy']:
            self.errors.append("output.adjacency_format must be one of: csv, json, npy")
    
    def _validate_strategy(self, strategy_config: GenerationStrategyConfig):
        """Validate strategy configuration."""
        if strategy_config.strategy not in ['random', 'preset_variations', 'gradient', 'mixed']:
            self.errors.append("strategy.strategy must be one of: random, preset_variations, gradient, mixed")
        
        if strategy_config.seed is not None and (not isinstance(strategy_config.seed, int) or strategy_config.seed < 0):
            self.errors.append("strategy.seed must be a non-negative integer")
        
        if strategy_config.parallel_workers < 1:
            self.errors.append("strategy.parallel_workers must be at least 1")
        elif strategy_config.parallel_workers > 32:
            self.warnings.append("strategy.parallel_workers is very high, may cause resource issues")
    
    def _validate_cross_component_consistency(self, config: DataGeneratorConfig):
        """Validate consistency across different configuration components."""
        # Check that generation ranges are consistent with manufacturing config
        mfg_cat_pct = config.manufacturing.categorical_percentage
        gen_cat_min, gen_cat_max = config.generation_ranges.categorical_percentage
        
        if not (gen_cat_min <= mfg_cat_pct <= gen_cat_max):
            self.warnings.append("manufacturing.categorical_percentage is outside generation_ranges.categorical_percentage")
        
        # Check that continuous distribution ranges are reasonable
        cont_ranges = [
            config.generation_ranges.normal_percentage,
            config.generation_ranges.truncated_normal_percentage,
            config.generation_ranges.lognormal_percentage
        ]
        
        min_sum = sum(r[0] for r in cont_ranges)
        max_sum = sum(r[1] for r in cont_ranges)
        
        if min_sum > 1.0:
            self.errors.append("Minimum continuous distribution percentages sum to more than 1.0")
        elif max_sum < 0.5:
            self.warnings.append("Maximum continuous distribution percentages sum to less than 0.5")
        
        # Check that categorical distribution ranges are reasonable
        cat_ranges = [
            config.generation_ranges.uniform_categorical_percentage,
            config.generation_ranges.non_uniform_categorical_percentage
        ]
        
        min_sum = sum(r[0] for r in cat_ranges)
        max_sum = sum(r[1] for r in cat_ranges)
        
        if min_sum > 1.0:
            self.errors.append("Minimum categorical distribution percentages sum to more than 1.0")
        elif max_sum < 0.5:
            self.warnings.append("Maximum categorical distribution percentages sum to less than 0.5")
    
    def _validate_file_system(self, config: DataGeneratorConfig):
        """Validate file system requirements."""
        output_path = Path(config.output_dir)
        
        # Check if output directory can be created
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.errors.append(f"Cannot create output directory: {config.output_dir} (permission denied)")
        except OSError as e:
            self.errors.append(f"Cannot create output directory: {config.output_dir} ({e})")
        
        # Check available disk space (rough estimate)
        try:
            stat = os.statvfs(output_path.parent)
            free_space_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
            estimated_size_gb = self._estimate_output_size(config)
            
            if estimated_size_gb > free_space_gb * 0.9:  # Use 90% of free space as threshold
                self.warnings.append(f"Estimated output size ({estimated_size_gb:.1f} GB) may exceed available disk space")
        except (OSError, AttributeError):
            # statvfs not available on Windows or other issues
            pass
    
    def _estimate_output_size(self, config: DataGeneratorConfig) -> float:
        """Estimate the output size in GB."""
        # Rough estimation based on typical data sizes
        avg_nodes = sum(config.graph_structure.num_nodes_range) / 2
        avg_samples = sum(config.data_generation.num_samples_range) / 2
        
        # Estimate size per dataset (in bytes)
        # Data: samples * nodes * 8 bytes (float64)
        # Graph: nodes * nodes * 1 byte (boolean)
        # Adjacency: nodes * nodes * 8 bytes (float64)
        # Metadata: ~1KB
        data_size = avg_samples * avg_nodes * 8
        graph_size = avg_nodes * avg_nodes * 1
        adj_size = avg_nodes * avg_nodes * 8
        meta_size = 1024
        
        size_per_dataset = data_size + graph_size + adj_size + meta_size
        total_size = size_per_dataset * config.total_datasets
        
        return total_size / (1024**3)  # Convert to GB


def validate_config_file(config_path: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator()
    
    try:
        # Load configuration
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            return False, [f"Unsupported file format: {config_path.suffix}"], []
        
        # Convert to DataGeneratorConfig
        config = DataGeneratorConfig.from_dict(config_dict)
        
        # Validate
        return validator.validate_config(config)
        
    except Exception as e:
        return False, [f"Failed to load configuration: {e}"], []


def validate_config_dict(config_dict: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator()
    
    try:
        config = DataGeneratorConfig.from_dict(config_dict)
        return validator.validate_config(config)
    except Exception as e:
        return False, [f"Failed to create configuration: {e}"], []


if __name__ == "__main__":
    """CLI interface for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data generator configuration")
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    config_path = Path(args.config_path)
    is_valid, errors, warnings = validate_config_file(config_path)
    
    if is_valid:
        print("✅ Configuration is valid")
        if warnings and args.verbose:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
    else:
        print("❌ Configuration is invalid")
        print("\nErrors:")
        for error in errors:
            print(f"  ❌ {error}")
        
        if warnings and args.verbose:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        sys.exit(1)
