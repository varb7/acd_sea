"""
Configuration schema and validation for the data generation pipeline.

This module defines the complete configuration structure with validation,
defaults, and type checking for all data generation parameters.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json


@dataclass
class GraphStructureConfig:
    """Configuration for graph structure parameters."""
    num_nodes_range: List[int] = field(default_factory=lambda: [10, 20])
    root_nodes_percentage_range: List[float] = field(default_factory=lambda: [0.10, 0.30])
    edges_density_range: List[float] = field(default_factory=lambda: [0.30, 0.80])
    
    def __post_init__(self):
        """Validate graph structure parameters."""
        if len(self.num_nodes_range) != 2 or self.num_nodes_range[0] >= self.num_nodes_range[1]:
            raise ValueError("num_nodes_range must be [min, max] with min < max")
        if len(self.root_nodes_percentage_range) != 2:
            raise ValueError("root_nodes_percentage_range must be [min, max]")
        if len(self.edges_density_range) != 2:
            raise ValueError("edges_density_range must be [min, max]")


@dataclass
class DataGenerationConfig:
    """Configuration for data generation parameters."""
    num_samples_range: List[int] = field(default_factory=lambda: [1000, 50000])
    default_num_samples: int = 1000
    
    def __post_init__(self):
        """Validate data generation parameters."""
        if len(self.num_samples_range) != 2 or self.num_samples_range[0] >= self.num_samples_range[1]:
            raise ValueError("num_samples_range must be [min, max] with min < max")
        if self.default_num_samples < self.num_samples_range[0] or self.default_num_samples > self.num_samples_range[1]:
            raise ValueError("default_num_samples must be within num_samples_range")


@dataclass
class ManufacturingConfig:
    """Configuration for manufacturing-style distributions."""
    categorical_percentage: float = 0.10
    continuous_percentage: float = 0.90
    continuous_distributions: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.60,
        "truncated_normal": 0.30,
        "lognormal": 0.10
    })
    categorical_distributions: Dict[str, float] = field(default_factory=lambda: {
        "uniform": 0.50,
        "non_uniform": 0.50
    })
    noise_level: float = 0.005
    noise_type: str = "normal"
    noise_params: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "std": 0.005})
    
    def __post_init__(self):
        """Validate manufacturing configuration."""
        if not (0 <= self.categorical_percentage <= 1):
            raise ValueError("categorical_percentage must be between 0 and 1")
        if abs(self.categorical_percentage + self.continuous_percentage - 1.0) > 1e-6:
            raise ValueError("categorical_percentage + continuous_percentage must equal 1.0")
        
        # Validate continuous distributions sum to 1.0
        cont_sum = sum(self.continuous_distributions.values())
        if abs(cont_sum - 1.0) > 1e-6:
            raise ValueError("continuous_distributions must sum to 1.0")
        
        # Validate categorical distributions sum to 1.0
        cat_sum = sum(self.categorical_distributions.values())
        if abs(cat_sum - 1.0) > 1e-6:
            raise ValueError("categorical_distributions must sum to 1.0")


@dataclass
class GenerationRangesConfig:
    """Configuration for parameter ranges used in dynamic generation."""
    categorical_percentage: Tuple[float, float] = (0.02, 0.40)
    normal_percentage: Tuple[float, float] = (0.50, 0.85)
    truncated_normal_percentage: Tuple[float, float] = (0.10, 0.45)
    lognormal_percentage: Tuple[float, float] = (0.05, 0.25)
    uniform_categorical_percentage: Tuple[float, float] = (0.30, 0.70)
    non_uniform_categorical_percentage: Tuple[float, float] = (0.30, 0.70)
    noise_level: Tuple[float, float] = (0.001, 0.02)
    
    def __post_init__(self):
        """Validate generation ranges and coerce YAML lists to tuples."""
        # Coerce list -> tuple for all 2-length range fields
        range_field_names = [
            'categorical_percentage', 'normal_percentage', 'truncated_normal_percentage',
            'lognormal_percentage', 'uniform_categorical_percentage', 'non_uniform_categorical_percentage',
            'noise_level',
        ]
        for attr_name in range_field_names:
            value = getattr(self, attr_name)
            if isinstance(value, list) and len(value) == 2:
                setattr(self, attr_name, (value[0], value[1]))
        # Validate percentage ranges
        for attr_name in [
            'categorical_percentage', 'normal_percentage', 'truncated_normal_percentage',
            'lognormal_percentage', 'uniform_categorical_percentage', 'non_uniform_categorical_percentage',
        ]:
            value = getattr(self, attr_name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f"{attr_name} must be a tuple of (min, max)")
            if value[0] >= value[1]:
                raise ValueError(f"{attr_name} min must be less than max")
        # Validate noise level
        if not isinstance(self.noise_level, tuple) or len(self.noise_level) != 2:
            raise ValueError("noise_level must be a tuple of (min, max)")
        if self.noise_level[0] >= self.noise_level[1]:
            raise ValueError("noise_level min must be less than max")


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_dir: str = "causal_meta_dataset"
    save_metadata: bool = True
    save_graphs: bool = True
    save_adjacency_matrices: bool = True
    metadata_format: str = "json"  # json, yaml, pickle
    graph_format: str = "npy"  # npy, pickle, json
    adjacency_format: str = "csv"  # csv, json, npy


@dataclass
class GenerationStrategyConfig:
    """Configuration for generation strategies."""
    strategy: str = "random"  # random, preset_variations, gradient, mixed
    seed: Optional[int] = 42
    total_datasets: int = 100
    parallel_workers: int = 1
    progress_reporting: bool = True
    save_intermediate_results: bool = False


@dataclass
class DataGeneratorConfig:
    """Main configuration class for the data generator."""
    # Core settings
    total_datasets: int = 100
    output_dir: str = "causal_meta_dataset"
    
    # Component configurations
    graph_structure: GraphStructureConfig = field(default_factory=GraphStructureConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    manufacturing: ManufacturingConfig = field(default_factory=ManufacturingConfig)
    generation_ranges: GenerationRangesConfig = field(default_factory=GenerationRangesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    strategy: GenerationStrategyConfig = field(default_factory=GenerationStrategyConfig)
    
    # Additional settings
    log_level: str = "INFO"
    random_seed: Optional[int] = 42
    validate_config: bool = True
    
    def __post_init__(self):
        """Validate the complete configuration."""
        if self.validate_config:
            self._validate_config()
    
    def _validate_config(self):
        """Validate the entire configuration."""
        if self.total_datasets <= 0:
            raise ValueError("total_datasets must be positive")
        
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        
        # Validate that ranges are consistent
        if (self.generation_ranges.normal_percentage[0] + 
            self.generation_ranges.truncated_normal_percentage[0] + 
            self.generation_ranges.lognormal_percentage[0] > 1.0):
            raise ValueError("Minimum continuous distribution percentages sum to more than 1.0")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataGeneratorConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        if 'graph_structure' in config_dict:
            config_dict['graph_structure'] = GraphStructureConfig(**config_dict['graph_structure'])
        if 'data_generation' in config_dict:
            config_dict['data_generation'] = DataGenerationConfig(**config_dict['data_generation'])
        if 'manufacturing' in config_dict:
            config_dict['manufacturing'] = ManufacturingConfig(**config_dict['manufacturing'])
        if 'generation_ranges' in config_dict:
            config_dict['generation_ranges'] = GenerationRangesConfig(**config_dict['generation_ranges'])
        if 'output' in config_dict:
            config_dict['output'] = OutputConfig(**config_dict['output'])
        if 'strategy' in config_dict:
            config_dict['strategy'] = GenerationStrategyConfig(**config_dict['strategy'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'DataGeneratorConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DataGeneratorConfig':
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_legacy_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for backward compatibility."""
        return {
            "TOTAL_DATASETS": self.total_datasets,
            "OUTPUT_DIR": self.output_dir,
            "MANUFACTURING_CONFIG": self.manufacturing.__dict__,
            "GENERATION_CONFIG": {
                "categorical_percentage": self.generation_ranges.categorical_percentage,
                "normal_percentage": self.generation_ranges.normal_percentage,
                "truncated_normal_percentage": self.generation_ranges.truncated_normal_percentage,
                "lognormal_percentage": self.generation_ranges.lognormal_percentage,
                "uniform_categorical_percentage": self.generation_ranges.uniform_categorical_percentage,
                "non_uniform_categorical_percentage": self.generation_ranges.non_uniform_categorical_percentage,
                "noise_level": self.generation_ranges.noise_level,
                "graph_structure": self.graph_structure.__dict__,
                "data_generation": self.data_generation.__dict__,
            }
        }


def create_default_config() -> DataGeneratorConfig:
    """Create a default configuration."""
    return DataGeneratorConfig()


def create_minimal_config() -> DataGeneratorConfig:
    """Create a minimal configuration for quick testing."""
    config = DataGeneratorConfig()
    config.total_datasets = 5
    config.graph_structure.num_nodes_range = [5, 10]
    config.data_generation.num_samples_range = [100, 1000]
    return config


def create_large_scale_config() -> DataGeneratorConfig:
    """Create a configuration for large-scale dataset generation."""
    config = DataGeneratorConfig()
    config.total_datasets = 1000
    config.graph_structure.num_nodes_range = [20, 50]
    config.data_generation.num_samples_range = [10000, 100000]
    config.strategy.parallel_workers = 4
    return config
