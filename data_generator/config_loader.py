"""
Configuration loader and management utilities for the data generation pipeline.

This module provides utilities to load, validate, and manage configurations
from various sources (YAML, JSON, environment variables, command line).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
import json

try:
    from .config_schema import (
        DataGeneratorConfig,
        create_default_config,
        create_minimal_config,
        create_large_scale_config,
    )
except ImportError:
    from config_schema import (
        DataGeneratorConfig,
        create_default_config,
        create_minimal_config,
        create_large_scale_config,
    )


class ConfigLoader:
    """Configuration loader with support for multiple sources and formats."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory to search for configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent
        self.configs_dir = self.config_dir / "configs"
        # Build default search order across base dir and configs/ subdir
        default_names = ["config.yaml", "config.yml", "config.json", "config.py"]
        self.default_config_paths = [
            *(self.config_dir / name for name in default_names),
            *(self.configs_dir / name for name in default_names),
        ]
    
    def load_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   config_type: Optional[str] = None,
                   preset: Optional[str] = None,
                   override_dict: Optional[Dict[str, Any]] = None) -> DataGeneratorConfig:
        """
        Load configuration from various sources.
        
        Args:
            config_path: Path to configuration file
            config_type: Type of configuration ('yaml', 'json', 'py', 'auto')
            preset: Preset configuration ('default', 'minimal', 'large_scale')
            override_dict: Dictionary of values to override in loaded config
            
        Returns:
            DataGeneratorConfig: Loaded and validated configuration
        """
        # Handle presets first
        if preset:
            config = self._load_preset(preset)
        elif config_path:
            config = self._load_from_file(config_path, config_type)
        else:
            config = self._auto_load_config()
        
        # Apply overrides if provided
        if override_dict:
            config = self._apply_overrides(config, override_dict)
        
        return config
    
    def _load_preset(self, preset: str) -> DataGeneratorConfig:
        """Load a preset configuration."""
        presets = {
            'default': create_default_config,
            'minimal': create_minimal_config,
            'large_scale': create_large_scale_config
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]()
    
    def _load_from_file(self, config_path: Union[str, Path], config_type: Optional[str] = None) -> DataGeneratorConfig:
        """Load configuration from a specific file."""
        config_path = self._resolve_config_path(Path(config_path))
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Auto-detect file type if not specified
        if not config_type:
            config_type = config_path.suffix.lower().lstrip('.')
        
        if config_type in ['yaml', 'yml']:
            return DataGeneratorConfig.from_yaml(config_path)
        elif config_type == 'json':
            return DataGeneratorConfig.from_json(config_path)
        elif config_type == 'py':
            return self._load_from_python(config_path)
        else:
            raise ValueError(f"Unsupported configuration file type: {config_type}")
    
    def _load_from_python(self, config_path: Path) -> DataGeneratorConfig:
        """Load configuration from Python file (legacy format)."""
        # Add the config file's directory to Python path
        sys.path.insert(0, str(config_path.parent))
        
        try:
            # Import the config module
            config_module = __import__(config_path.stem)
            
            # Convert legacy format to new format
            config_dict = {
                'total_datasets': getattr(config_module, 'TOTAL_DATASETS', 100),
                'output_dir': getattr(config_module, 'OUTPUT_DIR', 'causal_meta_dataset'),
                'manufacturing': getattr(config_module, 'MANUFACTURING_CONFIG', {}),
                'generation_ranges': getattr(config_module, 'GENERATION_CONFIG', {}),
            }
            
            return DataGeneratorConfig.from_dict(config_dict)
        
        finally:
            # Clean up the path
            if str(config_path.parent) in sys.path:
                sys.path.remove(str(config_path.parent))
    
    def _auto_load_config(self) -> DataGeneratorConfig:
        """Automatically find and load configuration from default locations."""
        for config_path in self.default_config_paths:
            if config_path.exists():
                try:
                    return self._load_from_file(config_path)
                except Exception as e:
                    print(f"Warning: Failed to load {config_path}: {e}")
                    continue
        
        # If no config file found, return default
        print("No configuration file found, using default configuration")
        return create_default_config()

    def _resolve_config_path(self, path: Path) -> Path:
        """Resolve a config path by checking common locations.

        Order: absolute path, CWD, config_dir, configs_dir.
        """
        candidates = [
            path,
            Path.cwd() / path,
            self.config_dir / path.name if not path.is_absolute() else path,
            self.configs_dir / path.name if not path.is_absolute() else path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Return original; caller will handle missing file
        return path
    
    def _apply_overrides(self, config: DataGeneratorConfig, overrides: Dict[str, Any]) -> DataGeneratorConfig:
        """Apply override values to configuration."""
        # Convert config to dict for easier manipulation
        config_dict = config.to_dict()
        
        # Apply overrides recursively
        self._deep_update(config_dict, overrides)
        
        # Convert back to DataGeneratorConfig
        return DataGeneratorConfig.from_dict(config_dict)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update dictionary with nested values."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config_template(self, output_path: Union[str, Path], preset: str = 'default') -> None:
        """Save a configuration template file."""
        output_path = Path(output_path)
        config = self._load_preset(preset)
        
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            config.to_yaml(output_path)
        elif output_path.suffix.lower() == '.json':
            config.to_json(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """Validate a configuration file without loading it."""
        try:
            self._load_from_file(config_path)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


def load_config_from_args() -> DataGeneratorConfig:
    """Load configuration from command line arguments."""
    parser = argparse.ArgumentParser(description='Data Generator Configuration')
    
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--preset', '-p', type=str, choices=['default', 'minimal', 'large_scale'],
                       help='Use a preset configuration')
    parser.add_argument('--output-dir', type=str, help='Output directory for generated datasets')
    parser.add_argument('--total-datasets', type=int, help='Total number of datasets to generate')
    parser.add_argument('--strategy', type=str, choices=['random', 'preset_variations', 'gradient', 'mixed'],
                       help='Generation strategy')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Build override dictionary from command line args
    overrides = {}
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.total_datasets:
        overrides['total_datasets'] = args.total_datasets
    if args.strategy:
        overrides['strategy'] = {'strategy': args.strategy}
    if args.seed:
        overrides['random_seed'] = args.seed
        overrides['strategy'] = overrides.get('strategy', {})
        overrides['strategy']['seed'] = args.seed
    if args.workers:
        overrides['strategy'] = overrides.get('strategy', {})
        overrides['strategy']['parallel_workers'] = args.workers
    
    # Load configuration
    loader = ConfigLoader()
    return loader.load_config(
        config_path=args.config,
        preset=args.preset,
        override_dict=overrides if overrides else None
    )


def load_config_from_env() -> DataGeneratorConfig:
    """Load configuration from environment variables."""
    overrides = {}
    
    # Map environment variables to config keys
    env_mappings = {
        'DATA_GEN_TOTAL_DATASETS': ('total_datasets', int),
        'DATA_GEN_OUTPUT_DIR': ('output_dir', str),
        'DATA_GEN_STRATEGY': ('strategy.strategy', str),
        'DATA_GEN_SEED': ('random_seed', int),
        'DATA_GEN_WORKERS': ('strategy.parallel_workers', int),
    }
    
    for env_var, (config_key, value_type) in env_mappings.items():
        if env_var in os.environ:
            try:
                value = value_type(os.environ[env_var])
                # Handle nested keys
                if '.' in config_key:
                    key_parts = config_key.split('.')
                    nested_dict = overrides
                    for part in key_parts[:-1]:
                        if part not in nested_dict:
                            nested_dict[part] = {}
                        nested_dict = nested_dict[part]
                    nested_dict[key_parts[-1]] = value
                else:
                    overrides[config_key] = value
            except ValueError as e:
                print(f"Warning: Invalid value for {env_var}: {e}")
    
    # Load base configuration and apply overrides
    loader = ConfigLoader()
    return loader.load_config(override_dict=overrides if overrides else None)


# Convenience functions for common use cases
def get_default_config() -> DataGeneratorConfig:
    """Get the default configuration."""
    return create_default_config()


def get_minimal_config() -> DataGeneratorConfig:
    """Get a minimal configuration for testing."""
    return create_minimal_config()


def get_large_scale_config() -> DataGeneratorConfig:
    """Get a large-scale configuration."""
    return create_large_scale_config()


def create_config_file(output_path: Union[str, Path], preset: str = 'default') -> None:
    """Create a configuration file from a preset."""
    loader = ConfigLoader()
    loader.save_config_template(output_path, preset)
    print(f"Configuration template saved to: {output_path}")


if __name__ == "__main__":
    # CLI interface for configuration management
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_loader.py <command> [options]")
        print("Commands:")
        print("  create <output_path> [preset] - Create configuration template")
        print("  validate <config_path> - Validate configuration file")
        print("  load [config_path] - Load and display configuration")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        preset = sys.argv[3] if len(sys.argv) > 3 else "default"
        create_config_file(output_path, preset)
    
    elif command == "validate":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        loader = ConfigLoader()
        is_valid = loader.validate_config_file(config_path)
        print(f"Configuration is {'valid' if is_valid else 'invalid'}")
    
    elif command == "load":
        config_path = sys.argv[2] if len(sys.argv) > 2 else None
        loader = ConfigLoader()
        config = loader.load_config(config_path)
        print("Loaded configuration:")
        print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
