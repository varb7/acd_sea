# main.py

import argparse
import sys
from pathlib import Path

from generator.dag_generator import generate_meta_dataset, generate_meta_dataset_with_manufacturing_distributions
from generator.dag_generator_enhanced import generate_meta_dataset_with_diverse_configurations
from config_loader import ConfigLoader, load_config_from_args, load_config_from_env
from config_schema import DataGeneratorConfig


def main():
    """Main entry point for the data generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic causal datasets with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default configuration
  python main.py --config config.yaml              # Use specific config file
  python main.py --preset minimal                  # Use minimal preset
  python main.py --total-datasets 50 --strategy random  # Override specific parameters
  python main.py --config config.yaml --output-dir my_data  # Override output directory
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', '-c', type=str, 
                             help='Path to configuration file (YAML, JSON, or Python)')
    config_group.add_argument('--preset', '-p', type=str, 
                             choices=['default', 'minimal', 'large_scale'],
                             help='Use a preset configuration')
    config_group.add_argument('--env', action='store_true',
                             help='Load configuration from environment variables')
    
    # Core parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--total-datasets', type=int,
                           help='Total number of datasets to generate')
    core_group.add_argument('--output-dir', type=str,
                           help='Output directory for generated datasets')
    core_group.add_argument('--seed', type=int,
                           help='Random seed for reproducibility')
    
    # Generation strategy
    strategy_group = parser.add_argument_group('Generation Strategy')
    strategy_group.add_argument('--strategy', type=str,
                               choices=['random', 'preset_variations', 'gradient', 'mixed'],
                               help='Generation strategy')
    strategy_group.add_argument('--workers', type=int,
                               help='Number of parallel workers')
    
    # Legacy options
    legacy_group = parser.add_argument_group('Legacy Options')
    legacy_group.add_argument('--use-legacy', action='store_true',
                             help='Use legacy configuration system (config.py)')
    legacy_group.add_argument('--legacy-strategy', type=str,
                             choices=['manufacturing', 'original'],
                             help='Legacy generation strategy')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without generating data')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.use_legacy:
            config = load_legacy_config()
        elif args.env:
            config = load_config_from_env()
        else:
            # Build override dictionary from command line args
            overrides = {}
            if args.total_datasets:
                overrides['total_datasets'] = args.total_datasets
            if args.output_dir:
                overrides['output_dir'] = args.output_dir
            if args.seed:
                overrides['random_seed'] = args.seed
                overrides['strategy'] = overrides.get('strategy', {})
                overrides['strategy']['seed'] = args.seed
            if args.strategy:
                overrides['strategy'] = overrides.get('strategy', {})
                overrides['strategy']['strategy'] = args.strategy
            if args.workers:
                overrides['strategy'] = overrides.get('strategy', {})
                overrides['strategy']['parallel_workers'] = args.workers
            
            loader = ConfigLoader()
            config = loader.load_config(
                config_path=args.config,
                preset=args.preset,
                override_dict=overrides if overrides else None
            )
        
        # Validate configuration if requested
        if args.validate_only:
            print("✅ Configuration is valid")
            if args.verbose:
                print_config_summary(config)
            return 0
        
        # Show configuration if dry run
        if args.dry_run:
            print("🔍 Dry run - Configuration:")
            print_config_summary(config)
            return 0
        
        # Generate datasets
        return run_generation(config, args.verbose)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def load_legacy_config():
    """Load configuration using the legacy system."""
    try:
        from config import TOTAL_DATASETS, OUTPUT_DIR, MANUFACTURING_CONFIG, GENERATION_CONFIG
        
        # Convert legacy config to new format
        config_dict = {
            'total_datasets': TOTAL_DATASETS,
            'output_dir': OUTPUT_DIR,
            'manufacturing': MANUFACTURING_CONFIG,
            'generation_ranges': GENERATION_CONFIG,
        }
        
        return DataGeneratorConfig.from_dict(config_dict)
    except ImportError as e:
        raise RuntimeError(f"Failed to load legacy configuration: {e}")


def print_config_summary(config: DataGeneratorConfig):
    """Print a summary of the configuration."""
    print(f"Total datasets: {config.total_datasets}")
    print(f"Output directory: {config.output_dir}")
    print(f"Strategy: {config.strategy.strategy}")
    print(f"Random seed: {config.random_seed}")
    print(f"Graph nodes range: {config.graph_structure.num_nodes_range}")
    print(f"Sample size range: {config.data_generation.num_samples_range}")
    print(f"Parallel workers: {config.strategy.parallel_workers}")


def run_generation(config: DataGeneratorConfig, verbose: bool = False):
    """Run the data generation process."""
    if verbose:
        print("🚀 Starting data generation...")
        print_config_summary(config)
        print("=" * 60)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Choose generation method based on strategy
    if config.strategy.strategy == "preset_variations":
        print("Using preset variations strategy...")
        generate_meta_dataset_with_diverse_configurations(
            total_datasets=config.total_datasets,
            output_dir=config.output_dir,
            config_strategy="preset_variations",
            seed=config.random_seed
        )
    elif config.strategy.strategy == "random":
        print("Using random configuration strategy...")
        generate_meta_dataset_with_diverse_configurations(
            total_datasets=config.total_datasets,
            output_dir=config.output_dir,
            config_strategy="random",
            seed=config.random_seed
        )
    elif config.strategy.strategy == "gradient":
        print("Using gradient configuration strategy...")
        generate_meta_dataset_with_diverse_configurations(
            total_datasets=config.total_datasets,
            output_dir=config.output_dir,
            config_strategy="gradient",
            seed=config.random_seed
        )
    elif config.strategy.strategy == "mixed":
        print("Using mixed configuration strategy...")
        generate_meta_dataset_with_diverse_configurations(
            total_datasets=config.total_datasets,
            output_dir=config.output_dir,
            config_strategy="mixed",
            seed=config.random_seed
        )
    else:
        # Fallback to manufacturing distributions
        print("Using manufacturing distributions strategy...")
        generate_meta_dataset_with_manufacturing_distributions(
            total_datasets=config.total_datasets,
            output_dir=config.output_dir,
            manufacturing_config=config.manufacturing.__dict__
        )
    
    print(f"✅ Successfully generated {config.total_datasets} datasets in {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
