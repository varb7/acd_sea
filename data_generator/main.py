"""
Simplified main entry point for causal dataset generation.
Generate datasets from config file for algorithm testing.
"""

import argparse
import sys
from pathlib import Path
import yaml

try:
    from generator.dataset_generator import generate_all_datasets
    from generator.csuite_dag_generator import generate_csuite_meta_dataset as generate_csuite_datasets
    from simple_config import load_config, save_config_template
except ImportError:
    from data_generator.generator.dataset_generator import generate_all_datasets
    from data_generator.generator.csuite_dag_generator import generate_csuite_meta_dataset as generate_csuite_datasets
    from data_generator.simple_config import load_config, save_config_template


def main():
    """Main entry point for the data generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic causal datasets for algorithm testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default config
  python main.py --config configs/config.yaml      # Use specific config
  python main.py --config configs/config.yaml --mode csuite  # CSuite mode
  
  python main.py --save-config my_config.yaml      # Create config template
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml',
                       help='Path to configuration file (default: configs/config.yaml)')
    parser.add_argument('--mode', type=str, choices=['simple', 'csuite'], default='simple',
                       help='Generation mode: simple (default) or csuite')
    parser.add_argument('--save-config', type=str,
                       help='Save a configuration template and exit')
    
    args = parser.parse_args()
    
    # Save config template if requested
    if args.save_config:
        config = load_config()  # Get default config
        save_config_template(args.save_config, config)
        print(f"Configuration template saved to: {args.save_config}")
        return 0
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = load_config()
    
    # Print configuration summary
    print("=" * 60)
    print("Dataset Generation Configuration")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Number of datasets: {config.get('num_datasets', 100)}")
    print(f"Output directory: {config.get('output_dir', 'causal_meta_dataset')}")
    print(f"Seed: {config.get('seed', 42)}")
    print(f"Nodes range: {config.get('nodes_range', [10, 20])}")
    print(f"Samples range: {config.get('samples_range', [1000, 50000])}")
    print("=" * 60)
    print()
    
    # Generate datasets
    try:
        if args.mode == 'csuite':
            print(f"Generating CSuite-style datasets...")
            # Extract CSuite-specific parameters from config or use defaults
            patterns = config.get('csuite_patterns', None)
            num_nodes_range = tuple(config.get('nodes_range', [2, 5]))
            num_samples = config.get('samples_range', [1000, 1000])[0]  # Use first value
            output_dir = config.get('output_dir', 'csuite_datasets')
            seed = config.get('seed', 42)
            
            generate_csuite_datasets(
                patterns=patterns,
                num_nodes_range=num_nodes_range,
                num_samples=num_samples,
                output_dir=output_dir,
                seed=seed
            )
        else:
            print(f"Generating simple datasets...")
            generate_all_datasets(config)
        
        print()
        print("=" * 60)
        print("Generation complete!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
