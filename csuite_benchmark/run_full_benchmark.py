#!/usr/bin/env python3
"""
CSuite Full Benchmark Runner

This script orchestrates the complete CSuite benchmark pipeline:
1. Generate CSuite datasets
2. Run causal discovery evaluation
3. Analyze results
4. Generate comprehensive reports
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import logging


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(command: list, logger: logging.Logger) -> bool:
    """Run a command and return success status."""
    try:
        logger.info(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def generate_datasets(
    patterns: list = None,
    num_nodes_range: tuple = (2, 5),
    num_samples: int = 1000,
    output_dir: str = "csuite_benchmark/datasets",
    seed: int = 42,
    logger: logging.Logger = None
) -> bool:
    """Generate CSuite datasets."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Step 1: Generating CSuite datasets...")
    
    command = [
        sys.executable, "data_generator/generate_csuite_datasets.py",
        "--output", output_dir,
        "--samples", str(num_samples),
        "--seed", str(seed)
    ]
    
    if patterns:
        command.extend(["--patterns"] + patterns)
    else:
        command.append("--all-patterns")
    
    command.extend(["--nodes", str(num_nodes_range[0]), str(num_nodes_range[1])])
    
    return run_command(command, logger)


def run_evaluation(
    datasets_dir: str = "csuite_benchmark/datasets",
    results_dir: str = "csuite_benchmark/results",
    algorithms: list = None,
    patterns: list = None,
    logger: logging.Logger = None
) -> bool:
    """Run causal discovery evaluation."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Step 2: Running causal discovery evaluation...")
    
    command = [
        sys.executable, "csuite_benchmark/scripts/run_csuite_evaluation.py",
        "--datasets-dir", datasets_dir,
        "--results-dir", results_dir
    ]
    
    if algorithms:
        command.extend(["--algorithms"] + algorithms)
    
    if patterns:
        command.extend(["--patterns"] + patterns)
    
    return run_command(command, logger)


def analyze_results(
    results_file: str = "csuite_benchmark/results/causal_discovery_results.csv",
    output_dir: str = "csuite_benchmark/results/pattern_analysis",
    create_plots: bool = True,
    logger: logging.Logger = None
) -> bool:
    """Analyze evaluation results."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Step 3: Analyzing results...")
    
    command = [
        sys.executable, "csuite_benchmark/scripts/analyze_patterns.py",
        "--results-file", results_file,
        "--output-dir", output_dir
    ]
    
    if create_plots:
        command.append("--create-plots")
    
    return run_command(command, logger)


def generate_reports(
    results_file: str = "csuite_benchmark/results/causal_discovery_results.csv",
    output_dir: str = "csuite_benchmark/results",
    formats: list = None,
    logger: logging.Logger = None
) -> bool:
    """Generate benchmark reports."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Step 4: Generating benchmark reports...")
    
    if formats is None:
        formats = ["html", "json", "csv"]
    
    command = [
        sys.executable, "csuite_benchmark/scripts/generate_benchmark_report.py",
        "--results-file", results_file,
        "--output-dir", output_dir,
        "--formats"
    ] + formats
    
    return run_command(command, logger)


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(
        description="Run complete CSuite benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csuite_benchmark/run_full_benchmark.py --all-patterns
  python csuite_benchmark/run_full_benchmark.py --patterns chain collider --algorithms pc ges fges
  python csuite_benchmark/run_full_benchmark.py --quick-test
        """
    )
    
    # Dataset generation options
    gen_group = parser.add_argument_group('Dataset Generation')
    gen_group.add_argument('--patterns', nargs='+',
                          choices=['chain', 'collider', 'backdoor', 'mixed_confounding', 'weak_arrow', 'large_backdoor'],
                          help='Specific patterns to generate')
    gen_group.add_argument('--all-patterns', action='store_true',
                          help='Generate all available patterns')
    gen_group.add_argument('--nodes', nargs=2, type=int, metavar=('MIN', 'MAX'),
                          default=[2, 5], help='Range of number of nodes (default: 2 5)')
    gen_group.add_argument('--samples', type=int, default=1000,
                          help='Number of samples per dataset (default: 1000)')
    gen_group.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--algorithms', nargs='+',
                           default=['pc', 'ges', 'fges', 'fci', 'rfci', 'gfci'],
                           help='Algorithms to evaluate')
    eval_group.add_argument('--eval-patterns', nargs='+',
                           help='Specific patterns to evaluate (default: all generated)')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--datasets-dir', type=str,
                             default='csuite_benchmark/datasets',
                             help='Directory for generated datasets')
    output_group.add_argument('--results-dir', type=str,
                             default='csuite_benchmark/results',
                             help='Directory for evaluation results')
    output_group.add_argument('--no-plots', action='store_true',
                             help='Skip generating plots')
    output_group.add_argument('--report-formats', nargs='+',
                             default=['html', 'json', 'csv'],
                             choices=['html', 'json', 'csv'],
                             help='Report formats to generate')
    
    # Pipeline control
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument('--skip-generation', action='store_true',
                               help='Skip dataset generation (use existing datasets)')
    pipeline_group.add_argument('--skip-evaluation', action='store_true',
                               help='Skip evaluation (use existing results)')
    pipeline_group.add_argument('--skip-analysis', action='store_true',
                               help='Skip analysis (use existing analysis)')
    pipeline_group.add_argument('--skip-reports', action='store_true',
                               help='Skip report generation')
    
    # Quick test option
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with minimal datasets and algorithms')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting CSuite Full Benchmark Pipeline")
    
    # Handle quick test
    if args.quick_test:
        logger.info("Running quick test mode...")
        args.patterns = ['chain', 'collider']
        args.nodes = [3, 4]
        args.samples = 100
        args.algorithms = ['pc', 'ges', 'fges']
        args.all_patterns = False
    
    # Determine patterns
    if args.all_patterns:
        patterns = None  # Will use all patterns
    elif args.patterns:
        patterns = args.patterns
    else:
        patterns = ['chain', 'collider', 'backdoor']  # Default patterns
    
    # Track success
    success = True
    
    try:
        # Step 1: Generate datasets
        if not args.skip_generation:
            success &= generate_datasets(
                patterns=patterns,
                num_nodes_range=tuple(args.nodes),
                num_samples=args.samples,
                output_dir=args.datasets_dir,
                seed=args.seed,
                logger=logger
            )
        else:
            logger.info("Skipping dataset generation")
        
        # Step 2: Run evaluation
        if success and not args.skip_evaluation:
            success &= run_evaluation(
                datasets_dir=args.datasets_dir,
                results_dir=args.results_dir,
                algorithms=args.algorithms,
                patterns=args.eval_patterns,
                logger=logger
            )
        else:
            logger.info("Skipping evaluation")
        
        # Step 3: Analyze results
        if success and not args.skip_analysis:
            results_file = os.path.join(args.results_dir, "causal_discovery_results.csv")
            analysis_dir = os.path.join(args.results_dir, "pattern_analysis")
            success &= analyze_results(
                results_file=results_file,
                output_dir=analysis_dir,
                create_plots=not args.no_plots,
                logger=logger
            )
        else:
            logger.info("Skipping analysis")
        
        # Step 4: Generate reports
        if success and not args.skip_reports:
            results_file = os.path.join(args.results_dir, "causal_discovery_results.csv")
            success &= generate_reports(
                results_file=results_file,
                output_dir=args.results_dir,
                formats=args.report_formats,
                logger=logger
            )
        else:
            logger.info("Skipping report generation")
        
        # Final status
        if success:
            logger.info("🎉 CSuite benchmark pipeline completed successfully!")
            logger.info(f"Results available in: {args.results_dir}")
            logger.info(f"HTML report: {os.path.join(args.results_dir, 'csuite_benchmark_report.html')}")
            return 0
        else:
            logger.error("❌ CSuite benchmark pipeline failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
