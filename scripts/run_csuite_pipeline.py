"""
Unified orchestrator for the CSuite workflow.

Usage:
    python run_csuite_pipeline.py generate          # Step 1: Generate datasets
    python run_csuite_pipeline.py run              # Step 2: Run experiments
    python run_csuite_pipeline.py analyze          # Step 3: Analyze results
    python run_csuite_pipeline.py all              # Run all steps sequentially
"""

import argparse
import subprocess
import sys
from pathlib import Path


def print_step(step_name: str, description: str):
    """Print a formatted step header."""
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    print(description)
    print()


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    exists = filepath.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists


def run_generate(mode: str = "grid", config: str = None):
    """Step 1: Generate datasets."""
    print_step(
        "GENERATE DATASETS",
        f"Generating CSuite datasets using mode: {mode}"
    )
    
    data_gen_dir = Path(__file__).parent.parent / "src" / "acd_sea" / "data_generator"
    
    if mode == "grid":
        # Use experiment grid generator
        script = data_gen_dir / "experiments" / "generate_csuite_grid.py"
        cmd = [sys.executable, str(script)]
        if config:
            cmd.extend(["--config", config])
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(data_gen_dir.parent.parent.parent))
        
        if result.returncode == 0:
            index_file = data_gen_dir.parent.parent.parent / "data" / "csuite_grid_datasets" / "index.csv"
            print(f"\n[OK] Generation complete!")
            print(f"  Datasets: csuite_grid_datasets/")
            print(f"  Index: {index_file}")
            return True
        else:
            print(f"\n[ERROR] Generation failed with return code {result.returncode}")
            return False
            
    elif mode == "quick":
        # Use quick CSuite generator
        script = data_gen_dir / "main.py"
        config_file = config or (data_gen_dir / "configs" / "csuite_config.yaml")
        cmd = [sys.executable, str(script), "--mode", "csuite", "--config", str(config_file)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(data_gen_dir.parent.parent.parent))
        
        if result.returncode == 0:
            print(f"\n[OK] Generation complete!")
            print(f"  Check output directory specified in config")
            return True
        else:
            print(f"\n[ERROR] Generation failed with return code {result.returncode}")
            return False
    else:
        print(f"Unknown mode: {mode}. Use 'grid' or 'quick'")
        return False


def run_experiments(index_path: str = None, results_path: str = None, with_prior: bool = True):
    """Step 2: Run experiments on generated datasets."""
    print_step(
        "RUN EXPERIMENTS",
        "Running causal discovery algorithms on generated datasets"
    )
    
    # Auto-detect index file if not provided
    if not index_path:
        candidates = [
            Path("csuite_grid_datasets") / "index.csv",
            Path("data_generator") / "csuite_datasets_test" / "index.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                index_path = str(candidate)
                print(f"Auto-detected index: {index_path}")
                break
        
        if not index_path:
            print("✗ Could not find index.csv. Please specify with --index")
            return False
    
    # Set default results path
    if not results_path:
        results_path = "causal_discovery_results2/experiment_results.csv"
    
    # Check index exists
    index_file = Path(index_path)
    if not index_file.exists():
        print(f"[ERROR] Index file not found: {index_path}")
        return False
    
    print(f"  Index: {index_path}")
    print(f"  Results: {results_path}")
    print(f"  With prior knowledge: {with_prior}")
    
    # Run experiments
    inference_dir = Path(__file__).parent.parent / "src" / "acd_sea" / "inference_pipeline"
    script = inference_dir / "run_experiments.py"
    
    cmd = [
        sys.executable, str(script),
        "--index", index_path,
        "--results", results_path,
        "--with-prior", "true" if with_prior else "false"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(inference_dir.parent.parent.parent))
    
    if result.returncode == 0:
        print(f"\n[OK] Experiments complete!")
        print(f"  Results saved to: {results_path}")
        return True
    else:
        print(f"\n[ERROR] Experiments failed with return code {result.returncode}")
        return False


def run_analyze(results_path: str = None):
    """Step 3: Analyze experiment results."""
    print_step(
        "ANALYZE RESULTS",
        "Analyzing experiment results and computing metrics"
    )
    
    # Auto-detect results file if not provided
    if not results_path:
        candidates = [
            Path("causal_discovery_results2") / "experiment_results.csv",
            Path("causal_discovery_results2") / "causal_discovery_analysis.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                results_path = str(candidate)
                print(f"Auto-detected results: {results_path}")
                break
        
        if not results_path:
            print("[ERROR] Could not find results file. Please specify with --results")
            return False
    
    results_file = Path(results_path)
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return False
    
    print(f"  Results file: {results_path}")
    
    # Run analysis
    inference_dir = Path(__file__).parent.parent / "src" / "acd_sea" / "inference_pipeline"
    script = inference_dir / "analyze_results.py"
    
    cmd = [sys.executable, str(script), "--results-file", results_path]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(inference_dir.parent.parent.parent))
    
    if result.returncode == 0:
        print(f"\n[OK] Analysis complete!")
        return True
    else:
        print(f"\n[ERROR] Analysis failed with return code {result.returncode}")
        return False


def check_status():
    """Check the status of the pipeline."""
    print_step(
        "PIPELINE STATUS",
        "Checking current state of datasets and results"
    )
    
    print("Dataset directories:")
    check_file_exists(Path("csuite_grid_datasets") / "index.csv", "Experiment grid datasets")
    check_file_exists(Path("data_generator") / "csuite_datasets_test" / "index.csv", "Quick test datasets")
    check_file_exists(Path("data_generator") / "simple_test_datasets" / "index.csv", "Simple datasets")
    
    print("\nResult files:")
    check_file_exists(Path("causal_discovery_results2") / "experiment_results.csv", "Experiment results")
    check_file_exists(Path("causal_discovery_results2") / "causal_discovery_analysis.csv", "Legacy results")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CSuite Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate datasets (experiment grid)
  python run_csuite_pipeline.py generate

  # Generate quick test datasets
  python run_csuite_pipeline.py generate --mode quick

  # Run experiments
  python run_csuite_pipeline.py run

  # Run experiments with specific index
  python run_csuite_pipeline.py run --index csuite_grid_datasets/index.csv

  # Analyze results
  python run_csuite_pipeline.py analyze

  # Run full pipeline
  python run_csuite_pipeline.py all

  # Check status
  python run_csuite_pipeline.py status
        """
    )
    
    parser.add_argument(
        "action",
        choices=["generate", "run", "analyze", "all", "status"],
        help="Action to perform"
    )
    
    # Generation options
    parser.add_argument(
        "--mode",
        choices=["grid", "quick"],
        default="grid",
        help="Generation mode: 'grid' for experiment grid, 'quick' for quick test (default: grid)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (for generation)"
    )
    
    # Experiment options
    parser.add_argument(
        "--index",
        help="Path to dataset index.csv file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--results",
        help="Path to results CSV file (default: causal_discovery_results2/experiment_results.csv)"
    )
    parser.add_argument(
        "--with-prior",
        type=lambda x: x.lower() in ("true", "1", "yes", "y"),
        default=True,
        help="Run with prior knowledge (default: true)"
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.action == "generate":
        success = run_generate(mode=args.mode, config=args.config)
        
    elif args.action == "run":
        success = run_experiments(
            index_path=args.index,
            results_path=args.results,
            with_prior=args.with_prior
        )
        
    elif args.action == "analyze":
        success = run_analyze(results_path=args.results)
        
    elif args.action == "all":
        print("\n" + "=" * 70)
        print("RUNNING FULL PIPELINE")
        print("=" * 70)
        
        success = run_generate(mode=args.mode, config=args.config)
        if success:
            success = run_experiments(
                index_path=args.index,
                results_path=args.results,
                with_prior=args.with_prior
            )
        if success:
            success = run_analyze(results_path=args.results)
            
        if success:
            print("\n" + "=" * 70)
            print("[OK] FULL PIPELINE COMPLETE!")
            print("=" * 70)
        
    elif args.action == "status":
        check_status()
        return 0
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

