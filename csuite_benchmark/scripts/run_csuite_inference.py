#!/usr/bin/env python3
"""
CSuite Inference Pipeline Integration

This script provides a bridge between the CSuite dataset collection
and the main inference pipeline, allowing seamless evaluation.
"""

import argparse
import os
import sys
from pathlib import Path
import logging

# Add the inference pipeline to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "inference_pipeline"))

from main import main as run_inference_pipeline


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_csuite_inference(
    datasets_dir: str = "csuite_benchmark/datasets",
    output_dir: str = "csuite_benchmark/results",
    algorithms: list = None,
    logger: logging.Logger = None
) -> int:
    """Run inference pipeline on CSuite datasets."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Running inference pipeline on CSuite datasets...")
    
    # Set up arguments for the main inference pipeline
    sys.argv = [
        "inference_pipeline/main.py",
        "--input-dir", datasets_dir,
        "--output-dir", output_dir
    ]
    
    if algorithms:
        sys.argv.extend(["--algorithms"] + algorithms)
    
    # Run the inference pipeline
    try:
        return run_inference_pipeline()
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        return 1


def main():
    """Main function for CSuite inference integration."""
    parser = argparse.ArgumentParser(
        description="Run inference pipeline on CSuite datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--datasets-dir', type=str,
                       default='csuite_benchmark/datasets',
                       help='Directory containing CSuite datasets')
    parser.add_argument('--output-dir', type=str,
                       default='csuite_benchmark/results',
                       help='Output directory for results')
    parser.add_argument('--algorithms', nargs='+',
                       help='Specific algorithms to run (default: all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Run inference
    return run_csuite_inference(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        logger=logger
    )


if __name__ == "__main__":
    sys.exit(main())

