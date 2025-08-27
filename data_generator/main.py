# main.py

from generator.dag_generator import generate_meta_dataset, generate_meta_dataset_with_manufacturing_distributions
from generator.dag_generator_enhanced import generate_meta_dataset_with_diverse_configurations
from config import TOTAL_DATASETS, OUTPUT_DIR

if __name__ == "__main__":
    # Use diverse configurations by default (recommended for varied datasets)
    print("🚀 Generating diverse datasets with varied configurations...")
    print("=" * 60)
    print("Strategy: preset_variations (8 preset configs + variations)")
    print("This will generate datasets with different:")
    print("  - Categorical vs continuous ratios")
    print("  - Distribution type percentages")
    print("  - Noise levels")
    print("  - Categorical skewness")
    print("=" * 60)
    
    generate_meta_dataset_with_diverse_configurations(
        total_datasets=TOTAL_DATASETS, 
        output_dir=OUTPUT_DIR,
        config_strategy="preset_variations",  # Options: "preset_variations", "random", "gradient", "mixed"
        seed=42
    )
    
    # Alternative strategies (uncomment to use):
    
    # Strategy 1: Random configurations within parameter ranges
    # generate_meta_dataset_with_diverse_configurations(
    #     total_datasets=TOTAL_DATASETS, 
    #     output_dir=OUTPUT_DIR,
    #     config_strategy="random",
    #     seed=42
    # )
    
    # Strategy 2: Gradient configurations (systematic parameter variation)
    # generate_meta_dataset_with_diverse_configurations(
    #     total_datasets=TOTAL_DATASETS, 
    #     output_dir=OUTPUT_DIR,
    #     config_strategy="gradient",
    #     seed=42
    # )
    
    # Strategy 3: Mixed strategy (presets + random)
    # generate_meta_dataset_with_diverse_configurations(
    #     total_datasets=TOTAL_DATASETS, 
    #     output_dir=OUTPUT_DIR,
    #     config_strategy="mixed",
    #     seed=42
    # )
    
    # Strategy 4: Original manufacturing distributions (single config)
    # generate_meta_dataset_with_manufacturing_distributions(
    #     total_datasets=TOTAL_DATASETS, 
    #     output_dir=OUTPUT_DIR
    # )
    
    # Strategy 5: Original pipeline (no manufacturing distributions)
    # generate_meta_dataset(total_datasets=TOTAL_DATASETS, output_dir=OUTPUT_DIR)
