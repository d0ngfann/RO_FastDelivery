"""
generate_multiple_datasets.py

Generate 50 different datasets for the robust supply chain optimization problem.
Each dataset contains 4 DI scenarios (HD, MD, LD, Mixed) with different k parameters.

Usage:
    python3 generate_multiple_datasets.py
"""

import os
from DH_config import ProblemConfig
from DH_data_gen import generate_supply_chain_data

def generate_multiple_datasets(num_datasets=50, instance_type='full'):
    """
    Generate multiple datasets with different random seeds.

    Args:
        num_datasets: Number of datasets to generate (default: 50)
        instance_type: 'toy' or 'full'
    """
    print("=" * 80)
    print(f"GENERATING {num_datasets} DATASETS - {instance_type.upper()} INSTANCE")
    print("=" * 80)

    # Create data directory if not exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Initialize configuration
    config = ProblemConfig(instance_type=instance_type)

    # Generate datasets with seeds 1 to num_datasets
    for seed in range(1, num_datasets + 1):
        print(f"\n{'='*80}")
        print(f"Generating Dataset {seed}/{num_datasets} (seed={seed})")
        print('='*80)

        # Generate data
        data = generate_supply_chain_data(config, seed=seed)

        # Save with seed-specific filename
        filepath = f"{data_dir}/DH_data_{instance_type}_seed{seed}.pkl"
        data.save(filepath)

        # Print summary
        print(f"\nDataset {seed} Summary:")
        print(f"  Seed: {seed}")
        print(f"  Filepath: {filepath}")
        print(f"  DI Scenarios: HD, MD, LD, Mixed")

        # Print sample DI values from HD scenario
        if data.DI_scenarios:
            print(f"\n  Sample DI values (HD scenario):")
            for k in range(config.K):
                di_vec = data.DI_scenarios['HD'][k]
                k_param = data.DI_k_params['HD'][k]
                print(f"    Product {k}: [1.000, {di_vec[1]:.3f}, {di_vec[2]:.3f}] (k={k_param:.3f})")

    print("\n" + "=" * 80)
    print(f"✅ COMPLETED: Generated {num_datasets} datasets")
    print("=" * 80)
    print(f"\nFiles saved in: {data_dir}/")
    print(f"Filename pattern: DH_data_{instance_type}_seed[1-{num_datasets}].pkl")
    print(f"\nEach dataset contains 4 DI scenarios:")
    print(f"  - HD (High Demand sensitivity):   k ∈ [0.667, 1.000]")
    print(f"  - MD (Medium Demand sensitivity): k ∈ [0.333, 0.667]")
    print(f"  - LD (Low Demand sensitivity):    k ∈ [0.000, 0.333]")
    print(f"  - Mixed: Product 0 (HD), Product 1 (MD), Product 2 (LD)")
    print("\nTo run experiments:")
    print(f"  python3 DH_main.py {instance_type} 10 --seed 1 --di HD")
    print(f"  python3 DH_main.py {instance_type} 10 --seed 1 --di Mixed")
    print()


if __name__ == "__main__":
    # Generate 50 full instance datasets
    generate_multiple_datasets(num_datasets=50, instance_type='full')

    print("\n" + "=" * 80)
    print("Optional: Generate toy instance datasets for testing")
    print("=" * 80)
    response = input("Generate 5 toy instance datasets? (y/n): ")

    if response.lower() == 'y':
        generate_multiple_datasets(num_datasets=5, instance_type='toy')
