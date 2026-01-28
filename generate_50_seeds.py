"""
Generate 50 data instances with seeds 1-50.
Updates existing seed files with new fixed cost parameters.
"""
import sys
from DH_config import ProblemConfig
from DH_data_gen import generate_supply_chain_data

def main():
    config = ProblemConfig(instance_type='full')

    print("=" * 60)
    print("GENERATING 50 DATA INSTANCES (seed 1-50)")
    print("=" * 60)

    for seed in range(1, 51):
        print(f"\nGenerating seed {seed}/50...")

        # Generate data with this seed
        data = generate_supply_chain_data(config, seed=seed)

        # Save to seed-specific file
        filepath = f"data/DH_data_full_seed{seed}.pkl"
        data.save(filepath)

        # Print progress
        if seed % 10 == 0:
            print(f"Progress: {seed}/50 completed")

    print("\n" + "=" * 60)
    print("ALL 50 INSTANCES GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
