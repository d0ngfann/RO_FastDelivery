"""
DH_main.py
Main execution script for the C&CG robust optimization algorithm.
Runs sensitivity analysis over different uncertainty budget (Gamma) values.
"""

"""
  # Seed 5, Mixed scenario, Gamma=20                                                                                                                                           
  python3 DH_main.py full 20 --seed 5 --di Mixed    
"""      

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from DH_config import ProblemConfig, SensitivityConfig
from DH_data_gen import SupplyChainData, generate_supply_chain_data
from DH_algo import CCGAlgorithm, print_solution_summary


def run_single_gamma(data, config, gamma):
    """
    Run C&CG algorithm for a single Gamma value.

    Args:
        data: SupplyChainData instance
        config: ProblemConfig instance
        gamma: Uncertainty budget value

    Returns:
        dict: Results from C&CG algorithm
    """
    print("\n" + "=" * 80)
    print(f"RUNNING C&CG FOR Γ = {gamma}")
    print("=" * 80)

    # Set gamma
    config.set_gamma(gamma)

    # Create and run C&CG algorithm
    ccg = CCGAlgorithm(data, config)
    results = ccg.run()

    # Print solution summary
    if results['optimal_solution'] is not None:
        print_solution_summary(results['optimal_solution'], data)

    return results


def run_sensitivity_analysis(instance_type='toy', gamma_values=None):
    """
    Run sensitivity analysis for different Gamma values.

    Args:
        instance_type: 'toy' or 'full'
        gamma_values: List of gamma values to test (if None, uses default range)

    Returns:
        pd.DataFrame: Results summary
    """
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS - {instance_type.upper()} INSTANCE")
    print("=" * 80)

    # Initialize configuration
    config = ProblemConfig(instance_type=instance_type)

    # Load or generate data
    if os.path.exists(config.data_file):
        print(f"Loading data from {config.data_file}...")
        data = SupplyChainData.load(config.data_file)
        data.summary()
    else:
        print(f"Data file not found. Generating new data...")
        data = generate_supply_chain_data(config, seed=42)
        data.save(config.data_file)
        data.summary()

    # Determine gamma values to test
    if gamma_values is None:
        sens_config = SensitivityConfig(config.R)
        gamma_values = sens_config.gamma_values

    print(f"\nTesting Gamma values: {gamma_values}")

    # Results storage
    results_list = []

    # Run for each gamma
    for gamma in gamma_values:
        try:
            results = run_single_gamma(data, config, gamma)

            # Extract key metrics
            row = {
                'Gamma': gamma,
                'Converged': results['converged'],
                'Iterations': results['iterations'],
                'Total_Time': results['total_time'],
                'Optimal_Value': results['optimal_value'] if results['optimal_value'] is not None else float('nan'),
                'LB': results['LB'],
                'UB': results['UB'],
                'Gap': results['gap'],
                'Num_Scenarios': results['num_scenarios']
            }

            results_list.append(row)

            # Save intermediate results
            df_temp = pd.DataFrame(results_list)
            os.makedirs(config.results_dir, exist_ok=True)
            temp_file = os.path.join(config.results_dir, f"DH_sensitivity_{instance_type}_temp.csv")
            df_temp.to_csv(temp_file, index=False)
            print(f"\nIntermediate results saved to {temp_file}")

        except Exception as e:
            print(f"\nERROR: Failed to run Γ = {gamma}")
            print(f"Exception: {str(e)}")
            import traceback
            traceback.print_exc()

            # Log failed run
            row = {
                'Gamma': gamma,
                'Converged': False,
                'Iterations': 0,
                'Total_Time': 0,
                'Optimal_Value': float('nan'),
                'LB': float('nan'),
                'UB': float('nan'),
                'Gap': float('nan'),
                'Num_Scenarios': 0
            }
            results_list.append(row)

    # Create results DataFrame
    df_results = pd.DataFrame(results_list)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.results_dir, f"DH_sensitivity_{instance_type}_{timestamp}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return df_results


def plot_sensitivity_results(df_results, instance_type='toy'):
    """
    Create plots for sensitivity analysis results.

    Args:
        df_results: DataFrame with results
        instance_type: 'toy' or 'full'
    """
    try:
        import matplotlib.pyplot as plt

        # Filter out failed runs
        df_valid = df_results[df_results['Converged'] == True].copy()

        if len(df_valid) == 0:
            print("No valid results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Optimal Value vs Gamma
        ax1 = axes[0, 0]
        ax1.plot(df_valid['Gamma'], df_valid['Optimal_Value'], marker='o', linewidth=2)
        ax1.set_xlabel('Uncertainty Budget (Γ)', fontsize=12)
        ax1.set_ylabel('Optimal Objective Value', fontsize=12)
        ax1.set_title('Robust Profit vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Number of Scenarios vs Gamma
        ax2 = axes[0, 1]
        ax2.plot(df_valid['Gamma'], df_valid['Num_Scenarios'], marker='s', linewidth=2, color='green')
        ax2.set_xlabel('Uncertainty Budget (Γ)', fontsize=12)
        ax2.set_ylabel('Number of Critical Scenarios', fontsize=12)
        ax2.set_title('Critical Scenarios vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Iterations vs Gamma
        ax3 = axes[1, 0]
        ax3.plot(df_valid['Gamma'], df_valid['Iterations'], marker='^', linewidth=2, color='orange')
        ax3.set_xlabel('Uncertainty Budget (Γ)', fontsize=12)
        ax3.set_ylabel('Number of Iterations', fontsize=12)
        ax3.set_title('Convergence Iterations vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Total Time vs Gamma
        ax4 = axes[1, 1]
        ax4.plot(df_valid['Gamma'], df_valid['Total_Time'], marker='d', linewidth=2, color='red')
        ax4.set_xlabel('Uncertainty Budget (Γ)', fontsize=12)
        ax4.set_ylabel('Total Time (seconds)', fontsize=12)
        ax4.set_title('Computation Time vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"result/DH_sensitivity_{instance_type}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")

        plt.close()

    except ImportError:
        print("Matplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")


def main():
    """Main execution function."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run C&CG Algorithm for Robust Supply Chain Optimization')
    parser.add_argument('instance_type', nargs='?', default='toy', choices=['toy', 'full'],
                        help='Instance type: toy or full (default: toy)')
    parser.add_argument('gamma', nargs='?', type=int, default=None,
                        help='Single Gamma value to run (optional, runs sensitivity analysis if not specified)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Dataset seed number (1-50 for full, 1-5 for toy). If not specified, uses default data.')
    parser.add_argument('--di', '--di_scenario', dest='di_scenario', default='HD',
                        choices=['HD', 'MD', 'LD', 'Mixed'],
                        help='DI scenario to use: HD (High), MD (Medium), LD (Low), or Mixed (default: HD)')

    args = parser.parse_args()

    instance_type = args.instance_type
    single_gamma = args.gamma
    seed = args.seed
    di_scenario = args.di_scenario

    print("\n" + "=" * 80)
    print("ROBUST SUPPLY CHAIN OPTIMIZATION - C&CG ALGORITHM")
    print("=" * 80)
    print(f"Instance Type: {instance_type.upper()}")
    if seed:
        print(f"Dataset Seed: {seed}")
    print(f"DI Scenario: {di_scenario}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize configuration and load data
    config = ProblemConfig(instance_type=instance_type)

    # Load data based on seed
    if seed is not None:
        data_file = f"data/DH_data_{instance_type}_seed{seed}.pkl"
        if not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            print(f"Available seeds for {instance_type}: ", end="")
            import glob
            available = glob.glob(f"data/DH_data_{instance_type}_seed*.pkl")
            if available:
                seeds = [int(f.split('seed')[1].split('.pkl')[0]) for f in available]
                print(f"{min(seeds)}-{max(seeds)}")
            else:
                print("None")
            sys.exit(1)
        print(f"Loading data from {data_file}...")
        data = SupplyChainData.load(data_file)
    else:
        # Use default data file
        if os.path.exists(config.data_file):
            print(f"Loading data from {config.data_file}...")
            data = SupplyChainData.load(config.data_file)
        else:
            print(f"Data file not found. Generating new data...")
            data = generate_supply_chain_data(config, seed=42)
            data.save(config.data_file)

    # Apply selected DI scenario
    if hasattr(data, 'DI_scenarios') and di_scenario in data.DI_scenarios:
        print(f"\nApplying DI scenario: {di_scenario}")
        DI_matrix = data.DI_scenarios[di_scenario]
        for m in range(config.M):
            for k in range(config.K):
                data.DI[(m, k)] = DI_matrix[k][m]
        print(f"DI values:")
        for k in range(config.K):
            k_param = data.DI_k_params[di_scenario][k]
            print(f"  Product {k}: {[f'{DI_matrix[k][m]:.3f}' for m in range(config.M)]} (k={k_param:.3f})")
    else:
        print(f"\nWarning: DI scenario '{di_scenario}' not found in data. Using default DI values.")

    data.summary()

    # Run either single gamma or sensitivity analysis
    start_time = time.time()

    if single_gamma is not None:
        # Run single gamma value
        print(f"\nTesting Gamma value: {single_gamma}")

        results = run_single_gamma(data, config, single_gamma)

        # Print results
        print("\n" + "=" * 80)
        print("SINGLE RUN RESULTS")
        print("=" * 80)
        print(f"Gamma: {single_gamma}")
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Total Time: {results['total_time']:.2f}s")
        print(f"Optimal Value: {results['optimal_value']:.2f}")
        print(f"Lower Bound: {results['lower_bound']:.2f}")
        print(f"Upper Bound: {results['upper_bound']:.2f}")
        print(f"Gap: {results['gap']:.6f}")
        print("=" * 80)

        # Save single run result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"_seed{seed}" if seed else ""
        output_file = f"result/DH_single_{instance_type}{seed_str}_{di_scenario}_gamma{single_gamma}_{timestamp}.csv"
        os.makedirs("result", exist_ok=True)
        df_single = pd.DataFrame([{
            'Gamma': single_gamma,
            'Seed': seed if seed else 'default',
            'DI_Scenario': di_scenario,
            'Converged': results['converged'],
            'Iterations': results['iterations'],
            'Total_Time': results['total_time'],
            'Optimal_Value': results['optimal_value'],
            'LB': results['lower_bound'],
            'UB': results['upper_bound'],
            'Gap': results['gap'],
            'Num_Scenarios': len(results['critical_scenarios']) if results['critical_scenarios'] else 0
        }])
        df_single.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    else:
        # Run sensitivity analysis
        df_results = run_sensitivity_analysis(instance_type=instance_type)

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df_results.to_string(index=False))
        print("=" * 80)

        # Create plots
        plot_sensitivity_results(df_results, instance_type=instance_type)

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
