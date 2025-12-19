"""
Quick test script to verify full-size model operates correctly.
Tests a single Gamma value to check convergence.
"""

from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_algo import CCGAlgorithm

def test_full_instance(gamma_value=0):
    """Test full instance with single Gamma value."""
    print("="*80)
    print("TESTING FULL-SIZE MODEL")
    print("="*80)
    print(f"Gamma = {gamma_value}")
    print()

    # Load data
    config = ProblemConfig(instance_type='full')
    config.Gamma = gamma_value

    print(f"Loading full-size data...")
    data = SupplyChainData.load(config.data_file)
    print(f"Data loaded: K={data.K}, I={data.I}, J={data.J}, R={data.R}, M={data.M}")
    print()

    # Run C&CG
    print("Running C&CG algorithm...")
    ccg = CCGAlgorithm(data, config)
    result = ccg.run()

    # Display results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Total Time: {result['total_time']:.2f}s")
    print(f"Final Gap: {result['gap']:.6f}")
    print(f"Optimal Value: {result['optimal_value']:.2f}")
    print(f"Lower Bound: {result['LB']:.2f}")
    print(f"Upper Bound: {result['UB']:.2f}")
    print(f"Critical Scenarios: {result['num_scenarios']}")
    print("="*80)

    # Verify convergence
    if result['converged']:
        print("\n✅ FULL-SIZE MODEL OPERATES CORRECTLY")
        return True
    else:
        print("\n❌ FULL-SIZE MODEL DID NOT CONVERGE")
        print(f"   Final gap: {result['gap']:.6f}")
        return False

if __name__ == "__main__":
    import sys
    gamma = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success = test_full_instance(gamma)
    sys.exit(0 if success else 1)
