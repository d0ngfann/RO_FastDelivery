"""
Check the optimal solution details from the full-size model.
"""

from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_algo import CCGAlgorithm

def check_solution(gamma_value=0):
    """Check solution details."""
    # Load data
    config = ProblemConfig(instance_type='full')
    config.Gamma = gamma_value

    print("Loading full-size data...")
    data = SupplyChainData.load(config.data_file)

    # Run C&CG (silently)
    config.gurobi_output_flag = 0  # Silent mode
    ccg = CCGAlgorithm(data, config)
    result = ccg.run()

    # Get solution
    solution = ccg.master.get_solution()

    print("="*80)
    print("OPTIMAL SOLUTION DETAILS")
    print("="*80)
    print(f"Objective Value: {result['optimal_value']:,.2f}")
    print()

    # Count opened facilities
    plants_opened = [i for i in range(data.I) if solution['x'][i] > 0.5]
    dcs_opened = [j for j in range(data.J) if solution['y'][j] > 0.5]

    print(f"Plants Opened: {len(plants_opened)}/{data.I}")
    print(f"  Indices: {plants_opened}")
    print()

    print(f"DCs Opened: {len(dcs_opened)}/{data.J}")
    print(f"  Indices: {dcs_opened}")
    print()

    # Count routes
    plant_dc_routes = sum(1 for (i,j), val in solution['z'].items() if val > 0.5)
    dc_customer_routes = sum(1 for (j,r), val in solution['w'].items() if val > 0.5)

    print(f"Active Routes (Plant→DC): {plant_dc_routes}")
    print(f"Active Routes (DC→Customer): {dc_customer_routes}")
    print()

    # Transportation modes
    mode_counts = [0] * data.M
    for (r, m), val in solution['beta'].items():
        if val > 0.5:
            mode_counts[m] += 1

    print("Transportation Mode Assignment:")
    for m in range(data.M):
        print(f"  Mode {m}: {mode_counts[m]} customers")

    print("="*80)

if __name__ == "__main__":
    import sys
    gamma = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    check_solution(gamma)
