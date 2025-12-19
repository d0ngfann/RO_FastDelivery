"""
DH_debug_gap.py
Comprehensive debugging tool for C&CG convergence gap issues.
Identifies exact source of Master-Subproblem mismatch.
"""

import numpy as np
from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_master import MasterProblem
from DH_sub import Subproblem


def debug_scenario_evaluation(data, config, scenario_id, eta_plus, eta_minus, first_stage_solution):
    """
    Evaluate a single scenario in both Master and Subproblem to identify discrepancies.

    Args:
        data: SupplyChainData instance
        config: ProblemConfig instance
        scenario_id: Scenario ID
        eta_plus: dict {(r,k): value}
        eta_minus: dict {(r,k): value}
        first_stage_solution: Solution dict with keys x, y, z, w, beta, alpha

    Returns:
        dict with detailed breakdown of both calculations
    """
    print("\n" + "="*80)
    print(f"DEBUGGING SCENARIO {scenario_id}")
    print("="*80)

    # Extract first-stage solution
    x_sol = first_stage_solution['x']
    y_sol = first_stage_solution['y']
    z_sol = first_stage_solution['z']
    w_sol = first_stage_solution['w']
    beta_sol = first_stage_solution['beta']
    alpha_sol = first_stage_solution['alpha']

    # Calculate strategic costs (same for both)
    OC = sum(data.O[j] * y_sol[j] for j in range(data.J))

    FC = (sum(data.C_plant[(k, i)] * x_sol[i]
              for k in range(data.K) for i in range(data.I)) +
          sum(data.C_dc[j] * y_sol[j] for j in range(data.J)) +
          sum(data.L1[(k, i, j)] * z_sol[(i, j)]
              for k in range(data.K) for i in range(data.I) for j in range(data.J)) +
          sum(data.L2[(j, r)] * w_sol[(j, r)]
              for j in range(data.J) for r in range(data.R)))

    print(f"\nStrategic Costs:")
    print(f"  OC = {OC:.2f}")
    print(f"  FC = {FC:.2f}")
    print(f"  Total Strategic = {-OC - FC:.2f}")

    # Calculate realized demand for this scenario
    d_realized = {}
    for r in range(data.R):
        for k in range(data.K):
            # Endogenous nominal demand using FIXED beta values
            nominal = sum(data.mu[(r, k)] * data.DI[(m, k)] * beta_sol[(r, m)]
                         for m in range(data.M))
            # Uncertainty deviation
            uncertainty = (eta_plus[(r, k)] - eta_minus[(r, k)]) * data.mu_hat[(r, k)]
            d_realized[(r, k)] = nominal + uncertainty

    print(f"\nRealized Demand Sample (first 3 customers, product 0):")
    for r in range(min(3, data.R)):
        k = 0
        print(f"  d[{r},{k}] = {d_realized[(r, k)]:.2f}")

    # ========== METHOD 1: Solve as Master Problem constraint ==========
    print("\n" + "-"*80)
    print("METHOD 1: Master Problem Optimality Cut Evaluation")
    print("-"*80)

    # Create a temporary Master Problem to evaluate
    master_temp = MasterProblem(data, config)

    # Fix first-stage variables by setting bounds
    for i in range(data.I):
        master_temp.x[i].LB = master_temp.x[i].UB = x_sol[i]
    for j in range(data.J):
        master_temp.y[j].LB = master_temp.y[j].UB = y_sol[j]
    for i in range(data.I):
        for j in range(data.J):
            master_temp.z[(i, j)].LB = master_temp.z[(i, j)].UB = z_sol[(i, j)]
    for j in range(data.J):
        for r in range(data.R):
            master_temp.w[(j, r)].LB = master_temp.w[(j, r)].UB = w_sol[(j, r)]
    for r in range(data.R):
        for m in range(data.M):
            master_temp.beta[(r, m)].LB = master_temp.beta[(r, m)].UB = beta_sol[(r, m)]
    for j in range(data.J):
        for r in range(data.R):
            for m in range(data.M):
                master_temp.alpha[(j, r, m)].LB = master_temp.alpha[(j, r, m)].UB = alpha_sol[(j, r, m)]

    # Add the scenario
    master_temp.add_scenario(scenario_id, eta_plus, eta_minus)

    # Solve with theta unbounded above
    master_temp.theta.UB = 1e10
    master_temp.solve()

    if master_temp.model.Status == 2:  # OPTIMAL
        # Get second-stage solution
        A_ij_master = {}
        A_jr_master = {}
        u_master = {}
        X_master = {}

        for k in range(data.K):
            for i in range(data.I):
                for j in range(data.J):
                    A_ij_master[(k, i, j)] = master_temp.A_ij[(k, i, j, scenario_id)].X

        for k in range(data.K):
            for j in range(data.J):
                for r in range(data.R):
                    A_jr_master[(k, j, r)] = master_temp.A_jr[(k, j, r, scenario_id)].X

        for r in range(data.R):
            for k in range(data.K):
                u_master[(r, k)] = master_temp.u[(r, k, scenario_id)].X

        for j in range(data.J):
            for r in range(data.R):
                for m in range(data.M):
                    for k in range(data.K):
                        X_master[(j, r, m, k)] = master_temp.X[(j, r, m, k, scenario_id)].X

        # Calculate operational profit components
        # Revenue
        revenue_master = sum(data.S * (d_realized[(r, k)] - u_master[(r, k)])
                            for r in range(data.R) for k in range(data.K))

        # Holding cost
        HC_master = sum((data.h[j] / 2) * A_ij_master[(k, i, j)]
                       for k in range(data.K) for i in range(data.I) for j in range(data.J))

        # Transportation cost
        TC1_master = sum(data.D1[(k, i, j)] * data.t * A_ij_master[(k, i, j)]
                        for k in range(data.K) for i in range(data.I) for j in range(data.J))

        TC2_master = sum(data.D2[(j, r)] * data.TC[m] * X_master[(j, r, m, k)]
                        for k in range(data.K) for j in range(data.J)
                        for r in range(data.R) for m in range(data.M))

        TC_master = TC1_master + TC2_master

        # Production cost
        PC_master = sum(data.F[(k, i)] * A_ij_master[(k, i, j)]
                       for k in range(data.K) for i in range(data.I) for j in range(data.J))

        # Shortage cost
        SC_master = sum(data.SC * u_master[(r, k)]
                       for r in range(data.R) for k in range(data.K))

        # Total operational profit
        op_profit_master = revenue_master - HC_master - TC_master - PC_master - SC_master

        # Total robust profit
        total_profit_master = -OC - FC + op_profit_master

        print(f"  Revenue:        {revenue_master:12.2f}")
        print(f"  Holding Cost:   {HC_master:12.2f}")
        print(f"  Transport Cost: {TC_master:12.2f}")
        print(f"    - TC1 (plant-DC):    {TC1_master:12.2f}")
        print(f"    - TC2 (DC-customer): {TC2_master:12.2f}")
        print(f"  Production Cost:{PC_master:12.2f}")
        print(f"  Shortage Cost:  {SC_master:12.2f}")
        print(f"  ----------------------------------------")
        print(f"  Operational Profit: {op_profit_master:12.2f}")
        print(f"  Total Robust Profit: {total_profit_master:12.2f}")

        # Check some second-stage variables
        total_shortage = sum(u_master.values())
        total_production = sum(A_ij_master.values())
        print(f"\n  Total shortage: {total_shortage:.2f}")
        print(f"  Total production: {total_production:.2f}")

        # Verify linearization: X should equal alpha * A_jr
        linearization_error = 0
        for j in range(data.J):
            for r in range(data.R):
                for m in range(data.M):
                    for k in range(data.K):
                        expected = alpha_sol[(j, r, m)] * A_jr_master[(k, j, r)]
                        actual = X_master[(j, r, m, k)]
                        error = abs(expected - actual)
                        if error > 1e-4:
                            linearization_error += error

        if linearization_error > 1e-3:
            print(f"  ⚠️ LINEARIZATION ERROR: {linearization_error:.6f}")
        else:
            print(f"  ✓ Linearization verified (error < 1e-3)")

    else:
        print(f"  ❌ Master Problem failed to solve! Status: {master_temp.model.Status}")
        op_profit_master = None
        total_profit_master = None

    # ========== METHOD 2: Solve Subproblem (Dual) ==========
    print("\n" + "-"*80)
    print("METHOD 2: Subproblem (Dual) Evaluation")
    print("-"*80)

    # Create Subproblem and fix first-stage
    subproblem = Subproblem(data, config)
    subproblem.fix_first_stage(first_stage_solution)

    # Fix eta variables to the specific scenario
    for r in range(data.R):
        for k in range(data.K):
            subproblem.eta_plus[(r, k)].LB = subproblem.eta_plus[(r, k)].UB = eta_plus[(r, k)]
            subproblem.eta_minus[(r, k)].LB = subproblem.eta_minus[(r, k)].UB = eta_minus[(r, k)]

    # Solve
    success = subproblem.solve()

    if success:
        # FIXED: Use get_worst_case_scenario() which includes revenue term
        Z_SP, _, _ = subproblem.get_worst_case_scenario()
        dual_obj = subproblem.model.ObjVal  # Just for reference
        total_profit_subproblem = -OC - FC + Z_SP

        print(f"  Z_SP (operational profit): {Z_SP:12.2f}")
        print(f"  Dual objective (raw):      {dual_obj:12.2f}")
        print(f"  Total Robust Profit:       {total_profit_subproblem:12.2f}")

        # Get dual variable values for analysis
        gamma_vals = {(r, k): subproblem.gamma[(r, k)].X
                     for r in range(data.R) for k in range(data.K)}

        print(f"\n  Sample gamma values (first 3 customers, product 0):")
        for r in range(min(3, data.R)):
            k = 0
            print(f"    γ[{r},{k}] = {gamma_vals[(r, k)]:8.2f}")

        # Check if gamma bounds are satisfied
        gamma_violations = 0
        for r in range(data.R):
            for k in range(data.K):
                if gamma_vals[(r, k)] < -(data.S + data.SC) - 1e-4:
                    gamma_violations += 1
                if gamma_vals[(r, k)] > data.S + 1e-4:
                    gamma_violations += 1

        if gamma_violations > 0:
            print(f"  ⚠️ GAMMA BOUND VIOLATIONS: {gamma_violations}")
        else:
            print(f"  ✓ Gamma bounds satisfied")

    else:
        print(f"  ❌ Subproblem failed to solve! Status: {subproblem.model.Status}")
        Z_SP = None
        total_profit_subproblem = None

    # ========== COMPARISON ==========
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    if op_profit_master is not None and Z_SP is not None:
        gap_operational = abs(op_profit_master - Z_SP)
        gap_total = abs(total_profit_master - total_profit_subproblem)

        print(f"  Master Operational Profit:     {op_profit_master:12.2f}")
        print(f"  Subproblem Z_SP:               {Z_SP:12.2f}")
        print(f"  Gap (Operational):             {gap_operational:12.6f}")
        print(f"")
        print(f"  Master Total Profit:           {total_profit_master:12.2f}")
        print(f"  Subproblem Total Profit:       {total_profit_subproblem:12.2f}")
        print(f"  Gap (Total):                   {gap_total:12.6f}")

        if gap_operational < 1e-3:
            print(f"\n  ✅ MATCH! Both methods agree within tolerance.")
        else:
            print(f"\n  ❌ MISMATCH! Gap = {gap_operational:.6f}")
            print(f"     This explains the C&CG convergence issue!")

            # Provide diagnostic hints
            if gap_operational > 1000:
                print(f"\n  💡 Large gap suggests:")
                print(f"     - Possible error in cost calculation formulas")
                print(f"     - Check if all cost terms match between Master and Subproblem")
            elif gap_operational > 10:
                print(f"\n  💡 Medium gap suggests:")
                print(f"     - Possible linearization accuracy issue")
                print(f"     - Check Big-M values (currently using MC_j)")
            else:
                print(f"\n  💡 Small gap suggests:")
                print(f"     - Numerical precision / solver tolerance issue")
                print(f"     - Consider tightening Gurobi MIP gap parameter")

        return {
            'operational_profit_master': op_profit_master,
            'operational_profit_subproblem': Z_SP,
            'total_profit_master': total_profit_master,
            'total_profit_subproblem': total_profit_subproblem,
            'gap_operational': gap_operational,
            'gap_total': gap_total
        }

    else:
        print("  ❌ Cannot compare - one or both methods failed")
        return None


def run_debug_session(instance_type='toy', gamma_value=0):
    """
    Run a debug session on a specific instance.

    Args:
        instance_type: 'toy' or 'full'
        gamma_value: Uncertainty budget to test
    """
    print("\n" + "="*80)
    print(f"C&CG GAP DEBUGGING SESSION")
    print(f"Instance: {instance_type}, Γ = {gamma_value}")
    print("="*80)

    # Load configuration and data
    config = ProblemConfig(instance_type=instance_type)
    config.set_gamma(gamma_value)

    if os.path.exists(config.data_file):
        data = SupplyChainData.load(config.data_file)
        print(f"\nLoaded data from {config.data_file}")
    else:
        print(f"\n❌ Data file not found: {config.data_file}")
        print("Please run: python3 DH_data_gen.py")
        return

    # Run one C&CG iteration manually
    from DH_algo import CCGAlgorithm

    ccg = CCGAlgorithm(data, config)
    ccg.initialize()

    # Iteration 1
    print("\n" + "="*80)
    print("ITERATION 1")
    print("="*80)

    # Solve Master
    success, mp_solution, theta, mp_time = ccg.solve_master()
    if not success:
        print("❌ Master Problem failed")
        return

    # Solve Subproblem
    success, Z_SP, eta_plus, eta_minus, solve_time = ccg.solve_subproblem(mp_solution)
    if not success:
        print("❌ Subproblem failed")
        return

    # Debug the scenario
    results = debug_scenario_evaluation(
        data, config,
        scenario_id=1,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        first_stage_solution=mp_solution
    )

    return results


if __name__ == "__main__":
    import os
    import sys

    # Get instance type from command line
    instance_type = sys.argv[1] if len(sys.argv) > 1 else 'toy'
    gamma_value = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    results = run_debug_session(instance_type, gamma_value)

    if results and results['gap_operational'] > 1e-3:
        print("\n" + "="*80)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*80)
        print("1. Check Gurobi solver parameters:")
        print("   - Set MIPGap = 1e-6 (tighter than current 1e-4)")
        print("   - Set FeasibilityTol = 1e-9")
        print("   - Set IntFeasTol = 1e-9")
        print("")
        print("2. Verify Big-M values are tight:")
        print("   - Currently using M_j = MC_j (DC capacity)")
        print("   - Check if smaller valid bounds exist")
        print("")
        print("3. Check for numerical scaling issues:")
        print("   - Normalize cost parameters to similar magnitudes")
        print("   - Scale demand values if needed")
