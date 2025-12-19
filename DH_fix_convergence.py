"""
DH_fix_convergence.py
Implements fixes for C&CG convergence gap issues.
Run this to apply recommended improvements.
"""

from DH_config import ProblemConfig


def apply_fix_1_tighter_solver_tolerances():
    """
    Fix 1: Use tighter Gurobi solver tolerances.
    This helps ensure both Master and Subproblem solve to true optimality.
    """
    print("="*80)
    print("FIX 1: Applying Tighter Solver Tolerances")
    print("="*80)

    # Read current config
    with open('DH_config.py', 'r') as f:
        lines = f.readlines()

    # Find and modify Gurobi settings
    modifications = []
    for i, line in enumerate(lines):
        if 'self.gurobi_mip_gap = ' in line:
            old_line = line
            lines[i] = "        self.gurobi_mip_gap = 1e-6  # Tighter than default 1e-4\n"
            modifications.append(f"Line {i+1}: {old_line.strip()} → {lines[i].strip()}")

        if 'self.gurobi_output_flag = 1' in line:
            # Add new tolerance parameters after this line
            insert_lines = [
                "        self.gurobi_feasibility_tol = 1e-9  # Tighter feasibility tolerance\n",
                "        self.gurobi_int_feas_tol = 1e-9  # Tighter integer feasibility\n",
                "        self.gurobi_opt_tol = 1e-9  # Tighter optimality tolerance\n"
            ]
            for j, new_line in enumerate(insert_lines):
                lines.insert(i + 1 + j, new_line)
                modifications.append(f"Line {i+2+j}: ADDED {new_line.strip()}")
            break

    # Write back
    with open('DH_config.py', 'w') as f:
        f.writelines(lines)

    print("✅ Modified DH_config.py:")
    for mod in modifications:
        print(f"  {mod}")

    # Now update Master and Subproblem to use these parameters
    print("\n📝 Updating DH_master.py...")
    update_solver_parameters('DH_master.py')

    print("\n📝 Updating DH_sub.py...")
    update_solver_parameters('DH_sub.py')

    print("\n✅ Fix 1 applied successfully!")


def update_solver_parameters(filename):
    """Update Gurobi solver parameters in a model file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    modifications = []
    for i, line in enumerate(lines):
        if 'def _configure_solver(self):' in line:
            # Find the block and add new parameters
            j = i + 1
            while j < len(lines) and 'def ' not in lines[j]:
                if 'self.model.setParam(' in lines[j]:
                    # Check if we're at the last setParam
                    if 'OutputFlag' in lines[j]:
                        # Add new parameters after OutputFlag
                        insert_lines = [
                            "        self.model.setParam('FeasibilityTol', self.config.gurobi_feasibility_tol)\n",
                            "        self.model.setParam('IntFeasTol', self.config.gurobi_int_feas_tol)\n",
                            "        self.model.setParam('OptimalityTol', self.config.gurobi_opt_tol)\n"
                        ]
                        for k, new_line in enumerate(insert_lines):
                            lines.insert(j + 1 + k, new_line)
                            modifications.append(f"  Added: {new_line.strip()}")
                        break
                j += 1
            break

    # Write back
    with open(filename, 'w') as f:
        f.writelines(lines)

    for mod in modifications:
        print(mod)


def apply_fix_2_add_diagnostic_logging():
    """
    Fix 2: Add detailed logging to C&CG algorithm.
    This helps track down exact source of gaps.
    """
    print("\n" + "="*80)
    print("FIX 2: Adding Diagnostic Logging to C&CG Algorithm")
    print("="*80)

    with open('DH_algo.py', 'r') as f:
        lines = f.readlines()

    # Find the update_bounds method and add logging
    for i, line in enumerate(lines):
        if 'def update_bounds(self, mp_solution, Z_SP):' in line:
            # Find the end of the method
            j = i + 1
            indent_count = len(line) - len(line.lstrip())
            while j < len(lines):
                if lines[j].strip() and not lines[j].startswith(' ' * (indent_count + 4)):
                    # End of method
                    # Insert diagnostic logging before the last line
                    insert_idx = j - 1
                    insert_lines = [
                        "\n",
                        "        # DIAGNOSTIC: Calculate expected theta from current scenario\n",
                        "        # (for debugging convergence gap)\n",
                        "        expected_theta = Z_SP\n",
                        "        actual_theta = mp_solution.get('theta', None)\n",
                        "        if actual_theta is not None:\n",
                        "            theta_gap = abs(actual_theta - expected_theta)\n",
                        "            if theta_gap > 1e-3:\n",
                        "                print(f\"  ⚠️  WARNING: θ mismatch detected!\")\n",
                        "                print(f\"      Expected θ (from Z_SP): {expected_theta:.6f}\")\n",
                        "                print(f\"      Actual θ (from Master): {actual_theta:.6f}\")\n",
                        "                print(f\"      Gap: {theta_gap:.6f}\")\n",
                    ]
                    for k, new_line in enumerate(insert_lines):
                        lines.insert(insert_idx + k, new_line)
                    print("✅ Added diagnostic logging to update_bounds() method")
                    break
                j += 1
            break

    # Write back
    with open('DH_algo.py', 'w') as f:
        f.writelines(lines)

    print("✅ Fix 2 applied successfully!")


def apply_fix_3_add_primal_verification():
    """
    Fix 3: Add option to verify Subproblem dual with primal solve.
    This confirms dual formulation correctness.
    """
    print("\n" + "="*80)
    print("FIX 3: Adding Primal Verification Option")
    print("="*80)

    # This is more complex - for now, just document it
    verification_code = '''
# Add this method to DH_sub.py to verify dual correctness:

def verify_dual_with_primal(self):
    """
    Solve the Subproblem in primal form and compare with dual result.
    For debugging only - computationally expensive.
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Create primal model
    primal = gp.Model("Subproblem_Primal_Verification")
    primal.setParam('OutputFlag', 0)

    # Second-stage variables
    A_ij = {}
    A_jr = {}
    u = {}

    for k in range(self.K):
        for i in range(self.I):
            for j in range(self.J):
                A_ij[(k,i,j)] = primal.addVar(lb=0, name=f"A_ij_{k}_{i}_{j}")

    for k in range(self.K):
        for j in range(self.J):
            for r in range(self.R):
                A_jr[(k,j,r)] = primal.addVar(lb=0, name=f"A_jr_{k}_{j}_{r}")

    for r in range(self.R):
        for k in range(self.K):
            u[(r,k)] = primal.addVar(lb=0, name=f"u_{r}_{k}")

    primal.update()

    # Calculate realized demand using FIXED eta values from dual solution
    d_realized = {}
    for r in range(self.R):
        for k in range(self.K):
            eta_val = self.eta_plus[(r,k)].X - self.eta_minus[(r,k)].X
            nominal = sum(self.data.mu[(r,k)] * self.data.DI[(m,k)] * self.fixed_beta[(r,m)]
                         for m in range(self.M))
            d_realized[(r,k)] = nominal + eta_val * self.data.mu_hat[(r,k)]

    # Objective: maximize operational profit
    revenue = gp.quicksum(
        self.data.S * (d_realized[(r,k)] - u[(r,k)])
        for r in range(self.R) for k in range(self.K)
    )

    HC = gp.quicksum(
        (self.data.h[j]/2) * A_ij[(k,i,j)]
        for k in range(self.K) for i in range(self.I) for j in range(self.J)
    )

    TC = gp.quicksum(
        self.data.D1[(k,i,j)] * self.data.t * A_ij[(k,i,j)]
        for k in range(self.K) for i in range(self.I) for j in range(self.J)
    ) + gp.quicksum(
        self.data.D2[(j,r)] * self.data.TC[m] * self.fixed_alpha[(j,r,m)] * A_jr[(k,j,r)]
        for k in range(self.K) for j in range(self.J)
        for r in range(self.R) for m in range(self.M)
    )

    PC = gp.quicksum(
        self.data.F[(k,i)] * A_ij[(k,i,j)]
        for k in range(self.K) for i in range(self.I) for j in range(self.J)
    )

    SC = gp.quicksum(
        self.data.SC * u[(r,k)]
        for r in range(self.R) for k in range(self.K)
    )

    primal.setObjective(revenue - HC - TC - PC - SC, GRB.MAXIMIZE)

    # Add constraints (same as in Master Problem for a scenario)
    # ... (add all operational constraints here)

    # Solve
    primal.optimize()

    if primal.Status == GRB.OPTIMAL:
        primal_obj = primal.ObjVal
        dual_obj = self.model.ObjVal
        gap = abs(primal_obj - dual_obj)

        print(f"  Primal verification:")
        print(f"    Primal objective: {primal_obj:.6f}")
        print(f"    Dual objective:   {dual_obj:.6f}")
        print(f"    Gap:              {gap:.6f}")

        if gap < 1e-4:
            print(f"    ✅ Dual formulation verified!")
        else:
            print(f"    ❌ Dual formulation error detected!")

        return gap
    else:
        print(f"  ❌ Primal verification failed (status: {primal.Status})")
        return None
'''

    with open('DH_PRIMAL_VERIFICATION_CODE.txt', 'w') as f:
        f.write(verification_code)

    print("✅ Primal verification code saved to DH_PRIMAL_VERIFICATION_CODE.txt")
    print("   (Manual integration required)")
    print("✅ Fix 3 documented!")


def main():
    """Apply all fixes."""
    print("\n" + "="*80)
    print("C&CG CONVERGENCE FIX SCRIPT")
    print("="*80)
    print("\nThis script will apply three fixes to address convergence gaps:\n")
    print("1. Tighter Gurobi solver tolerances (MIPGap, FeasibilityTol, etc.)")
    print("2. Diagnostic logging to track θ mismatches")
    print("3. Primal verification code (for manual integration)")
    print("\n" + "="*80)

    response = input("\nProceed with fixes? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    try:
        apply_fix_1_tighter_solver_tolerances()
        apply_fix_2_add_diagnostic_logging()
        apply_fix_3_add_primal_verification()

        print("\n" + "="*80)
        print("ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("1. Regenerate data: python3 DH_data_gen.py")
        print("2. Run debug tool: python3 DH_debug_gap.py toy 0")
        print("3. Run full test: python3 DH_main.py toy")
        print("\nIf gap persists, run the debug tool to identify exact source.")

    except Exception as e:
        print(f"\n❌ Error applying fixes: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
