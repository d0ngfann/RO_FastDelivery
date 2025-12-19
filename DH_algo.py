"""
DH_algo.py
Column-and-Constraint Generation (C&CG) algorithm implementation.
Iteratively solves Master Problem and Subproblem until convergence.
"""

import time
import numpy as np
from DH_data_gen import SupplyChainData
from DH_config import ProblemConfig
from DH_master import MasterProblem
from DH_sub import Subproblem


class CCGAlgorithm:
    """Column-and-Constraint Generation algorithm for robust optimization."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize C&CG algorithm.

        Args:
            data: SupplyChainData instance
            config: ProblemConfig instance (must have Gamma set)
        """
        self.data = data
        self.config = config

        if config.Gamma is None:
            raise ValueError("Gamma must be set in config before initializing C&CG")

        # Algorithm state
        self.LB = -float('inf')  # Lower bound
        self.UB = float('inf')  # Upper bound
        self.iteration = 0
        self.converged = False
        self.optimal_solution = None
        self.optimal_value = None
        self.critical_scenarios = []

        # History tracking
        self.history = {
            'iterations': [],
            'LB': [],
            'UB': [],
            'gap': [],
            'time': [],
            'mp_time': [],
            'sp_time': []
        }

        # Master and Subproblem instances
        self.master = None
        self.subproblem = None

        # Start time
        self.start_time = None

    def initialize(self):
        """Initialize algorithm with nominal scenario."""
        print("=" * 80)
        print("INITIALIZING C&CG ALGORITHM")
        print("=" * 80)
        print(f"Problem: K={self.data.K}, I={self.data.I}, J={self.data.J}, "
              f"R={self.data.R}, M={self.data.M}")
        print(f"Uncertainty Budget: Γ = {self.config.Gamma}")
        print(f"Convergence Tolerance: ε = {self.config.epsilon}")
        print("=" * 80)

        # Create Master Problem
        print("Creating Master Problem...")
        self.master = MasterProblem(self.data, self.config)

        # Add nominal scenario (η = 0)
        print("Adding nominal scenario (η = 0)...")
        eta_plus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        eta_minus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        self.master.add_scenario(scenario_id=0, eta_plus=eta_plus_nominal, eta_minus=eta_minus_nominal)
        self.critical_scenarios.append((0, eta_plus_nominal, eta_minus_nominal))

        # Create Subproblem
        print("Creating Subproblem...")
        self.subproblem = Subproblem(self.data, self.config)

        print("Initialization complete!\n")

    def solve_master(self):
        """
        Solve Master Problem.

        Returns:
            tuple: (success, solution, theta, solve_time)
        """
        print(f"[Iteration {self.iteration}] Solving Master Problem...")
        start_time = time.time()

        success = self.master.solve()
        solve_time = time.time() - start_time

        if not success:
            print(f"  Master Problem failed to solve! Status: {self.master.model.Status}")
            return False, None, None, solve_time

        solution = self.master.get_solution()
        theta = solution['theta']

        # Calculate strategic costs
        OC = sum(self.data.O[j] * solution['y'][j] for j in range(self.data.J))

        FC = (sum(self.data.C_plant[(k, i)] * solution['x'][i]
                  for k in range(self.data.K) for i in range(self.data.I)) +
              sum(self.data.C_dc[j] * solution['y'][j] for j in range(self.data.J)) +
              sum(self.data.L1[(k, i, j)] * solution['z'][(i, j)]
                  for k in range(self.data.K) for i in range(self.data.I) for j in range(self.data.J)) +
              sum(self.data.L2[(j, r)] * solution['w'][(j, r)]
                  for j in range(self.data.J) for r in range(self.data.R)))

        # Update upper bound: UB = -OC - FC + theta
        self.UB = -OC - FC + theta

        print(f"  Master solved in {solve_time:.2f}s")
        print(f"  Objective = {solution['objective']:.2f}, θ = {theta:.2f}")
        print(f"  Upper Bound (UB) = {self.UB:.2f}")

        return True, solution, theta, solve_time

    def solve_subproblem(self, mp_solution):
        """
        Solve Subproblem with fixed first-stage solution.

        Args:
            mp_solution: Solution dict from Master Problem

        Returns:
            tuple: (success, Z_SP, eta_plus, eta_minus, solve_time)
        """
        print(f"[Iteration {self.iteration}] Solving Subproblem...")
        start_time = time.time()

        # Fix first-stage variables
        self.subproblem.fix_first_stage(mp_solution)

        success = self.subproblem.solve()
        solve_time = time.time() - start_time

        if not success:
            print(f"  Subproblem failed to solve! Status: {self.subproblem.model.Status}")
            return False, None, None, None, solve_time

        Z_SP, eta_plus, eta_minus = self.subproblem.get_worst_case_scenario()

        print(f"  Subproblem solved in {solve_time:.2f}s")
        print(f"  Worst-case operational profit (Z_SP) = {Z_SP:.2f}")

        # Count number of deviating customers
        num_plus = sum(1 for v in eta_plus.values() if v > 0.5)
        num_minus = sum(1 for v in eta_minus.values() if v > 0.5)
        print(f"  Scenario: {num_plus} demand increases, {num_minus} demand decreases")

        return True, Z_SP, eta_plus, eta_minus, solve_time

    def update_bounds(self, mp_solution, Z_SP):
        """
        Update lower bound.

        Args:
            mp_solution: Master Problem solution
            Z_SP: Subproblem optimal value (worst-case operational profit)
        """
        # Calculate strategic costs
        OC = sum(self.data.O[j] * mp_solution['y'][j] for j in range(self.data.J))

        FC = (sum(self.data.C_plant[(k, i)] * mp_solution['x'][i]
                  for k in range(self.data.K) for i in range(self.data.I)) +
              sum(self.data.C_dc[j] * mp_solution['y'][j] for j in range(self.data.J)) +
              sum(self.data.L1[(k, i, j)] * mp_solution['z'][(i, j)]
                  for k in range(self.data.K) for i in range(self.data.I) for j in range(self.data.J)) +
              sum(self.data.L2[(j, r)] * mp_solution['w'][(j, r)]
                  for j in range(self.data.J) for r in range(self.data.R)))

        # True robust profit: -OC - FC + Z_SP
        Z_current = -OC - FC + Z_SP

        # Update lower bound
        self.LB = max(self.LB, Z_current)

        print(f"  True Robust Profit = {Z_current:.2f}")
        print(f"  Lower Bound (LB) = {self.LB:.2f}")

    def check_convergence(self):
        """Check convergence criterion."""
        gap = self.UB - self.LB
        rel_gap = gap / (abs(self.UB) + 1e-10)

        print(f"  Gap = {gap:.4f}, Relative Gap = {rel_gap:.6f}")

        if gap <= self.config.epsilon:
            self.converged = True
            print("  *** CONVERGED ***")
            return True

        return False

    def is_duplicate_scenario(self, eta_plus, eta_minus):
        """
        Check if scenario already exists in critical scenarios.

        Args:
            eta_plus: dict {(r,k): value}
            eta_minus: dict {(r,k): value}

        Returns:
            bool: True if scenario is duplicate
        """
        for _, existing_eta_plus, existing_eta_minus in self.critical_scenarios:
            # Check if all eta values match
            is_same = True
            for r in range(self.data.R):
                for k in range(self.data.K):
                    if (eta_plus[(r, k)] != existing_eta_plus[(r, k)] or
                        eta_minus[(r, k)] != existing_eta_minus[(r, k)]):
                        is_same = False
                        break
                if not is_same:
                    break

            if is_same:
                return True

        return False

    def add_scenario_to_master(self, eta_plus, eta_minus):
        """
        Add new scenario to Master Problem.

        Args:
            eta_plus: dict {(r,k): value}
            eta_minus: dict {(r,k): value}

        Returns:
            bool: True if scenario was added, False if duplicate
        """
        # Check for duplicates
        if self.is_duplicate_scenario(eta_plus, eta_minus):
            print(f"  Scenario is duplicate - not adding to Master Problem")
            return False

        scenario_id = len(self.critical_scenarios)
        print(f"  Adding scenario {scenario_id} to Master Problem...")

        self.master.add_scenario(scenario_id, eta_plus, eta_minus)
        self.critical_scenarios.append((scenario_id, eta_plus, eta_minus))
        return True

    def log_iteration(self, mp_time, sp_time):
        """Log iteration statistics."""
        elapsed = time.time() - self.start_time
        gap = self.UB - self.LB

        self.history['iterations'].append(self.iteration)
        self.history['LB'].append(self.LB)
        self.history['UB'].append(self.UB)
        self.history['gap'].append(gap)
        self.history['time'].append(elapsed)
        self.history['mp_time'].append(mp_time)
        self.history['sp_time'].append(sp_time)

    def run(self):
        """
        Execute the C&CG algorithm.

        Returns:
            dict: Results including optimal solution, value, and statistics
        """
        self.start_time = time.time()
        self.initialize()

        print("\n" + "=" * 80)
        print("STARTING C&CG ITERATIONS")
        print("=" * 80 + "\n")

        while not self.converged and self.iteration < self.config.max_iterations:
            self.iteration += 1
            print(f"\n{'='*80}")
            print(f"ITERATION {self.iteration}")
            print(f"{'='*80}")

            # Step 1: Solve Master Problem
            success, mp_solution, theta, mp_time = self.solve_master()
            if not success:
                print("ERROR: Master Problem failed. Terminating.")
                break

            # Step 2: Solve Subproblem
            success, Z_SP, eta_plus, eta_minus, sp_time = self.solve_subproblem(mp_solution)
            if not success:
                print("ERROR: Subproblem failed. Terminating.")
                break

            # Step 3: Update bounds
            self.update_bounds(mp_solution, Z_SP)

            # Log iteration
            self.log_iteration(mp_time, sp_time)

            # Step 4: Check convergence
            if self.check_convergence():
                self.optimal_solution = mp_solution
                self.optimal_value = self.LB
                break

            # Step 5: Add scenario to Master
            scenario_added = self.add_scenario_to_master(eta_plus, eta_minus)

            # If scenario is duplicate, algorithm has stalled
            if not scenario_added:
                print("\n  WARNING: Duplicate scenario detected - algorithm stalled!")
                print(f"  Current gap: {self.UB - self.LB:.6f}")
                print("  This may indicate numerical issues or model formulation error.")
                print("  Terminating with current best solution.\n")
                self.optimal_solution = mp_solution
                self.optimal_value = self.LB
                break

            # Recreate subproblem for next iteration (to avoid stale constraints)
            self.subproblem = Subproblem(self.data, self.config)

        # Final results
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("C&CG ALGORITHM COMPLETED")
        print("=" * 80)
        print(f"Total Iterations: {self.iteration}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Converged: {self.converged}")
        print(f"Final Lower Bound: {self.LB:.4f}")
        print(f"Final Upper Bound: {self.UB:.4f}")
        print(f"Final Gap: {self.UB - self.LB:.6f}")
        print(f"Number of Critical Scenarios: {len(self.critical_scenarios)}")
        print("=" * 80)

        results = {
            'converged': self.converged,
            'iterations': self.iteration,
            'total_time': total_time,
            'optimal_value': self.optimal_value,
            'optimal_solution': self.optimal_solution,
            'LB': self.LB,
            'UB': self.UB,
            'gap': self.UB - self.LB,
            'num_scenarios': len(self.critical_scenarios),
            'critical_scenarios': self.critical_scenarios,
            'history': self.history,
            'Gamma': self.config.Gamma
        }

        return results


def print_solution_summary(solution, data):
    """Print a summary of the optimal solution."""
    if solution is None:
        print("No solution available.")
        return

    print("\n" + "=" * 80)
    print("OPTIMAL SOLUTION SUMMARY")
    print("=" * 80)

    # Plants opened
    opened_plants = [i for i in range(data.I) if solution['x'][i] > 0.5]
    print(f"Plants Opened: {opened_plants} ({len(opened_plants)}/{data.I})")

    # DCs opened
    opened_dcs = [j for j in range(data.J) if solution['y'][j] > 0.5]
    print(f"DCs Opened: {opened_dcs} ({len(opened_dcs)}/{data.J})")

    # Routes plant-to-DC
    active_routes_ij = [(i, j) for (i, j) in solution['z'] if solution['z'][(i, j)] > 0.5]
    print(f"Active Routes (Plant→DC): {len(active_routes_ij)}")

    # Routes DC-to-customer
    active_routes_jr = [(j, r) for (j, r) in solution['w'] if solution['w'][(j, r)] > 0.5]
    print(f"Active Routes (DC→Customer): {len(active_routes_jr)}")

    # Mode distribution
    mode_counts = [0] * data.M
    for r in range(data.R):
        for m in range(data.M):
            if solution['beta'][(r, m)] > 0.5:
                mode_counts[m] += 1

    print(f"Transportation Modes: ", end="")
    for m in range(data.M):
        print(f"Mode {m}: {mode_counts[m]} customers", end="  ")
    print()

    print("=" * 80)
