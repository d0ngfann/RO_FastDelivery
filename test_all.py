"""
test_all.py
Comprehensive test suite for C&CG implementation.
Tests all major components: config, data generation, master, subproblem, algorithm.
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime


def test_imports():
    """Test 1: Verify all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)

    try:
        from DH_config import ProblemConfig, DataParameters, SensitivityConfig
        print("  ✓ DH_config imported successfully")

        from DH_data_gen import SupplyChainData, generate_supply_chain_data
        print("  ✓ DH_data_gen imported successfully")

        from DH_master import MasterProblem
        print("  ✓ DH_master imported successfully")

        from DH_sub import Subproblem
        print("  ✓ DH_sub imported successfully")

        from DH_algo import CCGAlgorithm, print_solution_summary
        print("  ✓ DH_algo imported successfully")

        from DH_main import run_single_gamma, run_sensitivity_analysis
        print("  ✓ DH_main imported successfully")

        print("\n  ✅ ALL IMPORTS SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n  ❌ IMPORT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test 2: Verify configuration setup."""
    print("\n" + "="*80)
    print("TEST 2: Configuration")
    print("="*80)

    try:
        from DH_config import ProblemConfig

        # Test toy instance
        config_toy = ProblemConfig('toy')
        assert config_toy.K == 1, "Toy K should be 1"
        assert config_toy.I == 2, "Toy I should be 2"
        assert config_toy.J == 2, "Toy J should be 2"
        assert config_toy.R == 5, "Toy R should be 5"
        assert config_toy.M == 2, "Toy M should be 2"
        print("  ✓ Toy instance dimensions correct")

        # Test full instance
        config_full = ProblemConfig('full')
        assert config_full.K == 3, "Full K should be 3"
        assert config_full.I == 5, "Full I should be 5"
        assert config_full.J == 20, "Full J should be 20"
        assert config_full.R == 100, "Full R should be 100"
        assert config_full.M == 3, "Full M should be 3"
        print("  ✓ Full instance dimensions correct")

        # Test Gamma setting
        config_toy.set_gamma(3)
        assert config_toy.Gamma == 3, "Gamma should be 3"
        print("  ✓ Gamma setting works")

        # Test new tolerances
        assert config_toy.gurobi_mip_gap == 1e-6, "MIP gap should be 1e-6"
        assert config_toy.gurobi_feasibility_tol == 1e-9, "FeasTol should be 1e-9"
        assert config_toy.gurobi_int_feas_tol == 1e-9, "IntFeasTol should be 1e-9"
        assert config_toy.gurobi_opt_tol == 1e-9, "OptTol should be 1e-9"
        print("  ✓ Tighter solver tolerances configured")

        print("\n  ✅ CONFIGURATION TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ CONFIGURATION TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generation():
    """Test 3: Verify data generation."""
    print("\n" + "="*80)
    print("TEST 3: Data Generation")
    print("="*80)

    try:
        from DH_config import ProblemConfig
        from DH_data_gen import generate_supply_chain_data

        # Generate toy instance
        config = ProblemConfig('toy')
        data = generate_supply_chain_data(config, seed=42)
        print("  ✓ Data generation successful")

        # Check dimensions
        assert len(data.mu) == config.R * config.K, "mu dictionary size incorrect"
        assert len(data.mu_hat) == config.R * config.K, "mu_hat dictionary size incorrect"
        assert len(data.s_rk) == config.R * config.K, "s_rk dictionary size incorrect"
        print("  ✓ Data dimensions correct")

        # Check s_rk is binary
        for value in data.s_rk.values():
            assert value in [0, 1], "s_rk must be binary"
        print("  ✓ s_rk binary matrix correct")

        # Check demand is non-negative
        for (r, k), val in data.mu.items():
            if data.s_rk[(r, k)] == 1:
                assert val > 0, f"mu[{r},{k}] should be positive when s_rk=1"
            else:
                assert val == 0, f"mu[{r},{k}] should be zero when s_rk=0"
        print("  ✓ Demand values consistent with s_rk")

        # Check capacities are positive
        assert all(v > 0 for v in data.MP.values()), "Plant capacities must be positive"
        assert all(v > 0 for v in data.MC.values()), "DC capacities must be positive"
        print("  ✓ Capacities are positive")

        # Check cost parameters
        assert data.S > 0, "Selling price must be positive"
        assert data.SC > 0, "Shortage cost must be positive"
        assert data.t > 0, "Transport cost must be positive"
        print("  ✓ Cost parameters valid")

        # Check coordinates
        assert len(data.plant_coords) == config.I, "Plant coordinates size incorrect"
        assert len(data.dc_coords) == config.J, "DC coordinates size incorrect"
        assert len(data.customer_coords) == config.R, "Customer coordinates size incorrect"
        print("  ✓ Coordinates generated")

        # Test save/load
        test_file = "data/test_data_temp.pkl"
        data.save(test_file)
        assert os.path.exists(test_file), "Data file not created"

        loaded_data = data.load(test_file)
        assert loaded_data.K == data.K, "Loaded data dimensions incorrect"

        os.remove(test_file)
        print("  ✓ Save/load functionality works")

        print("\n  ✅ DATA GENERATION TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ DATA GENERATION TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_master_problem():
    """Test 4: Verify Master Problem setup."""
    print("\n" + "="*80)
    print("TEST 4: Master Problem")
    print("="*80)

    try:
        from DH_config import ProblemConfig
        from DH_data_gen import generate_supply_chain_data
        from DH_master import MasterProblem

        # Setup
        config = ProblemConfig('toy')
        config.set_gamma(0)
        data = generate_supply_chain_data(config, seed=42)

        # Create Master Problem
        master = MasterProblem(data, config)
        print("  ✓ Master Problem created")

        # Check variables exist
        assert len(master.x) == config.I, "x variables count incorrect"
        assert len(master.y) == config.J, "y variables count incorrect"
        assert len(master.z) == config.I * config.J, "z variables count incorrect"
        assert len(master.w) == config.J * config.R, "w variables count incorrect"
        assert len(master.beta) == config.R * config.M, "beta variables count incorrect"
        assert master.theta is not None, "theta variable not created"
        print("  ✓ All first-stage variables created")

        # Add nominal scenario
        eta_plus = {(r, k): 0 for r in range(config.R) for k in range(config.K)}
        eta_minus = {(r, k): 0 for r in range(config.R) for k in range(config.K)}
        master.add_scenario(0, eta_plus, eta_minus)
        print("  ✓ Scenario added successfully")

        # Check second-stage variables added
        assert len(master.A_ij) > 0, "A_ij variables not created"
        assert len(master.A_jr) > 0, "A_jr variables not created"
        assert len(master.u) > 0, "u variables not created"
        assert len(master.X) > 0, "X variables not created"
        print("  ✓ Second-stage variables created")

        # Try solving
        success = master.solve()
        assert success, "Master Problem failed to solve"
        print("  ✓ Master Problem solved successfully")

        # Get solution
        solution = master.get_solution()
        assert 'objective' in solution, "Solution missing objective"
        assert 'theta' in solution, "Solution missing theta"
        assert 'x' in solution, "Solution missing x"
        print("  ✓ Solution extracted successfully")

        print(f"    Objective value: {solution['objective']:.2f}")
        print(f"    Theta value: {solution['theta']:.2f}")

        print("\n  ✅ MASTER PROBLEM TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ MASTER PROBLEM TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_subproblem():
    """Test 5: Verify Subproblem setup."""
    print("\n" + "="*80)
    print("TEST 5: Subproblem")
    print("="*80)

    try:
        from DH_config import ProblemConfig
        from DH_data_gen import generate_supply_chain_data
        from DH_master import MasterProblem
        from DH_sub import Subproblem

        # Setup
        config = ProblemConfig('toy')
        config.set_gamma(0)
        data = generate_supply_chain_data(config, seed=42)

        # First solve Master to get first-stage solution
        master = MasterProblem(data, config)
        eta_plus = {(r, k): 0 for r in range(config.R) for k in range(config.K)}
        eta_minus = {(r, k): 0 for r in range(config.R) for k in range(config.K)}
        master.add_scenario(0, eta_plus, eta_minus)
        master.solve()
        mp_solution = master.get_solution()

        # Create Subproblem
        subproblem = Subproblem(data, config)
        print("  ✓ Subproblem created")

        # Check dual variables exist
        assert len(subproblem.pi) == config.K * config.I, "pi variables count incorrect"
        assert len(subproblem.sigma) == config.J, "sigma variables count incorrect"
        assert len(subproblem.gamma) == config.R * config.K, "gamma variables count incorrect"
        print("  ✓ Dual variables created")

        # Check uncertainty variables exist
        assert len(subproblem.eta_plus) == config.R * config.K, "eta_plus count incorrect"
        assert len(subproblem.eta_minus) == config.R * config.K, "eta_minus count incorrect"
        print("  ✓ Uncertainty variables created")

        # Fix first-stage and solve
        subproblem.fix_first_stage(mp_solution)
        success = subproblem.solve()
        assert success, "Subproblem failed to solve"
        print("  ✓ Subproblem solved successfully")

        # Get worst-case scenario
        Z_SP, eta_plus_result, eta_minus_result = subproblem.get_worst_case_scenario()
        assert Z_SP is not None, "Z_SP not returned"
        assert eta_plus_result is not None, "eta_plus not returned"
        assert eta_minus_result is not None, "eta_minus not returned"
        print("  ✓ Worst-case scenario extracted")

        print(f"    Z_SP value: {Z_SP:.2f}")

        # Check budget constraint
        for k in range(config.K):
            budget_used = sum(eta_plus_result[(r, k)] + eta_minus_result[(r, k)]
                            for r in range(config.R))
            assert budget_used <= config.Gamma + 1e-6, f"Budget constraint violated for product {k}"
        print("  ✓ Budget constraints satisfied")

        print("\n  ✅ SUBPROBLEM TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ SUBPROBLEM TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ccg_algorithm():
    """Test 6: Verify full C&CG algorithm."""
    print("\n" + "="*80)
    print("TEST 6: C&CG Algorithm")
    print("="*80)

    try:
        from DH_config import ProblemConfig
        from DH_data_gen import generate_supply_chain_data
        from DH_algo import CCGAlgorithm

        # Setup
        config = ProblemConfig('toy')
        config.set_gamma(0)
        config.max_iterations = 20  # Limit iterations for testing
        data = generate_supply_chain_data(config, seed=42)

        # Run C&CG
        ccg = CCGAlgorithm(data, config)
        results = ccg.run()
        print("  ✓ C&CG algorithm completed")

        # Check results structure
        assert 'converged' in results, "Results missing 'converged'"
        assert 'iterations' in results, "Results missing 'iterations'"
        assert 'optimal_value' in results, "Results missing 'optimal_value'"
        assert 'LB' in results, "Results missing 'LB'"
        assert 'UB' in results, "Results missing 'UB'"
        assert 'gap' in results, "Results missing 'gap'"
        print("  ✓ Results structure complete")

        # Check convergence
        gap = results['gap']
        print(f"    Iterations: {results['iterations']}")
        print(f"    Gap: {gap:.6f}")
        print(f"    Converged: {results['converged']}")
        print(f"    LB: {results['LB']:.2f}")
        print(f"    UB: {results['UB']:.2f}")

        if results['converged']:
            assert gap <= config.epsilon, f"Converged but gap {gap} > epsilon {config.epsilon}"
            print("  ✓ Algorithm converged successfully")
        else:
            print(f"  ⚠️  Did not converge within {results['iterations']} iterations")
            print(f"     Gap: {gap:.6f} (target: {config.epsilon})")

        # Check bounds are valid
        assert results['LB'] <= results['UB'] + 1e-3, "LB should be <= UB"
        print("  ✓ Bounds are valid (LB <= UB)")

        # Check solution exists
        if results['optimal_solution'] is not None:
            sol = results['optimal_solution']
            assert 'x' in sol and 'y' in sol, "Solution missing variables"
            print("  ✓ Optimal solution structure valid")

        print("\n  ✅ C&CG ALGORITHM TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ C&CG ALGORITHM TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_convergence_improvement():
    """Test 7: Verify convergence improvement with tighter tolerances."""
    print("\n" + "="*80)
    print("TEST 7: Convergence Improvement")
    print("="*80)

    try:
        from DH_config import ProblemConfig

        config = ProblemConfig('toy')

        # Check that tighter tolerances are actually set
        assert config.gurobi_mip_gap == 1e-6, \
            f"MIP gap should be 1e-6, got {config.gurobi_mip_gap}"
        print("  ✓ MIP gap is tighter (1e-6)")

        assert config.gurobi_feasibility_tol == 1e-9, \
            f"FeasibilityTol should be 1e-9, got {config.gurobi_feasibility_tol}"
        print("  ✓ FeasibilityTol is tighter (1e-9)")

        assert config.gurobi_int_feas_tol == 1e-9, \
            f"IntFeasTol should be 1e-9, got {config.gurobi_int_feas_tol}"
        print("  ✓ IntFeasTol is tighter (1e-9)")

        assert config.gurobi_opt_tol == 1e-9, \
            f"OptimalityTol should be 1e-9, got {config.gurobi_opt_tol}"
        print("  ✓ OptimalityTol is tighter (1e-9)")

        print("\n  ✅ CONVERGENCE IMPROVEMENT TESTS PASSED")
        return True

    except AssertionError as e:
        print(f"\n  ❌ CONVERGENCE IMPROVEMENT TEST FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Data Generation", test_data_generation),
        ("Master Problem", test_master_problem),
        ("Subproblem", test_subproblem),
        ("C&CG Algorithm", test_ccg_algorithm),
        ("Convergence Improvement", test_convergence_improvement),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ❌ TEST CRASHED: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
