# Column-and-Constraint Generation (C&CG) Implementation

## Overview

This implementation provides a complete **Column-and-Constraint Generation (C&CG)** algorithm for solving a **two-stage robust supply chain optimization problem** with **bidirectional demand uncertainty**, as described in `algorithm_framework.tex`.

### Problem Characteristics
- **Three-echelon network**: Plants → Distribution Centers → Customers
- **Endogenous demand**: Transportation mode selection affects customer demand
- **Bidirectional uncertainty**: Considers both demand increases (shortage risk) and decreases (revenue loss)
- **Two-stage decisions**:
  - **First-stage** (strategic): Facility locations, routes, mode selections
  - **Second-stage** (operational): Production quantities, flows, shortages

---

## File Structure

### Core Implementation Files (Created)

1. **`DH_config.py`** - Configuration and parameters
   - `ProblemConfig`: Problem dimensions and algorithm settings
   - `DataParameters`: Default cost/capacity parameters for data generation
   - `SensitivityConfig`: Gamma sensitivity analysis settings

2. **`DH_data_gen.py`** - Synthetic data generation
   - `SupplyChainData`: Data container class
   - `generate_supply_chain_data()`: Creates random instances with realistic parameters
   - Generates coordinates and Euclidean distances
   - Supports both toy and full-scale instances

3. **`DH_master.py`** - Master Problem formulation
   - **Variables**: First-stage (x, y, z, w, β, α) + second-stage per scenario (A_ij, A_jr, u)
   - **Objective**: max -OC - FC + θ (maximize robust profit)
   - **Constraints**:
     - Network topology (single sourcing, mode selection)
     - Operational constraints per scenario (capacities, flow balance, demand satisfaction)
     - Optimality cuts: θ ≤ operational profit for each scenario
   - **Linearization**: Big-M method for bilinear terms α_jrm * A_jr^k

4. **`DH_sub.py`** - Subproblem (SP-Dual) formulation
   - **Dual formulation** of inner maximization problem
   - **Variables**: Dual variables (π, σ, ψ, φ, γ, κ) + uncertainty (η^+, η^-)
   - **Objective**: Minimize worst-case operational profit
   - **Constraints**:
     - Dual feasibility constraints
     - Uncertainty budget: Σ(η^+ + η^-) ≤ Γ per product
     - McCormick linearization for bilinear term (η^+ - η^-) * γ
   - **Tight bounds**: γ^L = -(S+SC), γ^U = S

5. **`DH_algo.py`** - Main C&CG algorithm
   - Iterative Master-Subproblem loop
   - Upper/lower bound tracking
   - Convergence detection (gap < ε)
   - **Duplicate scenario detection** (prevents infinite loops)
   - Iteration logging and statistics

6. **`DH_main.py`** - Execution script
   - Sensitivity analysis over Gamma values
   - Automatic results saving (CSV format)
   - Visualization (convergence plots)

---

## Usage

### Generate Data
```bash
python3 DH_data_gen.py
```
Creates `data/DH_data_toy.pkl` and `data/DH_data_full.pkl`

### Run Toy Instance
```bash
python3 DH_main.py toy
```

### Run Full Instance
```bash
python3 DH_main.py full
```

### Results
- **CSV**: `result/DH_sensitivity_{instance}_{timestamp}.csv`
- **Plots**: `result/DH_sensitivity_{instance}_{timestamp}.png`

---

## Implementation Details

### Problem Dimensions

**Toy Instance** (for testing):
- K=1, I=2, J=2, R=5, M=2

**Full Instance** (production):
- K=3, I=5, J=20, R=100, M=3

### Algorithm Parameters
- Convergence tolerance: ε = 10^-4
- Max iterations: 100
- Gurobi time limit: 3600s per solve

### Key Formulation Details

1. **Endogenous Demand**:
   ```
   d_rk = Σ_m μ_rk * DI_mk * β_rm + (η^+_rk - η^-_rk) * μ̂_rk
   ```

2. **Revenue Calculation** (critical for correctness):
   ```
   Revenue = S * (demand - shortage)
   ```
   Only satisfied demand generates revenue.

3. **Uncertainty Set**:
   ```
   η_rk ∈ {-1, 0, 1} via binary decomposition
   Σ_r |η_rk| ≤ Γ_k (budget constraint)
   ```

4. **McCormick Linearization**:
   - Decompose ξ_rk = (η^+ - η^-) * γ_rk into p^+ and p^-
   - Use tight bounds for better LP relaxation

---

## Current Status

### ✅ Completed
- All 6 core Python modules implemented
- Data generation working correctly
- Gurobi models build without errors
- Master Problem solves successfully
- Subproblem (dual formulation) solves successfully
- Duplicate scenario detection implemented
- Sensitivity analysis framework functional

### ⚠️  Known Issues

1. **Convergence Gap Persists**
   - Algorithm terminates with gap ~10,000-15,000 (toy instance)
   - Master's θ estimate doesn't match Subproblem's Z_SP
   - **Root cause**: Inconsistency between Master and Subproblem calculations

2. **Degenerate Solutions**
   - Toy instance finds solutions with no plants opened
   - All demand satisfied by shortages
   - **Possible cause**: Cost parameters make production unprofitable
   - Fixed costs (87k-145k) >> demand value with current prices

3. **Algorithm Stalling**
   - Terminates after 1-2 iterations for most Gamma values
   - Subproblem keeps finding same scenario (duplicate detection triggers)
   - **Interpretation**: Either (a) true convergence for degenerate solution, or (b) formulation bug

---

## Debugging Recommendations

### Priority 1: Verify Optimality Cut Formulation

**Issue**: Master's θ doesn't match Subproblem's Z_SP for the same scenario.

**Check**:
1. Compare Master's operational profit calculation (in `add_scenario()`) with Subproblem's dual objective
2. Verify that fixed first-stage variables in Subproblem match Master's solution
3. Add debug output to print:
   - Master: revenue, HC, TC, PC, SC for each scenario
   - Subproblem: Z_SP breakdown

**File**: `DH_master.py` lines 376-453 (optimality cut construction)

### Priority 2: Validate Cost Parameters

**Issue**: Solutions with no production might be economically infeasible but mathematically optimal.

**Fix Options**:
1. Reduce fixed costs in `DataParameters` class
2. Increase selling price S or reduce shortage cost SC
3. Add minimum production constraints

**File**: `DH_config.py` lines 101-137

### Priority 3: Test with Simplified Model

**Approach**:
1. Set Γ = 0 (no uncertainty)
2. Fix first-stage variables manually
3. Solve Master and Subproblem separately
4. Compare operational profits numerically

**File**: Create `DH_debug.py` for isolated testing

### Priority 4: Check Dual Formulation

**Issue**: Subproblem dual might have errors in McCormick constraints or objective.

**Verify**:
1. Manually derive dual for a small case (R=1, K=1)
2. Compare with implemented dual
3. Check McCormick bounds match: γ^L = -(S+SC), γ^U = S

**File**: `DH_sub.py` lines 225-265 (McCormick linearization)

---

## Next Steps

1. **Debug convergence issue** using recommendations above
2. **Validate formulation** with known test cases
3. **Tune cost parameters** for realistic solutions
4. **Add unit tests** for Master/Subproblem consistency
5. **Implement primal recovery** (optional) for Subproblem verification
6. **Scale to full instance** once toy instance converges correctly

---

## Technical Notes

### Dependencies
- `gurobipy` (Gurobi Optimizer 10.0+)
- `numpy`, `pandas`, `matplotlib`
- `pickle` (data serialization)

### Performance
- Toy instance: ~0.02s per Gamma value
- Full instance: Expected ~10-60s per iteration (untested)

### Mathematical Reference
See `algorithm_framework.tex` for:
- Complete mathematical formulation
- Dual derivation
- McCormick linearization proofs
- Convergence properties

---

## Contact / Support

For questions about the implementation, refer to:
- **Algorithm theory**: `algorithm_framework.tex`
- **Code structure**: This README
- **Debugging**: Section above
- **Data format**: `DH_data_gen.py` documentation

---

**Implementation Date**: 2025-12-18
**Status**: Core implementation complete, debugging in progress
**Next Milestone**: Resolve convergence gap for toy instance
