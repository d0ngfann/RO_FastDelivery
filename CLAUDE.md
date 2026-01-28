# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**whenever you change the code you should revise based on algorithm_framework.tex, this file is what we are implementing** 

## Project Overview

This is a research implementation of a **Robust Supply Chain Optimization** algorithm using **Column-and-Constraint Generation (C&CG)** for solving a three-echelon supply chain design problem under demand uncertainty. The project implements a two-stage robust optimization model with bidirectional uncertainty sets.

**Research Context**: This appears to be a paper submission codebase with 50 pre-generated datasets and 4 demand sensitivity scenarios (HD, MD, LD, Mixed) for statistical analysis.

## Running the Code

### Main Execution

```bash
# Run single experiment with specific parameters
python3 DH_main.py [instance_type] [gamma] --seed [seed] --di [scenario]

# Examples:
python3 DH_main.py full 10 --seed 1 --di HD        # Full instance, Gamma=10, Dataset 1, High Demand sensitivity
python3 DH_main.py toy 5 --di Mixed                # Toy instance, Gamma=5, default data, Mixed scenario
python3 DH_main.py full --seed 5 --di HD           # Sensitivity analysis (all Gamma values)
```

**Parameters**:
- `instance_type`: `toy` (K=1, I=2, J=2, R=5, M=2) or `full` (K=3, I=3, J=5, R=50, M=3)
- `gamma`: Uncertainty budget (0 to R). If omitted, runs full sensitivity analysis. Recommended: 10 for full instance
- `--seed`: Dataset number (1-50 for full, 1-5 for toy). If omitted, uses default data
- `--di`: DI scenario (`HD`, `MD`, `LD`, `Mixed`). **Must be uppercase**

### Batch Execution for SLURM

```bash
# Generate 200 SLURM job files (50 seeds × 4 scenarios)
python3 generate_sbatch_files.py

# Submit all jobs
cd sbatch && for f in *.sbatch; do sbatch $f; done
```

### Data Generation

```bash
# Generate multiple datasets with different random seeds
python3 generate_50_seeds.py
```

## Code Architecture

### Module Organization

The codebase follows a clean separation of concerns with specialized modules:

1. **DH_config.py** - Configuration management
2. **DH_data_gen.py** - Data generation and container
3. **DH_master.py** - Master Problem (MP) formulation
4. **DH_sub.py** - Subproblem (SP) dual formulation
5. **DH_algo.py** - C&CG algorithm orchestration
6. **DH_main.py** - Main execution and sensitivity analysis

### Algorithm Flow (C&CG)

The C&CG algorithm iterates between Master Problem and Subproblem:

```
Initialize with nominal scenario (η = 0)
Loop until convergence or max iterations:
  1. Solve Master Problem (MP)
     - Optimizes first-stage decisions (x, y, z, w, β, α)
     - Approximates worst-case operational profit (θ)
     - Updates Upper Bound (UB)

  2. Solve Subproblem (SP)
     - Finds worst-case demand scenario (η⁺, η⁻)
     - Fixed first-stage decisions from MP
     - Returns worst-case operational profit (Z_SP)
     - Updates Lower Bound (LB)

  3. Check convergence: |UB - LB| ≤ ε

  4. Add new scenario to Master Problem
     - Creates optimality cut
     - Prevents duplicate scenarios
```

### Key Design Patterns

**Product-Specific Plants**: Each product k has its own dedicated set of I candidate plant locations (NOT shared between products). This means:
- Total candidate plants = K × I = 9 for full instance (3 products × 3 plants each)
- Plant index `x[k,i]` refers to plant i for product k specifically
- Routes `z[k,i,j]` are also product-specific (product k from plant i to DC j)
- DC variables `y[j]` remain product-agnostic (shared across all products)

**Scenario Management**:
- Master Problem dynamically adds scenarios as columns/constraints
- Each scenario has `(scenario_id, eta_plus, eta_minus)` structure
- Duplicate detection prevents infinite loops

**Demand Sensitivity (DI) Scenarios**:
All datasets contain 4 pre-generated DI scenarios:
- **HD** (High Demand sensitivity): k ∈ [0.667, 1.000] → Fast delivery causes large demand increase
- **MD** (Medium): k ∈ [0.333, 0.667] → Moderate sensitivity
- **LD** (Low): k ∈ [0.000, 0.333] → Minimal sensitivity
- **Mixed**: Product-specific (P0=HD, P1=MD, P2=LD)

Formula: `DI_m = (3/2)^(k × m)` where m ∈ {0, 1, 2} (modes)

## Solver Dependencies

**Critical**: This code requires either **Gurobi** or **CPLEX** (legacy support).

- Gurobi is the primary solver (see `requirements.txt`)
- CPLEX variants exist in `__pycache__/` but are not actively maintained
- Solver parameters are tightly configured in `DH_config.py` for numerical stability:
  - `MIPGap = 1e-6` (very tight)
  - `FeasibilityTol = 1e-9`
  - `IntFeasTol = 1e-9`
  - `OptimalityTol = 1e-9`

**License requirement**: SLURM jobs load `module load gurobi/12.0.3` for license access.

## Result Files

Results are saved to `result/` directory:

**Single run**:
```
result/DH_single_[instance]_seed[N]_[DI]_gamma[G]_[timestamp].csv
```

**Sensitivity analysis**:
```
result/DH_sensitivity_[instance]_[timestamp].csv
result/DH_sensitivity_[instance]_[timestamp].png
```

CSV columns include: Gamma, Seed, DI_Scenario, Converged, Iterations, Total_Time, Optimal_Value, LB, UB, Gap, Num_Scenarios, Num_Plants_Opened, Num_DCs_Opened, Opened_Plants, Opened_DCs

## Mathematical Model

The implementation follows the formulation in `algorithm_framework.tex`:

- **First-stage decisions**: x (plant opening), y (DC opening), z (plant-DC routes), w (DC-customer assignment), β (mode selection), α (mode-route combination)
- **Second-stage decisions**: A_ij (plant-DC flow), A_jr (DC-customer flow), u (shortage)
- **Uncertainty**: Bidirectional demand deviations (η⁺, η⁻) within budget Γ_k
- **Objective**: Maximize robust profit = Revenue - Shortage Cost - Opening Costs - Route Costs - Variable Costs

## Development Notes

**Data persistence**: All datasets use pickle format (`.pkl`) and are stored in `data/` directory. Datasets include complete problem instances with all 4 DI scenarios embedded.

**Convergence tolerance**: Set to `epsilon = 200` (absolute gap) in config. This is calibrated for the problem scale and cost magnitudes.

**Computational complexity**:
- Toy instances: seconds to minutes
- Full instances with Gamma=10: 5-12 hours per run (reduced from 12-24 hours with J=10, Gamma=20)
- Full instances with Gamma=20: 12-24 hours per run (with old J=10 configuration)
- Memory usage: Up to 128GB for SLURM jobs (full instance)
- Configuration optimized for 1-2 DC hub-based networks (J=5, high capacity DCs)

**Result reproducibility**: Fixed seed guarantees identical data generation. Same (seed, DI, gamma) combination produces deterministic results.

**McCormick linearization**: The subproblem uses McCormick envelopes for bilinear term linearization (p_plus, p_minus variables).

## File Structure to Ignore

- `.venv/` - Python virtual environment (Python 3.7)
- `__pycache__/` - Contains both Python 3.7 and 3.10 bytecode, plus legacy CPLEX variants
- `Unused/` - Deprecated documentation and old implementations
- `.idea/` - PyCharm IDE settings
- `gurobi.log` - Solver log file

## Common Pitfalls

1. **DI scenario case sensitivity**: Must use uppercase (`HD`, not `hd`)
2. **Gamma must be set**: Config requires explicit `config.set_gamma(value)` before algorithm initialization
3. **Seed ranges**: Full instance supports seeds 1-50, toy supports 1-5
4. **Duplicate scenarios**: Algorithm detects and terminates when subproblem returns duplicate scenario (indicates stalling)
5. **Negative gaps**: Small negative gaps (< epsilon) are acceptable due to solver tolerance; large negative gaps indicate numerical issues
