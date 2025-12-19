# Quick Start Guide

## Installation

1. **Install Gurobi** (if not already installed):
   - Academic license available at: https://www.gurobi.com/academia/
   - Already installed and tested ✓

2. **Install required packages**:
```bash
pip3 install gurobipy numpy pandas matplotlib
```

---

## Running the Code

### Option 1: Quick Test (Toy Instance)
```bash
python3 DH_main.py toy
```
- Runs in ~1 second
- Tests Gamma values: [0, 1, 2, 3, 4, 5]
- Saves results to `result/DH_sensitivity_toy_*.csv`

### Option 2: Full Analysis
```bash
python3 DH_main.py full
```
- Runs K=3, I=5, J=20, R=100, M=3
- May take several minutes
- Tests Gamma: [0, 10, 20, ..., 100]

### Option 3: Generate New Data
```bash
python3 DH_data_gen.py
```
- Creates fresh random instances
- Saves to `data/DH_data_toy.pkl` and `data/DH_data_full.pkl`

---

## Checking Results

### View Results CSV
```bash
cat result/DH_sensitivity_toy_temp.csv
```

Expected columns:
- `Gamma`: Uncertainty budget
- `Converged`: True if gap < ε
- `Iterations`: Number of C&CG iterations
- `Total_Time`: Solve time in seconds
- `Optimal_Value`: Best robust profit found
- `LB`, `UB`: Lower/upper bounds
- `Gap`: UB - LB
- `Num_Scenarios`: Critical scenarios identified

### View Results Plot
```bash
open result/DH_sensitivity_toy_*.png
```

Shows 4 plots:
1. Optimal value vs Gamma
2. Number of scenarios vs Gamma
3. Iterations vs Gamma
4. Solve time vs Gamma

---

## Understanding Output

### Console Output Structure
```
================================================================================
RUNNING C&CG FOR Γ = 0
================================================================================
INITIALIZING C&CG ALGORITHM
  Problem: K=1, I=2, J=2, R=5, M=2
  Uncertainty Budget: Γ = 0
  Convergence Tolerance: ε = 0.0001
================================================================================
ITERATION 1
  [Master Problem solving...]
  Master solved in 0.02s
  Objective = -78269.00, θ = -5752.35
  Upper Bound (UB) = -78269.00

  [Subproblem solving...]
  Subproblem solved in 0.00s
  Worst-case operational profit (Z_SP) = -17257.04
  Scenario: 0 demand increases, 0 demand decreases
  True Robust Profit = -89773.69
  Lower Bound (LB) = -89773.69
  Gap = 11504.69
================================================================================
```

### Key Metrics to Monitor
- **Gap**: Should decrease to < 0.0001 at convergence
- **Iterations**: Typically 5-20 for small instances
- **Scenarios**: Number of critical worst-case scenarios found
- **Plants/DCs Opened**: Should be non-zero for feasible solution

---

## Current Known Issues ⚠️

1. **Algorithm Stalls with Gap**
   - Symptom: Terminates after 1-2 iterations with "Duplicate scenario detected"
   - Gap remains around 11,000-15,000
   - **Cause**: Formulation issue (see DH_README.md for details)

2. **No Plants Opened**
   - Symptom: Solution has 0 plants, all demand is shortage
   - **Cause**: Fixed costs too high relative to revenue

3. **Not Converging**
   - Symptom: Converged = False in all results
   - **Cause**: Master's θ estimate doesn't match Subproblem's Z_SP

---

## Debugging Tips

### Check if Gurobi is Working
```bash
python3 gurobitest.py
```
Should output:
```
Status: 2
x = 1.0
Objective = 1.0
```

### Test Data Generation
```bash
python3 -c "from DH_data_gen import *; config = ProblemConfig('toy'); data = generate_supply_chain_data(config); data.summary()"
```

### Test Master Problem Alone
```python
from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_master import MasterProblem

config = ProblemConfig('toy')
config.set_gamma(0)
data = SupplyChainData.load('data/DH_data_toy.pkl')
master = MasterProblem(data, config)

# Add nominal scenario
eta_plus = {(r,k): 0 for r in range(5) for k in range(1)}
eta_minus = {(r,k): 0 for r in range(5) for k in range(1)}
master.add_scenario(0, eta_plus, eta_minus)

# Solve
master.solve()
solution = master.get_solution()
print(f"Objective: {solution['objective']}")
print(f"Theta: {solution['theta']}")
```

### Enable Verbose Gurobi Output
Edit `DH_config.py` line 38:
```python
self.gurobi_output_flag = 1  # Already enabled
```

---

## Modifying Parameters

### Change Problem Size
Edit `DH_config.py`:
```python
# Line 18-22 (toy instance)
self.K = 2  # Products (was 1)
self.I = 3  # Plants (was 2)
self.J = 4  # DCs (was 2)
self.R = 10  # Customers (was 5)
self.M = 2  # Modes (was 2)
```

### Change Cost Parameters
Edit `DH_config.py` `DataParameters` class:
```python
# Line 105-107
C_plant_min = 5000   # Reduce fixed costs (was 50000)
C_plant_max = 15000  # (was 150000)
```

### Change Selling Price / Shortage Cost
Edit `DH_config.py`:
```python
# Line 102-103
S = 200.0   # Increase revenue (was 100.0)
SC = 30.0   # Reduce shortage penalty (was 50.0)
```

### Change Convergence Tolerance
Edit `DH_config.py`:
```python
# Line 33
self.epsilon = 1e-2  # Looser tolerance (was 1e-4)
```

---

## File Locations

```
/Users/dh_kim/PycharmProjects/FirstPaper/
├── DH_config.py          # Configuration
├── DH_data_gen.py        # Data generation
├── DH_master.py          # Master Problem
├── DH_sub.py             # Subproblem
├── DH_algo.py            # C&CG algorithm
├── DH_main.py            # Main script
├── DH_README.md          # Full documentation
├── DH_QUICKSTART.md      # This file
├── data/
│   ├── DH_data_toy.pkl   # Toy instance data
│   └── DH_data_full.pkl  # Full instance data
└── result/
    ├── DH_sensitivity_toy_*.csv   # Results
    └── DH_sensitivity_toy_*.png   # Plots
```

---

## Next Steps for Users

1. **Read** `DH_README.md` for detailed documentation
2. **Review** `algorithm_framework.tex` for mathematical formulation
3. **Debug** convergence issue (see DH_README.md "Debugging Recommendations")
4. **Adjust** cost parameters to get realistic solutions
5. **Test** with full instance once toy instance works correctly

---

## Questions?

Check these resources in order:
1. **This file** for usage instructions
2. **DH_README.md** for implementation details
3. **algorithm_framework.tex** for theory
4. **Code comments** in individual Python files
5. **Gurobi documentation** at https://www.gurobi.com/documentation/

---

**Last Updated**: 2025-12-18
**Status**: Implementation complete, debugging needed
