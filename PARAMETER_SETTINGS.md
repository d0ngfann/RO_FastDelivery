# Parameter Settings Documentation

This document describes how all model parameters are generated and configured in the robust supply chain optimization framework.

## Table of Contents
1. [Overview](#overview)
2. [Problem Dimensions](#problem-dimensions)
3. [Economic Parameters](#economic-parameters)
4. [Cost Parameters](#cost-parameters)
5. [Capacity Parameters](#capacity-parameters)
6. [Demand Parameters](#demand-parameters)
7. [Spatial Parameters](#spatial-parameters)
8. [Transportation Parameters](#transportation-parameters)
9. [Algorithm Settings](#algorithm-settings)
10. [Random Generation Process](#random-generation-process)

---

## Overview

Parameters are set through two main components:
- **DH_config.py**: Defines parameter ranges and configuration settings
- **DH_data_gen.py**: Implements random data generation using these ranges

All parameters use **uniform random distributions** unless otherwise specified, with fixed random seed (default: 42) for reproducibility.

---

## Problem Dimensions

Two instance sizes are supported:

### Toy Instance (for testing)
```
K = 1   # Products
I = 2   # Plants
J = 2   # Distribution Centers
R = 5   # Customers
M = 2   # Transportation modes
Grid Size = 50 × 50
```

### Full Instance (production)
```
K = 3   # Products
I = 5   # Plants
J = 20  # Distribution Centers
R = 100 # Customers
M = 3   # Transportation modes
Grid Size = 100 × 100
```

**Location**: `DH_config.py:10-34`

---

## Economic Parameters

### 1. Selling Price (S)
- **Value**: `S = 100.0` ($/unit)
- **Type**: Scalar, uniform across all products
- **Purpose**: Unit revenue for satisfied demand
- **Location**: `DH_config.py:81`

### 2. Shortage Cost (SC)
- **Value**: `SC = 50.0` ($/unit)
- **Type**: Scalar, penalty for unmet demand
- **Purpose**: Cost incurred per unit of shortage
- **Note**: SC = 50% of selling price (reasonable penalty)
- **Location**: `DH_config.py:82`

---

## Cost Parameters

All cost parameters are randomly generated using uniform distributions.

### 1. Plant Fixed Cost (C^plant_{ki})
Opening cost for plant i to produce product k.

```python
C^plant_{ki} ~ Uniform[5,000, 15,000] for all (k,i)
```

- **Dimensions**: K × I matrix
- **Range**: $5,000 to $15,000
- **Generation**: Independent random draw for each (k,i) pair
- **Location**: `DH_config.py:86-87`, `DH_data_gen.py:224-225`

### 2. DC Fixed Cost (C^dc_j)
Opening cost for distribution center j.

```python
C^dc_j ~ Uniform[3,000, 8,000] for all j
```

- **Dimensions**: J vector
- **Range**: $3,000 to $8,000
- **Generation**: Independent random draw for each DC
- **Location**: `DH_config.py:90-91`, `DH_data_gen.py:228-229`

### 3. Ordering Cost (O_j)
Fixed cost per order at DC j.

```python
O_j ~ Uniform[5,000, 15,000] for all j
```

- **Dimensions**: J vector
- **Range**: $5,000 to $15,000
- **Location**: `DH_config.py:94-95`, `DH_data_gen.py:232-233`

### 4. Production Unit Cost (F_{ki})
Variable cost to produce one unit of product k at plant i.

```python
F_{ki} ~ Uniform[10.0, 30.0] for all (k,i)
```

- **Dimensions**: K × I matrix
- **Range**: $10 to $30 per unit
- **Location**: `DH_config.py:106-107`, `DH_data_gen.py:237-238`

### 5. Holding Cost (h_j)
Unit inventory holding cost at DC j.

```python
h_j ~ Uniform[2.0, 8.0] for all j
```

- **Dimensions**: J vector
- **Range**: $2 to $8 per unit
- **Location**: `DH_config.py:110-111`, `DH_data_gen.py:241-242`

### 6. Plant-to-DC Transportation Cost (t)
Base transportation cost from plant to DC (per unit distance).

```python
t = 0.1 ($/unit·distance)
```

- **Type**: Scalar constant
- **Purpose**: Linear transportation cost component
- **Location**: `DH_config.py:114`

### 7. Route Fixed Cost: Plant-to-DC (L1_{kij})
Fixed cost for establishing route from plant i to DC j for product k.

```python
L1_{kij} = base_cost + 0.5 × D1_{kij}
where base_cost ~ Uniform[1,000, 5,000]
```

- **Dimensions**: K × I × J tensor
- **Components**:
  - Base cost: $1,000 to $5,000 (random)
  - Distance component: 0.5 × Euclidean distance
- **Location**: `DH_config.py:98-99`, `DH_data_gen.py:316-319`

### 8. Route Fixed Cost: DC-to-Customer (L2_{jr})
Fixed cost for establishing route from DC j to customer r.

```python
L2_{jr} = base_cost + 0.3 × D2_{jr}
where base_cost ~ Uniform[500, 2,000]
```

- **Dimensions**: J × R matrix
- **Components**:
  - Base cost: $500 to $2,000 (random)
  - Distance component: 0.3 × Euclidean distance
- **Location**: `DH_config.py:102-103`, `DH_data_gen.py:323-326`

---

## Capacity Parameters

### 1. Plant Capacity (MP_{ki})
Maximum production capacity of plant i for product k.

```python
MP_{ki} ~ Uniform[5,000, 15,000] units for all (k,i)
```

- **Dimensions**: K × I matrix
- **Range**: 5,000 to 15,000 units
- **Location**: `DH_config.py:131-132`, `DH_data_gen.py:247-248`

### 2. DC Capacity (MC_j)
Maximum storage/handling capacity of DC j.

```python
MC_j ~ Uniform[3,000, 10,000] units for all j
```

- **Dimensions**: J vector
- **Range**: 3,000 to 10,000 units
- **Location**: `DH_config.py:135-136`, `DH_data_gen.py:251-252`

---

## Demand Parameters

### 1. Demand Indicator (s_{rk})
Binary indicator: does customer r demand product k?

```python
s_{rk} ~ Bernoulli(0.5) for all (r,k)
```

- **Dimensions**: R × K binary matrix
- **Generation Process**:
  1. Random binary matrix with p=0.5 for each entry
  2. **Constraint**: Each customer must demand at least 1 product
  3. If all entries for customer r are 0, randomly set one to 1
- **Sparsity**: Approximately 50% of (r,k) pairs have demand
- **Location**: `DH_data_gen.py:256-266`

### 2. Nominal Demand (μ_{rk})
Expected demand quantity for product k at customer r.

```python
μ_{rk} = { 10 × Uniform[1, 5]  if s_{rk} = 1
         { 0                   if s_{rk} = 0
```

- **Dimensions**: R × K matrix
- **Range**: 10 to 50 units (when s_{rk} = 1)
- **Formula**: `μ_{rk} = 10 × U[1,5]` where U is uniform random variable
- **Location**: `DH_data_gen.py:269-276`

**Example Values**:
- If U[1,5] = 1.5 → μ = 15 units
- If U[1,5] = 4.8 → μ = 48 units

### 3. Demand Deviation (μ̂_{rk})
Maximum uncertainty in demand (±μ̂).

```python
μ̂_{rk} = { min(μ_{rk}, Uniform[4, 10])  if s_{rk} = 1
         { 0                             if s_{rk} = 0
```

- **Dimensions**: R × K matrix
- **Formula**: Take minimum of nominal demand and random value in [4,10]
- **Purpose**: Bounds the uncertainty range
- **Location**: `DH_data_gen.py:279-286`

**Example Values**:
- If μ = 15, U[4,10] = 7 → μ̂ = min(15, 7) = 7 units (±46.7% uncertainty)
- If μ = 45, U[4,10] = 9 → μ̂ = min(45, 9) = 9 units (±20% uncertainty)

### 4. Demand Increase Factors (DI_{mk})
Endogenous demand multiplier based on transportation mode.

```python
DI matrix (M × K):
Mode 0 (slow):   [1.0, 1.0, 1.0]  # No increase
Mode 1 (medium): [1.2, 1.2, 1.2]  # 20% increase
Mode 2 (fast):   [1.5, 1.5, 1.5]  # 50% increase
```

- **Dimensions**: M × K matrix
- **Interpretation**: Faster delivery modes increase base demand
- **Uniform across products**: Same factor for all products per mode
- **Location**: `DH_config.py:123-127`, `DH_data_gen.py:216-219`

### 5. Realized Demand Formula
The actual demand in a scenario is:

```
d_{rk} = Σ_m μ_{rk} × DI_{mk} × β_{rm} + (η^+_{rk} - η^-_{rk}) × μ̂_{rk}
```

Where:
- `β_{rm}` = 1 if mode m selected for customer r, 0 otherwise
- `η^+_{rk}` ∈ {0,1} = demand increase indicator
- `η^-_{rk}` ∈ {0,1} = demand decrease indicator
- Subject to uncertainty budget: `Σ_r (η^+_{rk} + η^-_{rk}) ≤ Γ_k`

---

## Spatial Parameters

### 1. Coordinate Generation

Three different spatial patterns are used for realism:

#### Plants (Uniform Distribution)
```python
(x_i, y_i) ~ Uniform[0, grid_size]² for each plant i
```
- **Pattern**: Uniformly scattered across entire grid
- **Rationale**: Plants are geographically diverse sources
- **Location**: `DH_data_gen.py:291`

#### DCs (Donut Pattern)
```python
(x_j, y_j) sampled from grid excluding central 60% box
```
- **Pattern**: Exclude region [0.2×grid, 0.8×grid]²
- **Rationale**: DCs typically in peripheral/suburban areas
- **Location**: `DH_data_gen.py:293`

#### Customers (Gaussian Clustering)
```python
x_r ~ N(grid_size/2, grid_size/5)
y_r ~ N(grid_size/2, grid_size/5)
```
- **Pattern**: Clustered around grid center (clipped to boundaries)
- **Rationale**: Customers concentrated in urban/central areas
- **Location**: `DH_data_gen.py:295`

### 2. Distance Calculation (D1_{kij}, D2_{jr})

Euclidean distances between facilities:

```python
D1_{kij} = √[(x_i - x_j)² + (y_i - y_j)²]  # Plant i to DC j
D2_{jr}  = √[(x_j - x_r)² + (y_j - y_r)²]  # DC j to customer r
```

- **D1**: Same for all products k (distance independent of product)
- **D2**: Directly used in transportation costs
- **Location**: `DH_data_gen.py:299-309`

---

## Transportation Parameters

### 1. Mode-Specific Costs (TC_m)
Variable transportation cost per unit distance for each mode.

```python
TC_0 = 0.05  # Slow/cheap (e.g., truck)
TC_1 = 0.10  # Medium (e.g., express truck)
TC_2 = 0.20  # Fast/expensive (e.g., air freight)
```

- **Dimensions**: M vector
- **Unit**: $/unit·distance
- **Trade-off**: Higher cost modes increase demand (via DI factors)
- **Location**: `DH_config.py:118`, `DH_data_gen.py:211-213`

### 2. Total Transportation Cost
From DC j to customer r using mode m with flow A^k_{jr}:

```
TC_total = TC_m × D2_{jr} × Σ_k A^k_{jr}
```

---

## Algorithm Settings

### 1. Convergence Parameters
```python
epsilon = 1e-4         # Convergence tolerance (UB - LB)
max_iterations = 100   # Maximum C&CG iterations
```
**Location**: `DH_config.py:37-38`

### 2. Gurobi Solver Settings
```python
# Time limits
time_limit = 3600      # seconds (1 hour per optimization)

# Tolerance settings
mip_gap = 1e-6         # MIP optimality gap
feasibility_tol = 1e-9 # Primal feasibility
int_feas_tol = 1e-9    # Integer feasibility
opt_tol = 1e-9         # Dual feasibility

# Parallelization
threads = 0            # Use all available CPU cores
```
**Location**: `DH_config.py:41-49`

### 3. Uncertainty Budget (Gamma)
```python
Γ_k ∈ [0, R] for each product k
```

- **Interpretation**: Maximum number of customers with demand uncertainty per product
- **Sensitivity Analysis**:
  - Toy instance: Γ ∈ {0, 1, 2, 3, 4, 5}
  - Full instance: Γ ∈ {0, 10, 20, ..., 100}
- **Location**: `DH_config.py:52-67`

---

## Random Generation Process

### Reproducibility
All random generation uses fixed seeds for reproducibility:

```python
Main seed: 42
Plant coordinates: seed = 42
DC coordinates: seed = 43
Customer coordinates: seed = 44
```
**Location**: `DH_data_gen.py:197, 291-295`

### Generation Order
1. **Scalar economic parameters** (S, SC, t)
2. **Transportation mode costs** (TC_m)
3. **Demand increase factors** (DI_{mk})
4. **Fixed costs** (C_plant, C_dc, O)
5. **Variable costs** (F, h)
6. **Capacities** (MP, MC)
7. **Demand structure**:
   - Binary indicators s_{rk}
   - Nominal demand μ_{rk} (conditional on s_{rk})
   - Demand deviation μ̂_{rk} (conditional on s_{rk})
8. **Spatial layout**:
   - Coordinates for plants, DCs, customers
   - Distances D1, D2
9. **Distance-based costs** (L1, L2)

**Location**: `DH_data_gen.py:186-329`

### Independence Assumptions
- All random draws are **independent** except:
  - μ̂_{rk} depends on μ_{rk} (via min function)
  - L1, L2 depend on distances D1, D2
  - s_{rk} enforces "at least one product per customer" constraint

---

## Data Storage

### File Format
```python
# Pickle files
data/DH_data_toy.pkl   # Toy instance
data/DH_data_full.pkl  # Full instance
```

### Data Structure
All parameters stored in `SupplyChainData` class with dictionary attributes:
- **Indexed by tuples**: e.g., `mu[(r,k)]`, `C_plant[(k,i)]`
- **0-based indexing**: All indices start from 0
- **Type safety**: Integer indices, float values

**Location**: `DH_data_gen.py:13-75`

---

## Validation and Summary

After generation, the `summary()` method displays:
- Problem dimensions
- Economic parameters
- Capacity ranges (min/max)
- Demand sparsity percentage
- Average products per customer
- Cost ranges across all categories

**Location**: `DH_data_gen.py:77-113`

### Example Summary Output (Toy Instance)
```
Dimensions: K=1, I=2, J=2, R=5, M=2
Economic Parameters:
  Selling price (S): 100.0
  Shortage cost (SC): 50.0
Capacities:
  Plant capacity (MP): min=7234, max=12891
  DC capacity (MC): min=4567, max=8912
Demand Structure (s_rk matrix):
  Total customer-product pairs: 3/5 (40.0% sparse)
  Average products per customer: 0.60
```

---

## References

### Configuration Files
- `DH_config.py`: Parameter ranges and settings
- `DH_data_gen.py`: Data generation implementation

### Mathematical Documentation
- `algorithm_framework.tex`: Complete mathematical formulation
- `FORMULA_VERIFICATION.md`: Formula implementation verification

### Usage
```bash
# Generate new data instances
python3 DH_data_gen.py

# Run with generated data
python3 DH_main.py toy    # or 'full'
```

---

## Notes

1. **Scaling**: Parameter ranges were carefully tuned to ensure:
   - Feasible solutions exist
   - Non-trivial optimization (not all facilities always open)
   - Realistic trade-offs between costs

2. **Consistency**: All indices use Python's 0-based convention

3. **Units**: All monetary values in dollars ($), all quantities in units

4. **Extensibility**: New instance sizes can be added by extending `ProblemConfig.__init__()`
