# Implementation Guide: Robust Optimization with C&CG (Bidirectional & Endogenous Demand)

This document outlines the implementation details, architectural decisions, and critical precautions for coding the Two-Stage Robust Optimization model described in `algorithm_framework.tex`.

## 1. Environment & Dependencies
* **Language:** Python 3.8+
* **Solver:** A commercial MILP solver is highly recommended due to the complexity of the decomposition (e.g., **Gurobi**, **CPLEX**, or **Xpress**). Open-source solvers (CBC, GLPK) may struggle with convergence speed for this specific bilevel structure.
* **Libraries:**
    * `gurobipy` (or `docplex`, `pulp`) for optimization modeling.
    * `numpy` / `pandas` for parameter management.
    * `dataclasses` (optional but recommended) for structured data storage.

---

## 2. Data Structures & Indexing
Consistency in indexing is vital. Use the following standard indices throughout the code:

* **Sets:**
    * `K`: Products
    * `I`: Plants
    * `J`: Distribution Centers (DCs)
    * `R`: Customers
    * `M`: Transportation Modes
* **Scenario Set ($\mathcal{L}$):**
    * Implement as a dynamic list.
    * Start with `L = [0]` (Nominal scenario where all $\eta = 0$).
    * In each iteration, append the new worst-case $\eta$ found by the Subproblem.

---

## 3. Master Problem (MP) Implementation
**Goal:** Determine strategic variables ($x, y, z, w, \beta, \alpha$) and the value $\vartheta$ (worst-case operational profit).

### 3.1. Variables
* **Strategic (Binary):** Define $x_i, y_j, z_{ij}, w_{jr}, \beta_{rm}, \alpha_{jrm}$ only once.
* **Operational (Continuous, Scenario-dependent):**
    * Create dictionaries indexed by `(l, ...)` where `l` is the scenario index.
    * $A_{ij}^{k, l}, A_{jr}^{k, l}, u_{rk}^{l}$.
* **Auxiliary:**
    * $\vartheta$ (continuous, free variable, though practically bounded by max possible profit).
    * $X_{jrm}^{k, l}$ (continuous) for linearization of $\alpha \cdot A$.

### 3.2. Constraints construction
1.  **Topology Constraints:** Enforce single sourcing and mode consistency (Constraints 12-17 in the framework).
2.  **Scenario-Specific Constraints (Iterative Loop):**
    * Wrap this in a function `add_scenario_constraints(model, l, eta_l)`.
    * When adding scenario $l$, generate the specific demand $\tilde{d}_{rk}^{(l)}$ using the *fixed* $\eta$ values from that scenario.
    * **Note:** The demand depends on variables $\beta_{rm}$. Ensure the RHS of the demand equality constraint includes the term $\sum \mu DI \beta$.
3.  **The Optimality Cut (Constraint 18):**
    * $\vartheta \le \text{Revenue}^{(l)} - \text{Costs}^{(l)}$.
    * **Crucial:** Ensure Revenue is $S \times (\tilde{d}^{(l)} - u^{(l)})$. Do not use shipment quantities for revenue.

### 3.3. Linearization of MP Transportation Cost
The term $\alpha_{jrm} \cdot A_{jr}^{k(l)}$ is bilinear. Implement the Big-M constraints:
* $X \le M \cdot \alpha$
* $X \le A$
* $X \ge A - M(1 - \alpha)$
* **Parameter $M$:** Set $M = MC_j$ (Capacity of DC $j$). Do not use an arbitrary large number (e.g., $10^9$) to avoid numerical instability.

---

## 4. Subproblem (SP) Implementation
**Goal:** Given fixed strategic decisions ($\hat{x}, \hat{y}, \dots$), find the worst-case realization of uncertainty $\eta$ (specifically binary $\eta^+, \eta^-$).

### 4.1. The Dual Formulation
Since the SP is a `min-max` problem, you must implement the **Dual** of the inner maximization.
* **Objective:** Minimization (Eq. 28 in framework).
* **Decision Variables:**
    * Duals: $\pi, \sigma, \psi, \phi$ (non-negative).
    * Duals: $\gamma, \kappa$ (free).
    * Uncertainty: $\eta^+, \eta^-$ (binary).
    * Linearization aux: $p^+, p^-$ (continuous).

### 4.2. McCormick Linearization (Critical Step)
The objective contains bilinear terms $\xi_{rk} = (\eta^+ - \eta^-)\gamma_{rk}$. You must decompose this.

1.  **Bounds:** Calculate these explicitly before building the model.
    * $\gamma^U = S$ (Selling Price).
    * $\gamma^L = -(S + SC)$ (Selling Price + Shortage Cost).
2.  **Constraints:**
    * Implement the 4 McCormick envelopes for $p^+ \approx \eta^+ \gamma$.
    * Implement the 4 McCormick envelopes for $p^- \approx \eta^- \gamma$.
    * Substitute $\xi$ in the objective with $(p^+ - p^-)$.

### 4.3. Constraints
* **Dual Feasibility:** These constraints link the dual variables to the costs ($F, h, t, TC$).
    * *Caution:* The fixed first-stage variables ($\hat{\alpha}, \hat{z}, \hat{w}$) appear as constants in the Right-Hand Side (RHS) of these dual constraints.
* **Uncertainty Budget:** $\sum (\eta^+ + \eta^-) \le \Gamma_k$.

---

## 5. Main Algorithm Loop (C\&CG)

```python
LB = -infinity
UB = +infinity
epsilon = 1e-4
L = [initial_zero_scenario]

while (UB - LB) > epsilon:
    # 1. Solve Master Problem
    mp_model.optimize()
    current_strategic_sol = get_strategic_values(mp_model)
    theta_val = mp_model.getVarByName("theta").X
    
    # Update UB
    UB = min(UB, mp_model.ObjVal) # Note: If MP objective is Profit, UB is MP.ObjVal
                                  # If MP objective is Cost, signs must be inverted.
                                  # Framework defines Max Profit.

    # 2. Solve Subproblem
    # Pass current_strategic_sol to SP
    sp_model = build_sp_model(current_strategic_sol)
    sp_model.optimize()
    
    worst_case_eta = get_eta_values(sp_model)
    real_profit = calculate_profit(current_strategic_sol, worst_case_eta)
    
    # Update LB
    LB = max(LB, real_profit)
    
    # 3. Check Convergence
    gap = (UB - LB) / abs(UB) # Relative gap
    if gap <= epsilon:
        break
        
    # 4. Add Cuts
    # Add worst_case_eta to list L
    # Create new variables and constraints for this scenario in MP