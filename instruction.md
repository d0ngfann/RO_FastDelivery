# Instruction: Implementing Exact C&CG for Robust Logistics Optimization

**Objective:** Refactor the existing heuristic-based Python code (`evaluate_HD_1.py`, `Final_Best_Code.py`) into an exact **Column-and-Constraint Generation (C&CG)** algorithm using **Gurobi Optimizer**, based on the formulation in `new_approach.tex`.

**Role:** Student Researcher & Advisor
**Target Solver:** Gurobi (`gurobipy`)
**Method:** C&CG (Iterative Master Problem & Subproblem)

---

## Phase 1: Preparation & Dependency Update

The first step is to switch the solver engine. We are moving from IBM CPLEX (`docplex`) to Gurobi.

1.  **Install Gurobi:**
    Ensure you have `gurobipy` installed and a valid license (academic licenses are free).
    ```bash
    pip install gurobipy
    ```

2.  **Data Loading (Keep Existing Logic):**
    * Do **not** rewrite `excel_reader.py` or `constraint_utils_numpy_1.py`.
    * Reuse the data loading block from `evaluate_HD_1.py` (Lines 14–115).
    * **Action:** Create a new file named `ccg_main.py`. Copy the parameter loading section (reading Excel files into dictionaries `MP`, `MC`, `mu_hat`, etc.) exactly as is.
    * *Tip:* Wrap the data loading in a class or function (e.g., `def load_data(): return params`) so you don't pollute the global namespace.

---

## Phase 2: The Master Problem (MP) Construction

The Master Problem determines the strategic decisions (Facility Location & Mode Selection). It starts optimistic and gets "cuts" added to it.

**File:** `ccg_main.py` (Function: `build_master_problem`)

1.  **Initialize Model:**
    Create a `gurobipy.Model("MasterProblem")`. Set `model.ModelSense = GRB.MAXIMIZE`.

2.  **First-Stage Variables (Strategic):**
    Define these as binary variables.
    * $x_i, y_j$: Open Plant/DC.
    * $z_{kij}$: Plant-DC link.
    * $w_{jr}$: DC-Customer link.
    * $\alpha_{jrm}, \beta_{rm}$: Transportation mode selection.
    * **Crucial:** Define a continuous variable `theta` ($\vartheta$) with `lb=-GRB.INFINITY`. This represents the *worst-case second-stage profit*.

3.  **First-Stage Constraints:**
    Implement constraints (8)–(19) from your original paper (connectivity, single sourcing, mode selection) using Gurobi syntax.
    * *Example:* `model.addConstr(quicksum(w[j,r] for j in J) == 1)`

4.  **Scenario Handling (The "Column" in C&CG):**
    You need a mechanism to add variables for *each* worst-case scenario found.
    * Create a list `scenarios = []`.
    * Initially, add one "Nominal Scenario" (where demand perturbation $\eta = 0$).

5.  **Scenario Constraints (The "Cut"):**
    For *every* scenario $l$ in your list:
    * Create **Second-Stage Variables** ($A_{ij}^{(l)}, A_{jr}^{(l)}, u_r^{(l)}$) specific to this scenario $l$.
    * Add **Operational Constraints** (Capacity, Flow Balance) specific to scenario $l$.
        * *Note:* The demand in the flow balance equation must use the specific $\eta^{(l)}$ values of that scenario.
        * Ref: Eq (22)-(28) in `new_approach.tex`.
    * **The Optimality Cut:**
        Add the constraint linking `theta` to the profit of this scenario.
        $$ \vartheta \le \text{Revenue}(A^{(l)}) - \text{OpsCost}(A^{(l)}) $$
    * *Logic:* The Master Problem tries to maximize $\vartheta$, but this constraint forces $\vartheta$ to be no larger than the profit in scenario $l$. As we add more "bad" scenarios, $\vartheta$ gets pushed down.

---

## Phase 3: The Subproblem (SP) Construction

The Subproblem assumes the Master Problem's solution ($x^*, y^*, \dots$) is fixed and tries to find the demand realization $\eta$ that *minimizes* profit (finding the worst case).

**File:** `ccg_main.py` (Function: `build_subproblem`)

1.  **Formulation:**
    Use **[SP-Dual]** from `new_approach.tex`. This converts the "Min-Max" structure into a standard "Min" (Dual) problem.

2.  **Input:**
    The function must take the current values of $\hat{x}, \hat{y}, \hat{z}, \hat{w}, \hat{\beta}, \hat{\alpha}$ from the Master Problem as *constants*.

3.  **Variables:**
    * **Dual Variables:** $\pi, \sigma, \psi, \phi$ (Continuous, $\ge 0$) and $\gamma, \kappa$ (Free).
    * **Uncertainty Variables:** $\eta_{rk}$ (Binary).
        * *Refinement:* In `new_approach.tex`, uncertainty is bidirectional ($\{-1, 0, 1\}$). Split this into two binary vars: $\eta^+_{rk}$ and $\eta^-_{rk}$ where $\eta = \eta^+ - \eta^-$.
    * **Linearization Variables:** $\xi_{rk}$. This represents the bilinear term $\eta_{rk} \cdot \gamma_{rk}$.

4.  **McCormick Linearization (Critical Step):**
    You cannot multiply variable $\eta$ by variable $\gamma$ in Gurobi (it makes it quadratic/non-convex). You must use the constraints from Eq (45)-(48) in the tex file.
    * Define Big-M (sufficiently large, e.g., slightly larger than max possible shortage cost).
    * Implement the 4 constraints for each $(r,k)$ to bound $\xi_{rk}$.

5.  **Objective:**
    Minimize the Dual Objective (Eq 34 in `new_approach.tex`).
    * Note: This calculates the worst-case "opportunity cost" or "adjusted profit".

---

## Phase 4: The C&CG Iterative Algorithm

This replaces your Genetic Algorithm loop.

**File:** `ccg_main.py` (Main Execution Block)

1.  **Initialization:**
    * `LB = -infinity`
    * `UB = +infinity`
    * `epsilon = 1e-4` (convergence threshold)
    * `scenarios = [zeros]` (Start with nominal demand)

2.  **The Loop (While `UB - LB > epsilon`):**

    * **Step A: Solve Master Problem**
        * Build/Update MP with current `scenarios`.
        * Solve `MP.optimize()`.
        * Get optimal obj value -> Update `UB = MP.ObjVal`.
        * Extract solution $\hat{x}, \hat{y}, \dots$

    * **Step B: Solve Subproblem**
        * Build SP using fixed $\hat{x}, \hat{y}, \dots$
        * Solve `SP.optimize()`.
        * Get optimal obj value ($Z_{SP}$).
        * Calculate True Profit: $Z_{true} = Z_{SP} - \text{FixedCosts}(\hat{x}, \hat{y})$.
        * Update `LB = max(LB, Z_{true})`.
        * **Extract Worst-Case Demand:** Get the values of $\eta^+$ and $\eta^-$ from the solution.

    * **Step C: Convergence Check & Update**
        * Calculate Gap: $(UB - LB) / UB$.
        * Print status: `Iter: X, UB: ..., LB: ..., Gap: ...`
        * If Gap $\le$ epsilon: **BREAK**.
        * Else: Append the found $\eta$ values to `scenarios` list.
        * *Constraint Generation:* In the next loop iteration, the MP will see this new scenario and create a new set of variables $A^{(new)}$ and the constraint $\vartheta \le \text{Profit}(A^{(new)})$.

---

## Technical Implementation Checklist for User

* [ ] **Parameters:** Ensure `BigM` in the Subproblem is large enough but not so large it causes numerical instability (start with $10^7$).
* [ ] **Dual Variables:** Ensure dual variables associated with inequality constraints are non-negative ($\ge 0$).
* [ ] **Bidirectional Uncertainty:** Ensure the constraint $\eta^+_{rk} + \eta^-_{rk} \le 1$ is added so demand doesn't simultaneously go up and down.
* [ ] **Solver Params:** Set `model.setParam('OutputFlag', 0)` inside the loop to avoid spamming the console, but print the summary per iteration.
* [ ] **Feasibility:** If MP becomes infeasible, check your `Recourse` logic. Robust optimization assumes a feasible solution always exists (e.g., by allowing shortages $u_r$).

---

## Expected Outcome

1.  **Optimality:** Unlike the heuristic, this code will define a lower bound and upper bound. When they meet, you have mathematically proven the optimal solution.
2.  **Performance:** C&CG usually converges in a few dozen iterations for this size of problem, whereas GA takes thousands of generations.
3.  **Paper Value:** This replaces the "weak" heuristic section with a rigorous "Exact Decomposition Algorithm," significantly increasing acceptance chances.