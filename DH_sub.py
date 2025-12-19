"""
DH_sub.py
Subproblem (SP-Dual) formulation for identifying worst-case scenarios.
Includes dual formulation, uncertainty set constraints, and McCormick linearization.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from DH_data_gen import SupplyChainData
from DH_config import ProblemConfig


class Subproblem:
    """Subproblem for identifying worst-case demand scenarios."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize Subproblem.

        Args:
            data: SupplyChainData instance
            config: ProblemConfig instance with Gamma set
        """
        self.data = data
        self.config = config
        self.K = data.K
        self.I = data.I
        self.J = data.J
        self.R = data.R
        self.M = data.M

        if config.Gamma is None:
            raise ValueError("Gamma must be set in config before creating Subproblem")
        self.Gamma = config.Gamma

        # Gurobi model
        self.model = gp.Model("Subproblem")
        self._configure_solver()

        # Decision variables
        # Dual variables
        self.pi = {}  # Plant capacity dual: pi[k,i]
        self.sigma = {}  # DC capacity dual: sigma[j]
        self.psi = {}  # Route plant-DC dual: psi[k,i,j]
        self.phi = {}  # Route DC-customer dual: phi[k,j,r]
        self.gamma = {}  # Demand satisfaction dual: gamma[r,k]
        self.kappa = {}  # Flow balance dual: kappa[k,j]

        # Uncertainty variables
        self.eta_plus = {}  # Demand increase: eta_plus[r,k]
        self.eta_minus = {}  # Demand decrease: eta_minus[r,k]

        # McCormick linearization variables
        self.p_plus = {}  # p_plus[r,k] = eta_plus[r,k] * gamma[r,k]
        self.p_minus = {}  # p_minus[r,k] = eta_minus[r,k] * gamma[r,k]
        # xi[r,k] = p_plus[r,k] - p_minus[r,k]

        # Fixed first-stage solution (will be set before solving)
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_z = None
        self.fixed_w = None
        self.fixed_beta = None
        self.fixed_alpha = None

        # Build model
        self._build_variables()
        self._build_constraints()
        # Objective will be built after fixing first-stage variables

        self.model.update()

    def _configure_solver(self):
        """Configure Gurobi solver parameters."""
        self.model.setParam('TimeLimit', self.config.gurobi_time_limit)
        self.model.setParam('MIPGap', self.config.gurobi_mip_gap)
        self.model.setParam('Threads', self.config.gurobi_threads)
        self.model.setParam('OutputFlag', self.config.gurobi_output_flag)
        self.model.setParam('FeasibilityTol', self.config.gurobi_feasibility_tol)
        self.model.setParam('IntFeasTol', self.config.gurobi_int_feas_tol)
        self.model.setParam('OptimalityTol', self.config.gurobi_opt_tol)

    def _build_variables(self):
        """Create decision variables."""
        # Dual variables
        # pi[k,i]: Plant capacity dual (≥ 0)
        for k in range(self.K):
            for i in range(self.I):
                self.pi[(k, i)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"pi_{k}_{i}"
                )

        # sigma[j]: DC capacity dual (≥ 0)
        for j in range(self.J):
            self.sigma[j] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"sigma_{j}"
            )

        # psi[k,i,j]: Route plant-DC dual (≥ 0)
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.psi[(k, i, j)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"psi_{k}_{i}_{j}"
                    )

        # phi[k,j,r]: Route DC-customer dual (≥ 0)
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.phi[(k, j, r)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"phi_{k}_{j}_{r}"
                    )

        # gamma[r,k]: Demand satisfaction dual (unrestricted, but bounded)
        # Bounds: gamma^L = -(S + SC), gamma^U = S
        gamma_L = -(self.data.S + self.data.SC)
        gamma_U = self.data.S
        for r in range(self.R):
            for k in range(self.K):
                self.gamma[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"gamma_{r}_{k}"
                )

        # kappa[k,j]: Flow balance dual (unrestricted)
        for k in range(self.K):
            for j in range(self.J):
                self.kappa[(k, j)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"kappa_{k}_{j}"
                )

        # Uncertainty variables (binary)
        for r in range(self.R):
            for k in range(self.K):
                self.eta_plus[(r, k)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"eta_plus_{r}_{k}"
                )
                self.eta_minus[(r, k)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"eta_minus_{r}_{k}"
                )

        # McCormick linearization variables
        # Bounds for p: based on gamma bounds and binary eta
        # p_plus, p_minus ∈ [gamma_L, gamma_U] when eta = 1, else 0
        for r in range(self.R):
            for k in range(self.K):
                self.p_plus[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"p_plus_{r}_{k}"
                )
                self.p_minus[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"p_minus_{r}_{k}"
                )

    def _build_constraints(self):
        """Build constraints (before fixing first-stage variables)."""
        # Uncertainty set constraints
        # Budget constraint: Σ_r (eta_plus + eta_minus) <= Gamma, ∀k
        for k in range(self.K):
            self.model.addConstr(
                gp.quicksum(
                    self.eta_plus[(r, k)] + self.eta_minus[(r, k)]
                    for r in range(self.R)
                ) <= self.Gamma,
                name=f"budget_k{k}"
            )

        # Mutual exclusivity: eta_plus + eta_minus <= 1, ∀r,k
        for r in range(self.R):
            for k in range(self.K):
                self.model.addConstr(
                    self.eta_plus[(r, k)] + self.eta_minus[(r, k)] <= 1,
                    name=f"mutex_r{r}_k{k}"
                )

        # McCormick linearization for p_plus = eta_plus * gamma
        gamma_L = -(self.data.S + self.data.SC)
        gamma_U = self.data.S

        for r in range(self.R):
            for k in range(self.K):
                eta_p = self.eta_plus[(r, k)]
                gamma_var = self.gamma[(r, k)]
                p_p = self.p_plus[(r, k)]

                # p_plus >= gamma_L * eta_plus
                self.model.addConstr(
                    p_p >= gamma_L * eta_p,
                    name=f"mc_pp1_r{r}_k{k}"
                )
                # p_plus <= gamma_U * eta_plus
                self.model.addConstr(
                    p_p <= gamma_U * eta_p,
                    name=f"mc_pp2_r{r}_k{k}"
                )
                # p_plus >= gamma - gamma_U * (1 - eta_plus)
                self.model.addConstr(
                    p_p >= gamma_var - gamma_U * (1 - eta_p),
                    name=f"mc_pp3_r{r}_k{k}"
                )
                # p_plus <= gamma - gamma_L * (1 - eta_plus)
                self.model.addConstr(
                    p_p <= gamma_var - gamma_L * (1 - eta_p),
                    name=f"mc_pp4_r{r}_k{k}"
                )

        # McCormick linearization for p_minus = eta_minus * gamma
        for r in range(self.R):
            for k in range(self.K):
                eta_m = self.eta_minus[(r, k)]
                gamma_var = self.gamma[(r, k)]
                p_m = self.p_minus[(r, k)]

                # p_minus >= gamma_L * eta_minus
                self.model.addConstr(
                    p_m >= gamma_L * eta_m,
                    name=f"mc_pm1_r{r}_k{k}"
                )
                # p_minus <= gamma_U * eta_minus
                self.model.addConstr(
                    p_m <= gamma_U * eta_m,
                    name=f"mc_pm2_r{r}_k{k}"
                )
                # p_minus >= gamma - gamma_U * (1 - eta_minus)
                self.model.addConstr(
                    p_m >= gamma_var - gamma_U * (1 - eta_m),
                    name=f"mc_pm3_r{r}_k{k}"
                )
                # p_minus <= gamma - gamma_L * (1 - eta_minus)
                self.model.addConstr(
                    p_m <= gamma_var - gamma_L * (1 - eta_m),
                    name=f"mc_pm4_r{r}_k{k}"
                )

    def fix_first_stage(self, solution):
        """
        Fix first-stage variables from Master Problem solution.

        Args:
            solution: dict with keys 'x', 'y', 'z', 'w', 'beta', 'alpha'
        """
        self.fixed_x = solution['x']
        self.fixed_y = solution['y']
        self.fixed_z = solution['z']
        self.fixed_w = solution['w']
        self.fixed_beta = solution['beta']
        self.fixed_alpha = solution['alpha']

        # Now add dual feasibility constraints and objective
        self._build_dual_feasibility()
        self._build_objective()
        self.model.update()

    def _build_dual_feasibility(self):
        """Build dual feasibility constraints with fixed first-stage variables."""
        # Dual feasibility for A_ij^k:
        # pi[k,i] + sigma[j] + psi[k,i,j] + kappa[k,j] >= -h_j/2 - D1[k,i,j]*t - F[k,i]
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    rhs = -(self.data.h[j] / 2) - self.data.D1[(k, i, j)] * self.data.t - self.data.F[(k, i)]
                    self.model.addConstr(
                        self.pi[(k, i)] + self.sigma[j] + self.psi[(k, i, j)] + self.kappa[(k, j)] >= rhs,
                        name=f"dual_Aij_k{k}_i{i}_j{j}"
                    )

        # Dual feasibility for A_jr^k:
        # phi[k,j,r] + gamma[r,k] - kappa[k,j] >= -Σ_m D2[j,r] * TC[m] * alpha[j,r,m]
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    rhs = -sum(
                        self.data.D2[(j, r)] * self.data.TC[m] * self.fixed_alpha[(j, r, m)]
                        for m in range(self.M)
                    )
                    self.model.addConstr(
                        self.phi[(k, j, r)] + self.gamma[(r, k)] - self.kappa[(k, j)] >= rhs,
                        name=f"dual_Ajr_k{k}_j{j}_r{r}"
                    )

        # Dual feasibility for u_rk:
        # gamma[r,k] >= -(S + SC)
        # This is already enforced by variable bounds, but we can add explicit constraint for clarity
        for r in range(self.R):
            for k in range(self.K):
                self.model.addConstr(
                    self.gamma[(r, k)] >= -(self.data.S + self.data.SC),
                    name=f"dual_u_r{r}_k{k}"
                )

    def _build_objective(self):
        """Build objective function (dual objective to minimize)."""
        # Dual objective: minimize
        # Σ_k Σ_i MP[k,i] * pi[k,i]
        # + Σ_j MC[j] * sigma[j]
        # + Σ_k Σ_i Σ_j MC[j] * z_ij * psi[k,i,j]
        # + Σ_k Σ_j Σ_r MC[j] * w_jr * phi[k,j,r]
        # + Σ_r Σ_k (Σ_m mu[r,k] * DI[m,k] * beta[r,m]) * gamma[r,k]
        # + Σ_r Σ_k mu_hat[r,k] * (p_plus[r,k] - p_minus[r,k])

        obj = 0

        # Plant capacity term
        obj += gp.quicksum(
            self.data.MP[(k, i)] * self.pi[(k, i)]
            for k in range(self.K)
            for i in range(self.I)
        )

        # DC capacity term
        obj += gp.quicksum(
            self.data.MC[j] * self.sigma[j]
            for j in range(self.J)
        )

        # Route plant-DC term
        obj += gp.quicksum(
            self.data.MC[j] * self.fixed_z[(i, j)] * self.psi[(k, i, j)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Route DC-customer term
        obj += gp.quicksum(
            self.data.MC[j] * self.fixed_w[(j, r)] * self.phi[(k, j, r)]
            for k in range(self.K)
            for j in range(self.J)
            for r in range(self.R)
        )

        # Nominal demand term
        obj += gp.quicksum(
            sum(self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.fixed_beta[(r, m)]
                for m in range(self.M)) * self.gamma[(r, k)]
            for r in range(self.R)
            for k in range(self.K)
        )

        # Uncertainty term (linearized)
        obj += gp.quicksum(
            self.data.mu_hat[(r, k)] * (self.p_plus[(r, k)] - self.p_minus[(r, k)])
            for r in range(self.R)
            for k in range(self.K)
        )

        self.model.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        """Solve the subproblem."""
        self.model.optimize()
        return self.model.Status == GRB.OPTIMAL

    def get_worst_case_scenario(self):
        """
        Extract worst-case scenario from subproblem solution.

        Returns:
            tuple: (Z_SP, eta_plus, eta_minus)
                Z_SP: Worst-case operational profit (includes revenue term)
                eta_plus: dict {(r,k): value}
                eta_minus: dict {(r,k): value}
        """
        if self.model.Status != GRB.OPTIMAL:
            return None, None, None

        # Get dual objective value
        dual_obj = self.model.ObjVal

        # Extract eta values
        eta_plus = {}
        eta_minus = {}
        for r in range(self.R):
            for k in range(self.K):
                eta_plus[(r, k)] = round(self.eta_plus[(r, k)].X)
                eta_minus[(r, k)] = round(self.eta_minus[(r, k)].X)

        # CRITICAL FIX: Add revenue term back
        # Revenue = S × Σ_{r,k} d_{rk} where d_{rk} = Σ_m μ DI β + (η+ - η-) μ̂
        # This term is constant in the dual (since β, η are fixed) so it's not in dual objective
        # We must add it back to get the true operational profit
        revenue = 0.0
        for r in range(self.R):
            for k in range(self.K):
                # Calculate realized demand for this scenario
                nominal_demand = sum(
                    self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.fixed_beta[(r, m)]
                    for m in range(self.M)
                )
                uncertainty = (eta_plus[(r, k)] - eta_minus[(r, k)]) * self.data.mu_hat[(r, k)]
                d_realized = nominal_demand + uncertainty

                # Revenue from this customer-product pair
                revenue += self.data.S * d_realized

        # True operational profit = Revenue (constant) + Dual objective
        Z_SP = revenue + dual_obj

        return Z_SP, eta_plus, eta_minus

    def get_dual_solution(self):
        """Get dual variable values (for debugging/analysis)."""
        if self.model.Status != GRB.OPTIMAL:
            return None

        dual_sol = {
            'pi': {(k, i): self.pi[(k, i)].X for k in range(self.K) for i in range(self.I)},
            'sigma': {j: self.sigma[j].X for j in range(self.J)},
            'psi': {(k, i, j): self.psi[(k, i, j)].X
                    for k in range(self.K) for i in range(self.I) for j in range(self.J)},
            'phi': {(k, j, r): self.phi[(k, j, r)].X
                    for k in range(self.K) for j in range(self.J) for r in range(self.R)},
            'gamma': {(r, k): self.gamma[(r, k)].X for r in range(self.R) for k in range(self.K)},
            'kappa': {(k, j): self.kappa[(k, j)].X for k in range(self.K) for j in range(self.J)}
        }

        return dual_sol
