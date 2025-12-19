"""
DH_master.py
Master Problem (MP) formulation for the Column-and-Constraint Generation algorithm.
Includes network topology constraints, operational constraints, and optimality cuts.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from DH_data_gen import SupplyChainData
from DH_config import ProblemConfig


class MasterProblem:
    """Master Problem for C&CG algorithm."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize Master Problem.

        Args:
            data: SupplyChainData instance with all parameters
            config: ProblemConfig instance
        """
        self.data = data
        self.config = config
        self.K = data.K
        self.I = data.I
        self.J = data.J
        self.R = data.R
        self.M = data.M

        # Gurobi model
        self.model = gp.Model("MasterProblem")
        self._configure_solver()

        # Decision variables
        self.x = {}  # Plant opening: x[i]
        self.y = {}  # DC opening: y[j]
        self.z = {}  # Plant-DC route: z[i,j]
        self.w = {}  # DC-Customer assignment: w[j,r]
        self.beta = {}  # Customer mode selection: beta[r,m]
        self.alpha = {}  # Mode-route combination: alpha[j,r,m]
        self.theta = None  # Worst-case operational profit

        # Second-stage variables per scenario (will be added dynamically)
        self.scenarios = []  # List of scenario data: [(scenario_id, eta_plus, eta_minus), ...]
        self.A_ij = {}  # Plant-DC flow: A_ij[k,i,j,l]
        self.A_jr = {}  # DC-Customer flow: A_jr[k,j,r,l]
        self.u = {}  # Shortage: u[r,k,l]
        self.X = {}  # Linearization variable: X[j,r,m,k,l] = alpha[j,r,m] * A_jr[k,j,r,l]

        # Build model
        self._build_variables()
        self._build_objective()
        self._build_network_constraints()

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
        """Create first-stage decision variables."""
        # Plant opening variables
        for i in range(self.I):
            self.x[i] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

        # DC opening variables
        for j in range(self.J):
            self.y[j] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

        # Plant-DC route variables
        for i in range(self.I):
            for j in range(self.J):
                self.z[(i, j)] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # DC-Customer assignment variables
        for j in range(self.J):
            for r in range(self.R):
                self.w[(j, r)] = self.model.addVar(vtype=GRB.BINARY, name=f"w_{j}_{r}")

        # Customer mode selection variables
        for r in range(self.R):
            for m in range(self.M):
                self.beta[(r, m)] = self.model.addVar(vtype=GRB.BINARY, name=f"beta_{r}_{m}")

        # Mode-route combination variables
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    self.alpha[(j, r, m)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"alpha_{j}_{r}_{m}"
                    )

        # Auxiliary variable for worst-case profit
        self.theta = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="theta"
        )

    def _build_objective(self):
        """Build objective function: max -OC - FC + theta."""
        # Ordering cost: OC = Σ_j O_j * y_j
        OC = gp.quicksum(self.data.O[j] * self.y[j] for j in range(self.J))

        # Fixed costs: FC = plant + DC + routes
        # Plant fixed costs: Σ_k Σ_i C_plant_{ki} * x_i
        plant_cost = gp.quicksum(
            self.data.C_plant[(k, i)] * self.x[i]
            for k in range(self.K)
            for i in range(self.I)
        )

        # DC fixed costs: Σ_j C_dc_j * y_j
        dc_cost = gp.quicksum(self.data.C_dc[j] * self.y[j] for j in range(self.J))

        # Route fixed costs plant-to-DC: Σ_k Σ_i Σ_j L1_{kij} * z_ij
        route1_cost = gp.quicksum(
            self.data.L1[(k, i, j)] * self.z[(i, j)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Route fixed costs DC-to-customer: Σ_j Σ_r L2_{jr} * w_jr
        route2_cost = gp.quicksum(
            self.data.L2[(j, r)] * self.w[(j, r)]
            for j in range(self.J)
            for r in range(self.R)
        )

        FC = plant_cost + dc_cost + route1_cost + route2_cost

        # Total objective: maximize -OC - FC + theta
        self.model.setObjective(-OC - FC + self.theta, GRB.MAXIMIZE)

    def _build_network_constraints(self):
        """Build network topology constraints."""
        # Single sourcing: each customer served by exactly one DC
        # Σ_j w_jr = 1, ∀r
        for r in range(self.R):
            self.model.addConstr(
                gp.quicksum(self.w[(j, r)] for j in range(self.J)) == 1,
                name=f"single_source_r{r}"
            )

        # Mode selection consistency: Σ_j alpha_jrm = w_jr, ∀j,r
        for j in range(self.J):
            for r in range(self.R):
                self.model.addConstr(
                    gp.quicksum(self.alpha[(j, r, m)] for m in range(self.M)) == self.w[(j, r)],
                    name=f"mode_consistency_j{j}_r{r}"
                )

        # Beta consistency: Σ_j alpha_jrm = beta_rm, ∀r,m
        for r in range(self.R):
            for m in range(self.M):
                self.model.addConstr(
                    gp.quicksum(self.alpha[(j, r, m)] for j in range(self.J)) == self.beta[(r, m)],
                    name=f"beta_consistency_r{r}_m{m}"
                )

        # Plant opening constraints: z_ij <= x_i, ∀i,j
        for i in range(self.I):
            for j in range(self.J):
                self.model.addConstr(
                    self.z[(i, j)] <= self.x[i],
                    name=f"plant_open_i{i}_j{j}"
                )

        # DC opening constraints for routes: z_ij <= y_j, ∀i,j
        for i in range(self.I):
            for j in range(self.J):
                self.model.addConstr(
                    self.z[(i, j)] <= self.y[j],
                    name=f"dc_open_route_i{i}_j{j}"
                )

        # DC opening constraints for customers: w_jr <= y_j, ∀j,r
        for j in range(self.J):
            for r in range(self.R):
                self.model.addConstr(
                    self.w[(j, r)] <= self.y[j],
                    name=f"dc_serve_j{j}_r{r}"
                )

    def add_scenario(self, scenario_id, eta_plus, eta_minus):
        """
        Add a new scenario with optimality cut and operational constraints.

        Args:
            scenario_id: Unique identifier for this scenario
            eta_plus: dict {(r,k): value} of eta^+ variables
            eta_minus: dict {(r,k): value} of eta^- variables
        """
        l = scenario_id
        self.scenarios.append((l, eta_plus, eta_minus))

        # Calculate realized demand for this scenario
        # d_rk^(l) = Σ_m μ_rk * DI_mk * beta_rm + (eta^+_rk - eta^-_rk) * μ̂_rk
        d_realized = {}
        for r in range(self.R):
            for k in range(self.K):
                # Endogenous nominal demand: Σ_m μ_rk * DI_mk * beta_rm
                nominal_expr = gp.quicksum(
                    self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.beta[(r, m)]
                    for m in range(self.M)
                )
                # Uncertainty deviation: (eta^+ - eta^-) * μ̂_rk
                uncertainty = (eta_plus[(r, k)] - eta_minus[(r, k)]) * self.data.mu_hat[(r, k)]
                d_realized[(r, k)] = nominal_expr + uncertainty

        # Add second-stage variables for this scenario
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.A_ij[(k, i, j, l)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"A_ij_{k}_{i}_{j}_{l}"
                    )

        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.A_jr[(k, j, r, l)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"A_jr_{k}_{j}_{r}_{l}"
                    )

        for r in range(self.R):
            for k in range(self.K):
                self.u[(r, k, l)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"u_{r}_{k}_{l}"
                )

        # Add linearization variables X_jrm^{k(l)} = alpha_jrm * A_jr^{k(l)}
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    for k in range(self.K):
                        self.X[(j, r, m, k, l)] = self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0, name=f"X_{j}_{r}_{m}_{k}_{l}"
                        )

        self.model.update()

        # Add linearization constraints for X (Big-M method)
        # X_jrm^{k(l)} = alpha_jrm * A_jr^{k(l)}
        for j in range(self.J):
            M_j = self.data.MC[j]  # Big-M = DC capacity for tightness
            for r in range(self.R):
                for m in range(self.M):
                    for k in range(self.K):
                        X_var = self.X[(j, r, m, k, l)]
                        alpha_var = self.alpha[(j, r, m)]
                        A_var = self.A_jr[(k, j, r, l)]

                        # X <= M * alpha
                        self.model.addConstr(
                            X_var <= M_j * alpha_var,
                            name=f"lin1_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )
                        # X <= A
                        self.model.addConstr(
                            X_var <= A_var,
                            name=f"lin2_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )
                        # X >= A - M(1 - alpha)
                        self.model.addConstr(
                            X_var >= A_var - M_j * (1 - alpha_var),
                            name=f"lin3_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )

        # Add operational constraints for this scenario
        # Plant capacity: Σ_j A_ij^{k(l)} <= MP_ki, ∀k,i
        for k in range(self.K):
            for i in range(self.I):
                self.model.addConstr(
                    gp.quicksum(self.A_ij[(k, i, j, l)] for j in range(self.J)) <= self.data.MP[(k, i)],
                    name=f"plant_cap_k{k}_i{i}_l{l}"
                )

        # DC capacity: Σ_k Σ_i A_ij^{k(l)} <= MC_j, ∀j
        for j in range(self.J):
            self.model.addConstr(
                gp.quicksum(
                    self.A_ij[(k, i, j, l)]
                    for k in range(self.K)
                    for i in range(self.I)
                ) <= self.data.MC[j],
                name=f"dc_cap_j{j}_l{l}"
            )

        # Route activation plant-to-DC: A_ij^{k(l)} <= MC_j * z_ij, ∀k,i,j
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.model.addConstr(
                        self.A_ij[(k, i, j, l)] <= self.data.MC[j] * self.z[(i, j)],
                        name=f"route_ij_k{k}_i{i}_j{j}_l{l}"
                    )

        # Route activation DC-to-customer: A_jr^{k(l)} <= MC_j * w_jr, ∀k,j,r
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.model.addConstr(
                        self.A_jr[(k, j, r, l)] <= self.data.MC[j] * self.w[(j, r)],
                        name=f"route_jr_k{k}_j{j}_r{r}_l{l}"
                    )

        # Demand satisfaction: Σ_j A_jr^{k(l)} + u_rk^{(l)} = d_rk^{(l)}, ∀k,r
        for k in range(self.K):
            for r in range(self.R):
                self.model.addConstr(
                    gp.quicksum(self.A_jr[(k, j, r, l)] for j in range(self.J)) + self.u[(r, k, l)]
                    == d_realized[(r, k)],
                    name=f"demand_k{k}_r{r}_l{l}"
                )

        # Flow balance at DC: Σ_i A_ij^{k(l)} = Σ_r A_jr^{k(l)}, ∀k,j
        for k in range(self.K):
            for j in range(self.J):
                self.model.addConstr(
                    gp.quicksum(self.A_ij[(k, i, j, l)] for i in range(self.I))
                    == gp.quicksum(self.A_jr[(k, j, r, l)] for r in range(self.R)),
                    name=f"balance_k{k}_j{j}_l{l}"
                )

        # Add optimality cut: theta <= Revenue - HC - TC - PC - SC
        # Revenue = Σ_r Σ_k S * (d_rk - u_rk)
        revenue = gp.quicksum(
            self.data.S * (d_realized[(r, k)] - self.u[(r, k, l)])
            for r in range(self.R)
            for k in range(self.K)
        )

        # Holding cost: HC = Σ_k Σ_i Σ_j (h_j/2) * A_ij^{k(l)}
        HC = gp.quicksum(
            (self.data.h[j] / 2) * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Transportation cost (plant-to-DC): Σ_k Σ_i Σ_j D1_kij * t * A_ij^{k(l)}
        TC1 = gp.quicksum(
            self.data.D1[(k, i, j)] * self.data.t * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Transportation cost (DC-to-customer): Σ_k Σ_j Σ_r Σ_m D2_jr * TC_m * X_jrm^{k(l)}
        TC2 = gp.quicksum(
            self.data.D2[(j, r)] * self.data.TC[m] * self.X[(j, r, m, k, l)]
            for k in range(self.K)
            for j in range(self.J)
            for r in range(self.R)
            for m in range(self.M)
        )

        TC = TC1 + TC2

        # Production cost: PC = Σ_k Σ_i Σ_j F_ki * A_ij^{k(l)}
        PC = gp.quicksum(
            self.data.F[(k, i)] * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Shortage cost: SC = Σ_r Σ_k SC * u_rk^{(l)}
        SC = gp.quicksum(
            self.data.SC * self.u[(r, k, l)]
            for r in range(self.R)
            for k in range(self.K)
        )

        # Add optimality cut
        self.model.addConstr(
            self.theta <= revenue - HC - TC - PC - SC,
            name=f"opt_cut_l{l}"
        )

        self.model.update()

    def solve(self):
        """Solve the master problem."""
        self.model.optimize()
        return self.model.Status == GRB.OPTIMAL

    def get_solution(self):
        """
        Extract solution from the master problem.

        Returns:
            dict with solution values
        """
        if self.model.Status != GRB.OPTIMAL:
            return None

        solution = {
            'objective': self.model.ObjVal,
            'theta': self.theta.X,
            'x': {i: self.x[i].X for i in range(self.I)},
            'y': {j: self.y[j].X for j in range(self.J)},
            'z': {(i, j): self.z[(i, j)].X for i in range(self.I) for j in range(self.J)},
            'w': {(j, r): self.w[(j, r)].X for j in range(self.J) for r in range(self.R)},
            'beta': {(r, m): self.beta[(r, m)].X for r in range(self.R) for m in range(self.M)},
            'alpha': {(j, r, m): self.alpha[(j, r, m)].X
                     for j in range(self.J) for r in range(self.R) for m in range(self.M)}
        }

        return solution
