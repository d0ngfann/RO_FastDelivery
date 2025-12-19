"""
DH_config.py
Configuration file for the Column-and-Constraint Generation (C&CG) algorithm.
Contains problem dimensions, parameters, and settings for the robust optimization framework.
"""

class ProblemConfig:
    """Configuration class for problem dimensions and algorithm settings."""

    def __init__(self, instance_type='full'):
        """
        Initialize problem configuration.

        Args:
            instance_type: 'toy' for small test instance, 'full' for complete problem
        """
        if instance_type == 'toy':
            # Small test instance for verification
            self.K = 1  # Products
            self.I = 2  # Plants
            self.J = 2  # DCs
            self.R = 5  # Customers
            self.M = 2  # Transport modes
            self.grid_size = 50  # Coordinate grid size
        elif instance_type == 'full':
            # Full-scale problem
            self.K = 3  # Products
            self.I = 5  # Plants
            self.J = 20  # DCs
            self.R = 100  # Customers
            self.M = 3  # Transport modes
            self.grid_size = 100  # Coordinate grid size
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        # Algorithm parameters
        self.epsilon = 1e-4  # Convergence tolerance
        self.max_iterations = 100  # Maximum C&CG iterations

        # Gurobi solver settings
        self.gurobi_time_limit = 3600  # seconds per optimization
        self.gurobi_mip_gap = 1e-6  # Tighter MIP gap (was 1e-4)
        self.gurobi_threads = 0  # 0 = use all available cores
        self.gurobi_output_flag = 1  # 1 = show logs, 0 = silent

        # Additional solver tolerances for better convergence
        self.gurobi_feasibility_tol = 1e-9  # Primal feasibility tolerance
        self.gurobi_int_feas_tol = 1e-9  # Integer feasibility tolerance
        self.gurobi_opt_tol = 1e-9  # Dual feasibility tolerance

        # Uncertainty budget (configurable for sensitivity analysis)
        # Will be set externally for each run
        self.Gamma = None

        # Data file paths
        self.data_file = f"data/DH_data_{instance_type}.pkl"
        self.results_dir = "result"

        # Output settings
        self.log_console = True
        self.save_results = True

    def set_gamma(self, gamma_value):
        """Set the uncertainty budget Γ_k (same for all products)."""
        if gamma_value < 0 or gamma_value > self.R:
            raise ValueError(f"Gamma must be in [0, {self.R}]")
        self.Gamma = gamma_value

    def __repr__(self):
        return (f"ProblemConfig(K={self.K}, I={self.I}, J={self.J}, "
                f"R={self.R}, M={self.M}, Γ={self.Gamma})")


class DataParameters:
    """
    Default parameter values for data generation.
    These are reasonable values for a supply chain optimization problem.
    """

    # Economic parameters
    S = 100.0  # Unit selling price (uniform for all products)
    SC = 50.0  # Unit shortage cost

    # Cost parameters (ranges for random generation)
    # Plant fixed costs: C^plant_{ki} - REDUCED for more realistic solutions
    C_plant_min = 5000   # was 50000
    C_plant_max = 15000  # was 150000

    # DC fixed costs: C^dc_j - REDUCED for more realistic solutions
    C_dc_min = 3000      # was 30000
    C_dc_max = 8000      # was 80000

    # Ordering costs at DCs: O_j
    O_min = 5000
    O_max = 15000

    # Route fixed costs plant-to-DC: L^1_{kij}
    L1_min = 1000
    L1_max = 5000

    # Route fixed costs DC-to-customer: L^2_{jr}
    L2_min = 500
    L2_max = 2000

    # Unit production cost: F_{ki}
    F_min = 10.0
    F_max = 30.0

    # Unit holding cost at DC: h_j
    h_min = 2.0
    h_max = 8.0

    # Unit transportation cost (plant to DC): t
    t = 0.1  # per unit distance

    # Transportation cost coefficients by mode: TC_m
    # Mode 0: slow/cheap, Mode 1: medium, Mode 2: fast/expensive
    TC_modes = [0.05, 0.10, 0.20]  # per unit distance

    # Demand increase factors by mode: DI_{mk}
    # Faster delivery → higher demand
    # Shape: (M, K) - mode × product
    DI_base = [
        [1.0, 1.0, 1.0],  # Mode 0: no increase
        [1.2, 1.2, 1.2],  # Mode 1: 20% increase
        [1.5, 1.5, 1.5]   # Mode 2: 50% increase
    ]

    # Capacity parameters
    # Plant capacity: MP_{ki}
    MP_min = 5000
    MP_max = 15000

    # DC capacity: MC_j
    MC_min = 3000
    MC_max = 10000

    # Demand parameters
    # Nominal demand: μ_{rk}
    mu_min = 10.0
    mu_max = 50.0

    # Demand deviation (maximum uncertainty): μ̂_{rk}
    # As a fraction of nominal demand
    mu_hat_factor_min = 0.2  # 20% of nominal
    mu_hat_factor_max = 0.5  # 50% of nominal

    @classmethod
    def get_TC_modes(cls, M):
        """Get transport mode costs, extended if M > 3."""
        if M <= len(cls.TC_modes):
            return cls.TC_modes[:M]
        else:
            # Extend with linear interpolation
            return cls.TC_modes + [cls.TC_modes[-1]] * (M - len(cls.TC_modes))

    @classmethod
    def get_DI_matrix(cls, M, K):
        """Get demand increase factors matrix (M × K)."""
        import numpy as np
        if M <= len(cls.DI_base) and K <= len(cls.DI_base[0]):
            # Extract the submatrix from nested list
            sublist = [cls.DI_base[m][:K] for m in range(M)]
            return np.array(sublist)
        else:
            # Create default matrix
            DI = np.ones((M, K))
            for m in range(M):
                # Higher mode → higher demand increase
                factor = 1.0 + 0.25 * m
                DI[m, :] = factor
            return DI


# Sensitivity analysis settings
class SensitivityConfig:
    """Configuration for Gamma sensitivity analysis."""

    def __init__(self, R):
        """
        Initialize sensitivity analysis configuration.

        Args:
            R: Number of customers (determines max Gamma)
        """
        self.gamma_values = list(range(0, R + 1, max(1, R // 10)))  # Test 0, R/10, 2R/10, ..., R
        self.output_file = "result/DH_sensitivity_results.csv"
        self.plot_file = "result/DH_sensitivity_plot.png"

    def __repr__(self):
        return f"SensitivityConfig(gamma_values={self.gamma_values})"
