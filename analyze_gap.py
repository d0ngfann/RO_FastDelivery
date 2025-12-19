"""Quick analysis of why the gap exists."""
from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_master import MasterProblem
from DH_sub import Subproblem

# Setup
config = ProblemConfig('toy')
config.set_gamma(0)
data = SupplyChainData.load('data/DH_data_toy.pkl')

print("="*80)
print("ANALYZING THE GAP")
print("="*80)

# Create Master with nominal scenario
master = MasterProblem(data, config)
eta_plus = {(r,k): 0 for r in range(config.R) for k in range(config.K)}
eta_minus = {(r,k): 0 for r in range(config.R) for k in range(config.K)}
master.add_scenario(0, eta_plus, eta_minus)

# Solve Master
print("\n1. Solving Master Problem...")
master.solve()
sol = master.get_solution()

print(f"   Master Objective: {sol['objective']:.2f}")
print(f"   Theta (θ): {sol['theta']:.2f}")

# Calculate strategic costs
OC = sum(data.O[j] * sol['y'][j] for j in range(config.J))
FC = (sum(data.C_plant[(k,i)] * sol['x'][i] for k in range(config.K) for i in range(config.I)) +
      sum(data.C_dc[j] * sol['y'][j] for j in range(config.J)) +
      sum(data.L1[(k,i,j)] * sol['z'][(i,j)] for k in range(config.K) for i in range(config.I) for j in range(config.J)) +
      sum(data.L2[(j,r)] * sol['w'][(j,r)] for j in range(config.J) for r in range(config.R)))

print(f"   Strategic costs (-OC-FC): {-OC-FC:.2f}")
print(f"   Therefore: Objective = -OC - FC + θ")
print(f"              {sol['objective']:.2f} = {-OC-FC:.2f} + {sol['theta']:.2f} ✓")

# Now solve Subproblem with same first-stage solution
print("\n2. Solving Subproblem with Master's solution...")
subproblem = Subproblem(data, config)
subproblem.fix_first_stage(sol)
subproblem.solve()
Z_SP, eta_p, eta_m = subproblem.get_worst_case_scenario()

print(f"   Subproblem Z_SP: {Z_SP:.2f}")
print(f"   True robust profit: -OC - FC + Z_SP = {-OC-FC:.2f} + {Z_SP:.2f} = {-OC-FC+Z_SP:.2f}")

# The gap
print("\n3. THE PROBLEM:")
print(f"   Master thinks θ = {sol['theta']:.2f}")
print(f"   But Subproblem shows worst-case operational profit = {Z_SP:.2f}")
print(f"   Difference: {sol['theta']:.2f} - {Z_SP:.2f} = {sol['theta'] - Z_SP:.2f}")
print()
print(f"   This means the Master's optimality cut is NOT properly constraining θ!")

# Let's check the realized demand calculation
print("\n4. CHECKING REALIZED DEMAND...")
for r in range(min(3, config.R)):  # Just first 3 customers
    for k in range(config.K):
        # Calculate using Master's β values
        nominal = sum(data.mu[(r,k)] * data.DI[(m,k)] * sol['beta'][(r,m)] 
                     for m in range(config.M))
        uncertainty = (eta_plus[(r,k)] - eta_minus[(r,k)]) * data.mu_hat[(r,k)]
        d_realized = nominal + uncertainty
        
        print(f"   Customer {r}, Product {k}:")
        print(f"      s_rk = {data.s_rk[(r,k)]}")
        print(f"      μ = {data.mu[(r,k)]:.2f}")
        print(f"      Realized demand = {d_realized:.2f}")
        print(f"      β values: {[sol['beta'][(r,m)] for m in range(config.M)]}")

# Check second-stage variables
print("\n5. CHECKING SECOND-STAGE VARIABLES FROM MASTER...")
print(f"   Number of A_ij variables: {len([v for k, v in master.A_ij.items() if k[3] == 0 and v.X > 1e-6])}")
print(f"   Number of A_jr variables: {len([v for k, v in master.A_jr.items() if k[3] == 0 and v.X > 1e-6])}")

total_production = sum(master.A_ij[(k,i,j,0)].X 
                      for k in range(config.K) 
                      for i in range(config.I) 
                      for j in range(config.J))
total_shortage = sum(master.u[(r,k,0)].X 
                    for r in range(config.R) 
                    for k in range(config.K))

print(f"   Total production: {total_production:.2f}")
print(f"   Total shortage: {total_shortage:.2f}")

# Calculate operational profit from Master's second-stage solution
revenue_master = sum(data.S * (
    (sum(data.mu[(r,k)] * data.DI[(m,k)] * sol['beta'][(r,m)] for m in range(config.M)) + 
     (eta_plus[(r,k)] - eta_minus[(r,k)]) * data.mu_hat[(r,k)]) - 
    master.u[(r,k,0)].X)
    for r in range(config.R) for k in range(config.K))

HC_master = sum((data.h[j]/2) * master.A_ij[(k,i,j,0)].X
               for k in range(config.K) for i in range(config.I) for j in range(config.J))

TC1_master = sum(data.D1[(k,i,j)] * data.t * master.A_ij[(k,i,j,0)].X
                for k in range(config.K) for i in range(config.I) for j in range(config.J))

TC2_master = sum(data.D2[(j,r)] * data.TC[m] * master.X[(j,r,m,k,0)].X
                for k in range(config.K) for j in range(config.J) 
                for r in range(config.R) for m in range(config.M))

PC_master = sum(data.F[(k,i)] * master.A_ij[(k,i,j,0)].X
               for k in range(config.K) for i in range(config.I) for j in range(config.J))

SC_master = sum(data.SC * master.u[(r,k,0)].X
               for r in range(config.R) for k in range(config.K))

op_profit_master = revenue_master - HC_master - (TC1_master + TC2_master) - PC_master - SC_master

print("\n6. OPERATIONAL PROFIT CALCULATION (from Master's 2nd-stage):")
print(f"   Revenue: {revenue_master:.2f}")
print(f"   - HC: {HC_master:.2f}")
print(f"   - TC: {TC1_master + TC2_master:.2f} (TC1: {TC1_master:.2f}, TC2: {TC2_master:.2f})")
print(f"   - PC: {PC_master:.2f}")
print(f"   - SC: {SC_master:.2f}")
print(f"   = Operational Profit: {op_profit_master:.2f}")

print("\n7. COMPARISON:")
print(f"   Master's θ (from solver): {sol['theta']:.2f}")
print(f"   Master's op profit (calculated): {op_profit_master:.2f}")
print(f"   Subproblem's Z_SP: {Z_SP:.2f}")
print(f"   Difference (θ - calculated): {sol['theta'] - op_profit_master:.6f} ← should be ~0")
print(f"   Difference (calculated - Z_SP): {op_profit_master - Z_SP:.6f} ← should be ~0")

print("\n" + "="*80)
print("CONCLUSION:")
if abs(sol['theta'] - op_profit_master) > 0.1:
    print("❌ Master's θ doesn't match calculated operational profit!")
    print("   → Optimality cut is NOT working correctly")
elif abs(op_profit_master - Z_SP) > 0.1:
    print("❌ Master's operational profit doesn't match Subproblem!")
    print("   → Problem with dual formulation or constraint evaluation")
else:
    print("✓ All values match - problem elsewhere")
print("="*80)
