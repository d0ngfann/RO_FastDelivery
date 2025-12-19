import pandas as pd
from docplex.mp.model import Model
from excel_reader import read_1d_dict, read_2d_dict, read_3d_dict
from chrome_to_dict_numpy import format_chromosome
from typing import List
import docplex.mp.environment as env

# === 2. Parameters Loading ===
parameter_data = "data/data1.xlsx"


# evaluation.py 맨 위에 추가
def set_cplex_threads(mdl, n=1):
    """mdl(도플렉스 모델)에 사용할 CPU 스레드 수 지정"""
    mdl.context.cplex_parameters.threads = n

# === Fixed cost 계산 ===
def fixed_cost(z_star: dict, w_star: dict) -> float:
    fj = read_1d_dict(parameter_data, "13", "B2:B11")
    fp = read_2d_dict(parameter_data, "9", "B2:D4")
    fd = read_1d_dict(parameter_data, "10", "B1:B10")
    fb = read_3d_dict(parameter_data, "11", "D2:D91", (3, 3, 10))
    fc = read_2d_dict(parameter_data, "12", "A2:AX11")
    """
    fp * x + (fj + fd) * y + fb * z + fc * w
    를 하나의 스칼라 값으로 반환한다.
    필요한 비용 테이블(fp, fj, fd, fb, fc)은
    evaluate() 상위 스코프(모듈 전역)에 이미 로딩돼 있다고 가정.
    """
    # 1) 집합 준비
    active_plants   = set()        # (k,i)
    active_dcs      = set()        # j

    cost = 0.0

    # 2) z_star: plant→DC 링크
    for (k, i, j), on in z_star.items():
        if on:
            active_plants.add((k, i))
            active_dcs.add(j)
            cost += fb.get((k, i, j), 0)      # 노선 고정비

    # 3) w_star: DC→customer 링크
    for (j, r), on in w_star.items():
        if on:
            active_dcs.add(j)
            cost += fc.get((j, r), 0)         # 노선 고정비

    # 4) plant 건설비
    for (k, i) in active_plants:
        cost += fp.get((k, i), 0)

    # 5) DC 건설비 (fj + fd)
    for j in active_dcs:
        cost += fj.get(j, 0) + fd.get(j, 0)

    return cost


# === 1. Index sets and Model Initialization ===
numP = 3  # Number of plants
numVD = 10  # Number of distribution centers
numR = 50  # Number of customers
Msize = 3  # Number of transport modes
ProductNum = 3  # Number of products

P = range(numP)
Vd = range(numVD)
R = range(numR)
M = range(Msize)
K = range(ProductNum)




#####################################################################
# MP: Max production at plant
MP = read_2d_dict(parameter_data,sheet_name="7",cell_range="B2:D4")
#####################################################################
# MC: DC capacity
MC = read_1d_dict(parameter_data, sheet_name="8", cell_range="B1:B10")

#####################################################################
# mu_hat: demand
df_hat = pd.read_excel(parameter_data, "5", header=None)
hat_range = df_hat.iloc[1:51, 0:3].reset_index(drop=True)
hat_range.columns = range(hat_range.shape[1])
mu_hat = {}
for r in R:
    for k in K:
        mu_hat[(r, k)] = hat_range.iloc[r, k]
#####################################################################
# mu_bar: demand
df_bar = pd.read_excel(parameter_data, "6", header=None)
bar_range = df_bar.iloc[1:51, 0:3].reset_index(drop=True)
bar_range.columns = range(bar_range.shape[1])
mu_bar = {}
for r in R:
    for k in K:
        mu_bar[(r, k)] = bar_range.iloc[r, k]
#####################################################################
# s_rk
df_srk = pd.read_excel(parameter_data, "17", header=None)
srk_range = df_srk.iloc[1:51, 0:3].reset_index(drop=True)
srk_range.columns = range(srk_range.shape[1])
s_rk = {}
for r in R:
    for k in K:
        s_rk[(r, k)] = srk_range.iloc[r, k]
#####################################################################
# TC: Transport costsss
TC = {0: 1, 1: 1.09, 2: 1.31}
#####################################################################
# DI
# 엑셀 파일에서 A1:C3 범위의 데이터를 읽어오기
df_demand = pd.read_excel(parameter_data, sheet_name='demand', usecols='A:C', header=None, nrows=3, engine='openpyxl')
# DataFrame을 리스트로 변환
data_di = df_demand.values.tolist()

# data_di의 실제 크기를 바탕으로 M과 K의 범위를 설정
num_rows = len(data_di)
num_cols = len(data_di[0]) if num_rows > 0 else 0

# 사용자께서 알려주신 m=3, k=3에 맞춰짐
M = range(num_rows)  # num_rows는 3이므로, M은 range(3) -> 0, 1, 2
K = range(num_cols)  # num_cols는 3이므로, K는 range(3) -> 0, 1, 2

# M과 K의 범위가 데이터 크기와 일치하므로 오류 없이 실행됨
DI = {(m, k): data_di[m][k] for m in M for k in K}
#####################################################################
# SC: Shortage cost
SC = {0: 750, 1: 750, 2: 750}
EC = {0: 500, 1: 500, 2: 500}
#####################################################################
# B
B = {0: 2500, 1: 2500, 2: 2500}
#####################################################################
# Gamma: Budget
Gamma_1 = {0: 7, 1: 7, 2: 7}
Gamma_2 = {0: 7, 1: 7, 2: 7}
#####################################################################
# D1: Distance factor 1
D1 = read_3d_dict(parameter_data,"2","D2:D91",(3,3,10))
#####################################################################
# D2: Distance factor 2
D2 = read_2d_dict(parameter_data,"3","A2:AX11")
#####################################################################
# b: Unit distance cost
b = 1
#####################################################################
# h: Handling cost
df_h = pd.read_excel(parameter_data, sheet_name="14", usecols="B", skiprows=1, nrows=10, header=None)
h = df_h.iloc[:, 0].to_dict()
#####################################################################
# Fq: Production cost
df_fq = pd.read_excel(parameter_data, sheet_name="15", header=None)
fq_range = df_fq.iloc[1:4, 1:4].reset_index(drop=True)
fq_range.columns = range(fq_range.shape[1])
Fq = {}
for k in K:
    for i in P:
        Fq[(k, i)] = fq_range.iloc[k, i]
#####################################################################
def evaluate(chrom: List[int], return_vars: bool = False):  # -> float 타입 힌트는 일단 유지
    z_star, w_star, beta_star, alpha_star = format_chromosome(chrom)
    # === 3. Decision variables ===
    mdl = Model(name='Supply_Chain_Optimization')
    set_cplex_threads(mdl,n=1)
    # --- Primal variables ---
    Aij = mdl.continuous_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='Aij', lb=0)
    Ajr = mdl.continuous_var_dict([(j, r, k) for j in Vd for r in R for k in K], name='Ajr', lb=0)
    u = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='u', lb=0)
    v = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='v', lb=0)
    eta = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='eta', lb=0)
    tau = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='tau', lb=0)

    # --- Dual variables (non-negative) ---
    pi = mdl.continuous_var_dict([(k, i) for k in K for i in P], name='pi', lb=0)
    sigma = mdl.continuous_var_dict(Vd, name='sigma', lb=0)
    phi = mdl.continuous_var_dict([(j, r, k) for j in Vd for r in R for k in K], name='phi', lb=0)
    psi = mdl.continuous_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='psi', lb=0)
    xi = mdl.continuous_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='xi', lb=0)
    omega = mdl.continuous_var_dict([(j, r, k) for j in Vd for r in R for k in K], name='omega', lb=0)
    rho = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='rho', lb=0)
    DD = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='DD', lb=0)

    # Free variables modeled as a difference of two non-negative variables
    kappaPlus = mdl.continuous_var_dict([(j, k) for j in Vd for k in K], name='kappaPlus', lb=0)
    kappaMinus = mdl.continuous_var_dict([(j, k) for j in Vd for k in K], name='kappaMinus', lb=0)
    gammaPlus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='gammaPlus', lb=0)
    gammaMinus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='gammaMinus', lb=0)

    # --- Binary switches for Big-M ---
    delta = mdl.binary_var_dict([(k, i) for k in K for i in P], name='delta')
    theta = mdl.binary_var_dict(Vd, name='theta')
    lambda_var = mdl.binary_var_dict([(j, r, k) for j in Vd for r in R for k in K], name='lambda')
    zeta = mdl.binary_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='zeta')
    AA = mdl.binary_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='AA')
    BB = mdl.binary_var_dict([(j, r, k) for j in Vd for r in R for k in K], name='BB')
    CC = mdl.binary_var_dict([(r, k) for r in R for k in K], name='CC')
    EE = mdl.binary_var_dict([(r, k) for r in R for k in K], name='EE')

    # === 4. Big-M Constant ===
    BigM_Dual = 1e6

    # === 목적 함수 구성 요소 정의 (이 부분이 추가/수정됩니다) ===
    # 1. 각 비용/매출 항목을 개별 표현식으로 정의합니다.
    total_revenue = mdl.sum(B[k] * Ajr[j, r, k] for j in Vd for r in R for k in K)
    transport_cost_2 = mdl.sum(
        (mdl.sum(TC[m] * alpha_star[j, r, m] for m in M) * D2[j, r]) * Ajr[j, r, k] for j in Vd for r in R for k in K)
    holding_cost = mdl.sum((h[j] / 2) * Aij[k, i, j] for k in K for i in P for j in Vd)
    transport_cost_1 = mdl.sum((D1[k, i, j] * b) * Aij[k, i, j] for k in K for i in P for j in Vd)
    production_cost = mdl.sum(Fq[k, i] * Aij[k, i, j] for k in K for i in P for j in Vd)
    shortage_cost = mdl.sum(SC[k] * u[r, k] for r in R for k in K)
    surplus_cost = mdl.sum(EC[k] * v[r, k] for r in R for k in K)  # 사용자가 추가 요청한 항목

    # 2. 원래의 목적 함수를 정의합니다.
    # 사용자의 설명에 따라 매출(total_revenue)은 양수로, 비용들은 음수로 변환하여 최대화합니다.
    mdl.maximize(
        -total_revenue + transport_cost_2 + holding_cost + transport_cost_1 + production_cost + shortage_cost + surplus_cost)

    # === 6. Constraints ===

    # --- 6.1. Primal Feasibility ---
    # (C1) Plant capacity
    for k in K:
        for i in P:
            mdl.add_constraint(mdl.sum(Aij[k, i, j] for j in Vd) <= MP[k, i], ctname=f"C1_{k}_{i}")

    # (C2) DC capacity
    for j in Vd:
        mdl.add_constraint(mdl.sum(Aij[k, i, j] for k in K for i in P) <= MC[j], ctname=f"C2_{j}")

    # (C3) DC->customer link availability
    for j in Vd:
        for r in R:
            for k in K:
                mdl.add_constraint(Ajr[j, r, k] <= MC[j] * w_star[j, r], ctname=f"C3_{j}_{r}_{k}")

    # (C4) Plant->DC link availability
    for i in P:
        for j in Vd:
            for k in K:
                mdl.add_constraint(Aij[k, i, j] <= MC[j] * z_star[k, i, j], ctname=f"C4_{i}_{j}_{k}")

    # (C5) Demand satisfaction
    for r in R:
        for k in K:
            demand_rhs = s_rk[r, k] * (mdl.sum(mu_bar[r, k] * beta_star[r, m] * DI[m, k] for m in M) + eta[r, k] * mu_hat[r, k] - tau[r,k] * mu_hat[r,k])
            mdl.add_constraint(mdl.sum(Ajr[j, r, k] for j in Vd) + u[r, k] - v[r,k] == demand_rhs, ctname=f"C5_{r}_{k}")



                    # (C6) Flow conservation at each DC
    for j in Vd:
        for k in K:
            mdl.add_constraint(mdl.sum(Aij[k, i, j] for i in P) == mdl.sum(Ajr[j, r, k] for r in R), ctname=f"C6_{j}_{k}")

    # (C7) Gamma-budget on active scenarios
    for k in K:
        mdl.add_constraint(mdl.sum(eta[r, k] for r in R) <= Gamma_1[k], ctname=f"C7.1_{k}")
        mdl.add_constraint(mdl.sum(tau[r, k] for r in R) <= Gamma_2[k], ctname=f"C7.2_{k}")

    # Variable bounds
    for r in R:
        for k in K:
            mdl.add_constraint(eta[r, k] <= s_rk[r, k], ctname=f"BOUND_eta_{r}_{k}")
            mdl.add_constraint(tau[r, k] <= s_rk[r, k], ctname=f"BOUND_tau_{r}_{k}")
            mdl.add_constraint(eta[r,k] <= 1)
            mdl.add_constraint(tau[r, k] <= 1)

    # --- 6.2. Stationarity (KKT first-order conditions) ---
    for k in K:
        for i in P:
            for j in Vd:
                mdl.add_constraint((h[j] / 2 + D1[k, i, j] * b + Fq[k, i]) + pi[k, i] + sigma[j] + psi[k, i, j]
                                   + kappaPlus[j, k] - kappaMinus[j, k] - xi[k, i, j] == 0, ctname=f"STAT_Aij_{k}_{i}_{j}")

    for j in Vd:
        for r in R:
            for k in K:
                mdl.add_constraint((-B[k] + mdl.sum(TC[m] * alpha_star[j, r, m] for m in M) * D2[j, r]) + phi[j, r, k]
                                   + gammaPlus[r, k] - gammaMinus[r, k] - kappaPlus[j, k] + kappaMinus[j, k] - omega[j, r, k] == 0, ctname=f"STAT_Ajr_{j}_{r}_{k}")

    for r in R:
        for k in K:
            mdl.add_constraint(SC[k] + gammaPlus[r, k] - gammaMinus[r, k] - rho[r, k] == 0, ctname=f"STAT_u_{r}_{k}")

    for r in R:
        for k in K:
            mdl.add_constraint(EC[k] - gammaPlus[r, k] + gammaMinus[r, k] - DD[r, k] == 0, ctname=f"STAT_v_{r}_{k}")

    # --- 6.3. Big-M Linking (Linearized Complementary Slackness) ---
    for k in K:
        for i in P:
            mdl.add_constraint(MP[k, i] - mdl.sum(Aij[k, i, j] for j in Vd) <= BigM_Dual * delta[k, i])
            mdl.add_constraint(pi[k, i] <= BigM_Dual * (1 - delta[k, i]))

    # --- OPL 수정 사항 반영 4: sigma/theta의 Big-M 제약식을 C2와 일치하도록 수정 ---
    for j in Vd:
        mdl.add_constraint(MC[j] - mdl.sum(Aij[k, i, j] for k in K for i in P) <= BigM_Dual * theta[j])
        mdl.add_constraint(sigma[j] <= BigM_Dual * (1 - theta[j]))

    for j in Vd:
        for r in R:
            for k in K:
                mdl.add_constraint(MC[j] * w_star[j, r] - Ajr[j, r, k] <= BigM_Dual * lambda_var[j, r, k])
                mdl.add_constraint(phi[j, r, k] <= BigM_Dual * (1 - lambda_var[j, r, k]))

    for k in K:
        for i in P:
            for j in Vd:
                mdl.add_constraint(MC[j] * z_star[k, i, j] - Aij[k, i, j] <= BigM_Dual * zeta[k, i, j])
                mdl.add_constraint(psi[k, i, j] <= BigM_Dual * (1 - zeta[k, i, j]))

                mdl.add_constraint(xi[k, i, j] <= BigM_Dual * AA[k, i, j])
                mdl.add_constraint(Aij[k, i, j] <= BigM_Dual * (1 - AA[k, i, j]))

    for j in Vd:
        for r in R:
            for k in K:
                mdl.add_constraint(omega[j, r, k] <= BigM_Dual * BB[j, r, k])
                mdl.add_constraint(Ajr[j, r, k] <= BigM_Dual * (1 - BB[j, r, k]))

    for r in R:
        for k in K:
            mdl.add_constraint(rho[r, k] <= BigM_Dual * CC[r, k])
            mdl.add_constraint(u[r, k] <= BigM_Dual * (1 - CC[r, k]))
            mdl.add_constraint(DD[r, k] <= BigM_Dual * EE[r, k])
            mdl.add_constraint(v[r, k] <= BigM_Dual * (1 - EE[r, k]))


    # 8. Solve
    mdl.set_time_limit(15)
    sol = mdl.solve(log_output=False)

    if sol:
        objective_value = mdl.objective_value + fixed_cost(z_star, w_star)
        if return_vars:
            var_values = {}
            for var in mdl.iter_continuous_vars():
                if var.solution_value > 1e-6:
                    var_values[var.name] = var.solution_value

            # 매출 및 비용 항목별 계산 결과를 저장
            breakdown_values = {
                "Total Revenue": sol.get_value(total_revenue),
                "Transportation Cost 1": sol.get_value(transport_cost_1),
                "Transportation Cost 2": sol.get_value(transport_cost_2),
                "Holding Cost": sol.get_value(holding_cost),
                "Production Cost": sol.get_value(production_cost),
                "Shortage Cost": sol.get_value(shortage_cost),
                "Surplus Cost": sol.get_value(surplus_cost),
                "Fixed Cost": fixed_cost(z_star, w_star)  # 고정비도 추가
            }
            return objective_value, {"variables": var_values, "breakdown": breakdown_values}
        else:
            return objective_value
    else:
        if return_vars:
            return float("inf"), None
        else:
            return float("inf")

#### 결정변수 결과확인을 위해, 해당 코드 수정함  0607_23시

# If used as script
if __name__ == '__main__':
    # example: random chromosome
    print("start")
    from random import randint
    chrom = [1] * 190
    obj = evaluate(chrom)
    print('Objective:', obj)
