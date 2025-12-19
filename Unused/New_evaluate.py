import pandas as pd
from docplex.mp.model import Model
from excel_reader import read_1d_dict, read_2d_dict, read_3d_dict
from chrome_to_dict_numpy import format_chromosome
from typing import List
import docplex.mp.environment as env

# === Parameters Loading ===
parameter_data = "data/data1.xlsx"


# evaluation.py 맨 위에 추가
def set_cplex_threads(mdl, n=1):
    """mdl(도플렉스 모델)에 사용할 CPU 스레드 수 지정"""
    mdl.context.cplex_parameters.threads = n


# [수정] 성능 개선을 위해 파라미터를 함수 외부에서 미리 로드
# === Fixed Cost Parameters ===
fj_param = read_1d_dict(parameter_data, "13", "B2:B21")
fp_param = read_2d_dict(parameter_data, "9", "B2:F4")
fd_param = read_1d_dict(parameter_data, "10", "B1:B20")
fb_param = read_3d_dict(parameter_data, "11", "D2:D301", (3, 5, 20))
fc_param = read_2d_dict(parameter_data, "12", "A2:CV21")


# === Fixed cost 계산 ===
# [수정] 미리 로드된 파라미터를 인자로 받도록 함수 시그니처 변경
def fixed_cost(z_star: dict, w_star: dict, fp, fj, fd, fb, fc) -> float:
    """
    fp * x + (fj + fd) * y + fb * z + fc * w
    를 하나의 스칼라 값으로 반환한다.
    """
    # 함수 내부에서 파일을 읽는 비효율적인 코드는 모두 제거됨
    active_plants = set()  # (k,i)
    active_dcs = set()  # j
    cost = 0.0

    # z_star: plant→DC 링크
    for (k, i, j), on in z_star.items():
        if on:
            active_plants.add((k, i))
            active_dcs.add(j)
            cost += fb.get((k, i, j), 0)  # 노선 고정비

    # w_star: DC→customer 링크
    for (j, r), on in w_star.items():
        if on:
            active_dcs.add(j)
            cost += fc.get((j, r), 0)  # 노선 고정비

    # plant 건설비
    for (k, i) in active_plants:
        cost += fp.get((k, i), 0)

    # DC 건설비 (fj + fd)
    for j in active_dcs:
        cost += fj.get(j, 0) + fd.get(j, 0)

    return cost


# === 1. Index sets and Model Initialization ===
numP = 5  # Number of plants
numVD = 20  # Number of distribution centers
numR = 100  # [수정] 10 -> 100 으로 변경 (데이터와 일치시킴)
Msize = 3  # Number of transport modes
ProductNum = 3  # Number of products

P = range(numP)
Vd = range(numVD)
R = range(numR)
M = range(Msize)
K = range(ProductNum)

#####################################################################
# MP: Max production at plant
MP = read_2d_dict(parameter_data, sheet_name="7", cell_range="B2:F4")
#####################################################################
# MC: DC capacity
MC = read_1d_dict(parameter_data, sheet_name="8", cell_range="B1:B20")

#####################################################################
# mu_hat: demand
df_hat = pd.read_excel(parameter_data, "5", header=None)
hat_range = df_hat.iloc[1:101, 0:3].reset_index(drop=True)
hat_range.columns = range(hat_range.shape[1])
mu_hat = {}
for r in R:
    for k in K:
        mu_hat[(r, k)] = hat_range.iloc[r, k]
#####################################################################
# mu_bar: demand
df_bar = pd.read_excel(parameter_data, "6", header=None)
bar_range = df_bar.iloc[1:101, 0:3].reset_index(drop=True)
bar_range.columns = range(bar_range.shape[1])
mu_bar = {}
for r in R:
    for k in K:
        mu_bar[(r, k)] = bar_range.iloc[r, k]
#####################################################################
# s_rk
df_srk = pd.read_excel(parameter_data, "17", header=None)
srk_range = df_srk.iloc[1:101, 0:3].reset_index(drop=True)
srk_range.columns = range(srk_range.shape[1])
s_rk = {}
for r in R:
    for k in K:
        s_rk[(r, k)] = srk_range.iloc[r, k]
#####################################################################
# SC: Shortage cost
SC = {0: 10000, 1: 10000, 2: 10000}
B = {0: 5000, 1: 5000, 2: 5000}  # price
factor = B[0] * (1 / 10000)
# TC: Transport cost
TC_nominal = {0: 1, 1: 1.09, 2: 1.31}
TC = {key: value * factor for key, value in TC_nominal.items()}
b = 1 * factor
#####################################################################
# DI
df_demand = pd.read_excel(parameter_data, sheet_name='demand_LD', usecols='A:C', header=None, nrows=3,
                          engine='openpyxl')
data_di = df_demand.values.tolist()
num_rows, num_cols = len(data_di), (len(data_di[0]) if len(data_di) > 0 else 0)
M, K = range(num_rows), range(num_cols)
DI = {(m, k): data_di[m][k] for m in M for k in K}
#####################################################################
# Gamma: Budget
Gamma_1 = {0: 20, 1: 20, 2: 20}
#####################################################################
# D1: Distance factor 1
D1 = read_3d_dict(parameter_data, "2", "D2:D301", (3, 5, 20))
#####################################################################
# D2: Distance factor 2
D2 = read_2d_dict(parameter_data, "3", "A2:CV21")
#####################################################################
# h: Handling cost
df_h = pd.read_excel(parameter_data, sheet_name="14", usecols="B", skiprows=1, nrows=20, header=None)
h = df_h.iloc[:, 0].to_dict()
#####################################################################
# Fq: Production cost
df_fq = pd.read_excel(parameter_data, sheet_name="15", header=None)
fq_range = df_fq.iloc[1:4, 1:6].reset_index(drop=True)
fq_range.columns = range(fq_range.shape[1])
Fq = {}
for k in K:
    for i in P:
        Fq[(k, i)] = fq_range.iloc[k, i]


#####################################################################

def evaluate(chrom: List[int], return_vars: bool = False):
    z_star, w_star, beta_star, alpha_star = format_chromosome(chrom)
    mdl = Model(name='Supply_Chain_Optimization')
    set_cplex_threads(mdl, n=1)

    # === Decision variables ===
    # [수정] eta를 Big-M 제약에 맞게 이진(binary) 변수로 변경
    eta = mdl.binary_var_dict([(r, k) for r in R for k in K], name='eta')

    # --- Dual variables (non-negative) ---
    pi = mdl.continuous_var_dict([(k, i) for k in K for i in P], name='pi', lb=0)
    sigma = mdl.continuous_var_dict(Vd, name='sigma', lb=0)
    phi = mdl.continuous_var_dict([(k, j, r) for k in K for j in Vd for r in R], name='phi', lb=0)
    psi = mdl.continuous_var_dict([(k, i, j) for k in K for i in P for j in Vd], name='psi', lb=0)

    # Free variables modeled as a difference of two non-negative variables
    kappaPlus = mdl.continuous_var_dict([(k, j) for k in K for j in Vd], name='kappaPlus', lb=0)
    kappaMinus = mdl.continuous_var_dict([(k, j) for k in K for j in Vd], name='kappaMinus', lb=0)
    gammaPlus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='gammaPlus', lb=0)
    gammaMinus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='gammaMinus', lb=0)
    tPlus = mdl.continuous_var(lb=0, name="tPlus")
    tMinus = mdl.continuous_var(lb=0, name="tMinus")
    xiPlus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='xiPlus', lb=0)
    xiMinus = mdl.continuous_var_dict([(r, k) for r in R for k in K], name='xiMinus', lb=0)

    # === Big-M Constant ===
    BigM = 1e6

    # === Objective Function ===
    mdl.minimize(tPlus - tMinus)

    # === Constraints ===
    mdl.add_constraint(tPlus - tMinus >= mdl.sum(MP[k, i] * pi[k, i] for k in K for i in P)
                       + mdl.sum(MC[j] * sigma[j] for j in Vd)
                       + mdl.sum(MC[j] * z_star[k, i, j] * psi[k, i, j] for k in K for i in P for j in Vd)
                       + mdl.sum(MC[j] * w_star[j, r] * phi[k, j, r] for k in K for j in Vd for r in R)
                       + mdl.sum(s_rk[r, k] * (mdl.sum(mu_bar[r, k] * beta_star[r, m] * DI[m, k] for m in M)) *
                                 gammaPlus[r, k] for r in R for k in K)
                       - mdl.sum(s_rk[r, k] * (mdl.sum(mu_bar[r, k] * beta_star[r, m] * DI[m, k] for m in M)) *
                                 gammaMinus[r, k] for r in R for k in K)
                       + mdl.sum(s_rk[r, k] * mu_hat[r, k] * xiPlus[r, k] for r in R for k in K)
                       - mdl.sum(s_rk[r, k] * mu_hat[r, k] * xiMinus[r, k] for r in R for k in K), ctname="about t")

    for k in K:
        for i in P:
            for j in Vd:
                mdl.add_constraint(
                    kappaPlus[k, j] - kappaMinus[k, j] + pi[k, i] + sigma[j] + psi[k, i, j] >= -h[j] / 2 - D1[
                        k, i, j] * b - Fq[k, i], ctname=f"Akij_{k}_{i}_{j}")

    for k in K:
        for j in Vd:
            for r in R:
                mdl.add_constraint(
                    phi[k, j, r] + gammaPlus[r, k] - gammaMinus[r, k] - kappaPlus[k, j] + kappaMinus[k, j] >= B[
                        k] - mdl.sum(
                        D2[j, r] * TC[m] * alpha_star[j, r, m] for m in M), ctname=f"Akjr_{k}_{j}_{r}")

    for r in R:
        for k in K:
            mdl.add_constraint(gammaPlus[r, k] - gammaMinus[r, k] >= -SC[k])

    for r in R:
        for k in K:
            mdl.add_constraint(eta[r, k] <= s_rk[r, k])

    for k in K:
        mdl.add_constraint(mdl.sum(eta[r, k] for r in R) <= Gamma_1[k])

    for r in R:
        for k in K:
            mdl.add_constraint(xiPlus[r, k] >= 0)
            mdl.add_constraint(xiPlus[r, k] >= gammaPlus[r, k] - BigM * (1 - eta[r, k]))
            mdl.add_constraint(xiPlus[r, k] <= BigM * eta[r, k])
            mdl.add_constraint(xiPlus[r, k] <= gammaPlus[r, k])

    for r in R:
        for k in K:
            mdl.add_constraint(xiMinus[r, k] >= 0)
            mdl.add_constraint(xiMinus[r, k] >= gammaMinus[r, k] - BigM * (1 - eta[r, k]))
            mdl.add_constraint(xiMinus[r, k] <= BigM * eta[r, k])
            mdl.add_constraint(xiMinus[r, k] <= gammaMinus[r, k])

    # === Solve ===
    mdl.set_time_limit(10)
    sol = mdl.solve(log_output=False)

    # === Return ===
    if sol:
        f_cost = fixed_cost(z_star, w_star, fp_param, fj_param, fd_param, fb_param, fc_param)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ [디버깅용] 중간값 확인을 위한 print문 추가 ★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        #print(f"Subproblem Value (t): {mdl.objective_value:.2f}, Fixed+Ordering Costs: {f_cost:.2f}")

        objective_value = mdl.objective_value - f_cost
        return objective_value
    else:
        return float("-inf")


# If used as script
if __name__ == '__main__':
    print("start")
    from random import randint

    # CHROM_LENGTH는 500이므로 길이를 맞춰줌
    chrom =[ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
 20,  6, 15,  6, 20, 15,
 20,  6,  6, 20, 20, 17, 20, 20, 20, 20,  6, 20, 20, 20, 20, 15,  6, 15,
  6, 20,  6, 20, 20,  6, 15, 17, 20, 17,  7,  6, 20,  6, 20,  6, 20,  6,
  6, 17,  6,  7, 17, 20, 20,  6, 20,  6, 20, 20, 20, 17, 20,  7, 20, 20,
 20, 17,  6,  6, 15, 17,  6, 17, 17,  7, 17, 17, 20, 15, 20,  7, 20,  6,
  7, 20, 20, 15,  6,  6,  6,  6, 20,  6,  7, 15, 17,  6,  6, 20,  6,  6,
 20, 20,  6,  6,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

    obj = evaluate(chrom)
    print('Objective:', obj)