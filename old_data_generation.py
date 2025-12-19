"""
data_generation.py

이 스크립트는 Colab 노트북(Data generation.ipynb)에 있는
데이터 생성 및 엑셀 저장 로직을 하나의 .py 파일로 정리한 것이다.

구성:
1. r–k 매핑 행렬 s_rk 생성
2. plant 후보 좌표 생성
3. customer 위치 생성 (Gaussian)
4. DC 후보 좌표 생성 (도넛 영역)
5. 수요 관련 파라미터(mu, mu_hat)
6. 거리 행렬 D_kij, D_jr 계산
7. 각종 비용/파라미터 시트 생성
8. 모든 결과를 data.xlsx로 저장
"""

import numpy as np
import pandas as pd
import random
from math import hypot

# ===============================
# 0. 공통 설정
# ===============================

# 재현성 확보
np.random.seed(252432342)
random.seed(252432342)

# 좌표 범위
GRID_MIN = 0
GRID_MAX = 1000

# 중앙 제외 영역 (도넛)
EXCLUDE_MIN = 200
EXCLUDE_MAX = 800


# ===============================
# 1. s_rk 생성 (r × k binary)
# ===============================

NUM_R = 50   # 고객 수
NUM_K = 3    # 제품/플랜트 그룹 수

# 0/1 랜덤 행렬 생성
matrix = np.random.randint(0, 2, size=(NUM_R, NUM_K))

# 각 행에 최소 하나의 1이 있도록 보정
for i in range(NUM_R):
    if np.all(matrix[i] == 0):
        matrix[i, np.random.randint(0, NUM_K)] = 1

s_rk = pd.DataFrame(
    matrix,
    index=[f"r{i+1}" for i in range(NUM_R)],
    columns=[f"k{j+1}" for j in range(NUM_K)]
)


# ===============================
# 2. plant 후보 좌표 생성
# ===============================

# 도넛 영역에서 가능한 모든 좌표
all_points = [
    (x, y)
    for x in range(GRID_MIN, GRID_MAX + 1)
    for y in range(GRID_MIN, GRID_MAX + 1)
    if not (EXCLUDE_MIN <= x < EXCLUDE_MAX and EXCLUDE_MIN <= y < EXCLUDE_MAX)
]

# k1, k2, k3 각각 3개씩 → 총 9개
selected_points = random.sample(all_points, 9)
random.shuffle(selected_points)

plant_candidates = pd.DataFrame(
    [
        selected_points[0:3],
        selected_points[3:6],
        selected_points[6:9]
    ],
    index=["k1", "k2", "k3"]
)


# ===============================
# 3. customer 위치 생성
# ===============================

# 2D Gaussian (mean=500, std=200)
x_coords = np.random.normal(500, 200, NUM_R)
y_coords = np.random.normal(500, 200, NUM_R)

# 정수화 + 범위 제한
x_int = np.clip(np.round(x_coords), GRID_MIN, GRID_MAX).astype(int)
y_int = np.clip(np.round(y_coords), GRID_MIN, GRID_MAX).astype(int)

customer_location = pd.DataFrame(
    {"point": list(zip(x_int, y_int))},
    index=[f"r{i+1}" for i in range(NUM_R)]
)


# ===============================
# 4. DC 후보 생성
# ===============================

NUM_DC = 10

dc_points = random.sample(all_points, NUM_DC)

dc_candidates = pd.DataFrame(
    {"point": dc_points},
    index=[f"dc{i+1}" for i in range(NUM_DC)]
)


# ===============================
# 5. 수요 파라미터 (mu, mu_hat)
# ===============================

# mu_k ~ 10 * U[1,5]
np.random.seed(439343)
mu_matrix = 10 * np.random.uniform(1, 5, size=(NUM_R, NUM_K))
mu_df = pd.DataFrame(mu_matrix, columns=["product1", "product2", "product3"])

# mu_hat = min(mu, U[4,10])
np.random.seed(74534)
U_matrix = np.random.uniform(4, 10, size=(NUM_R, NUM_K))
mu_hat_df = pd.DataFrame(
    np.minimum(mu_matrix, U_matrix),
    columns=["product1", "product2", "product3"]
)


# ===============================
# 6. D_kij 계산 (plant → DC)
# ===============================

plant_index = pd.MultiIndex.from_product(
    [["k1", "k2", "k3"], range(3)],
    names=["k", "i"]
)

D_ij = pd.DataFrame(index=plant_index, columns=dc_candidates.index, dtype=float)

for k, i in plant_index:
    px, py = plant_candidates.loc[k, i]
    for j in dc_candidates.index:
        dx, dy = dc_candidates.loc[j, "point"]
        D_ij.loc[(k, i), j] = hypot(px - dx, py - dy)

# long format (k, i, j, D_kij)
flat_df = (
    D_ij
    .reset_index()
    .melt(id_vars=["k", "i"], var_name="j", value_name="D_kij")
)

flat_df["j_num"] = flat_df["j"].str.extract(r"(\d+)").astype(int)
flat_df = flat_df.sort_values(by=["k", "i", "j_num"])
flat_df = flat_df[["k", "i", "j", "D_kij"]]


# ===============================
# 7. D_jr 계산 (DC → customer)
# ===============================

cust_points = np.vstack(customer_location["point"].values)
dc_points = np.vstack(dc_candidates["point"].values)

diffs = dc_points[:, None, :] - cust_points[None, :, :]
dist_mat = np.sqrt(np.sum(diffs ** 2, axis=2))

D_jr = pd.DataFrame(
    dist_mat,
    index=dc_candidates.index,
    columns=customer_location.index
)


# ===============================
# 8. 기타 비용/파라미터 시트 예시
# ===============================

# 예: DC 고정비
np.random.seed(200112312)
dc_construction_cost = pd.DataFrame(
    30 * (1 + np.random.uniform(-0.5, 0.5, size=(NUM_DC, 1))),
    index=dc_candidates.index
)

# 예: ij 경로 고정비
np.random.seed(25)
sheet_11_df = pd.DataFrame(
    flat_df["D_kij"] * (1 + (np.random.rand(len(flat_df)) - 0.5)) * 100
)


# ===============================
# 9. 엑셀 저장
# ===============================

file_name = "data.xlsx"

with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
    s_rk.to_excel(writer, sheet_name="17", startrow=1, header=False, index=False)
    plant_candidates.to_excel(writer, sheet_name="plant", header=False, index=False)
    customer_location.to_excel(writer, sheet_name="customer", header=False, index=False)
    dc_candidates.to_excel(writer, sheet_name="dc", header=False, index=False)

    mu_df.to_excel(writer, sheet_name="6", startrow=1, header=False, index=False)
    mu_hat_df.to_excel(writer, sheet_name="5", startrow=1, header=False, index=False)

    flat_df.to_excel(writer, sheet_name="2", index=False)
    D_jr.to_excel(writer, sheet_name="3", index=False)

    dc_construction_cost.to_excel(writer, sheet_name="10", header=False)
    sheet_11_df.to_excel(writer, sheet_name="11", startrow=1, startcol=3, header=False, index=False)

print("✅ data.xlsx 생성 완료")
