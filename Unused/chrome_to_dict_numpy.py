from typing import List, Dict, Tuple, Union # Union 추가
import numpy as np # numpy 임포트 추가


def format_chromosome(
    # chrom의 타입 힌트를 List와 ndarray 모두 가능하도록 수정
    chrom: Union[List[int], np.ndarray],
    num_plants: int = 5,
    num_products: int = 3,
    num_dcs: int = 20,
    num_customers: int = 100,
    num_modes: int = 3
) -> Tuple[
    Dict[Tuple[int,int,int], int],  # z_star[(k,i,j)]
    Dict[Tuple[int,int], int],      # w_star[(j,r)]
    Dict[Tuple[int,int], int],      # beta_star[(r,m)]
    Dict[Tuple[int,int,int], int]   # alpha_star[(j,r,m)]
]:
    """
    (Docstring은 수정 없음)
    """
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 시작 부분에 NumPy 배열을 Python 리스트로 변환하는 코드 추가 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if isinstance(chrom, np.ndarray):
        chrom = chrom.tolist()
    # =====================================================================

    # Validate chromosome length
    N_BINARY = num_products * num_plants * num_dcs
    N_1_5 = num_customers  # DC choice per customer
    N_1_3 = num_customers  # mode choice per customer
    expected_len = N_BINARY + N_1_5 + N_1_3
    if len(chrom) != expected_len:
        raise ValueError(
            f"Chromosome length {len(chrom)} does not match expected {expected_len}."
        )

    # 1) z_star mapping
    z_star: Dict[Tuple[int,int,int], int] = {}
    for k in range(num_products):
        for i in range(num_plants):
            for j in range(num_dcs):
                idx = k * (num_plants * num_dcs) + i * num_dcs + j
                z_star[(k, i, j)] = int(chrom[idx])

    # 2) w_star mapping
    w_star: Dict[Tuple[int,int], int] = {}
    for r in range(num_customers):
        j_sel = int(chrom[N_BINARY + r]) - 1  # convert 1-based to 0-based
        for j in range(num_dcs):
            w_star[(j, r)] = 1 if j == j_sel else 0

    # 3) beta_star mapping
    beta_star: Dict[Tuple[int,int], int] = {}
    for r in range(num_customers):
        m_sel = int(chrom[N_BINARY + N_1_5 + r]) - 1  # 1-based to 0-based
        for m in range(num_modes):
            beta_star[(r, m)] = 1 if m == m_sel else 0

    # 4) alpha_star mapping
    alpha_star: Dict[Tuple[int,int,int], int] = {}
    for j in range(num_dcs):
        for r in range(num_customers):
            for m in range(num_modes):
                alpha_star[(j, r, m)] = (
                    1 if w_star[(j, r)] == 1 and beta_star[(r, m)] == 1 else 0
                )

    return z_star, w_star, beta_star, alpha_star