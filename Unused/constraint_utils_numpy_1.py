import random
from excel_reader import read_2d_dict
from typing import List, Union
import numpy as np


def load_s_table(
        filepath: str = "data/data_1.xlsx",
        sheet_name: str = "17",
        cell_range: str = "A2:C101"
) -> List[List[int]]:
    """
    Excel 파일에서 s_table을 읽어 2차원 리스트로 반환합니다.
    (이 함수는 수정되지 않았습니다.)
    """
    s_dict = read_2d_dict(filepath, sheet_name, cell_range)
    max_row = max(i for (i, _) in s_dict.keys()) + 1
    max_col = max(j for (_, j) in s_dict.keys()) + 1
    s_table = [[int(s_dict.get((i, j), 0)) for j in range(max_col)] for i in range(max_row)]
    return s_table


def adjust_chromosome(
        chrom: Union[List[int], np.ndarray],
        s_table: List[List[int]],
        num_plants: int = 5,
        num_products: int = 3,
        num_dcs: int = 20
) -> Union[List[int], np.ndarray]:
    """
    주어진 유전자가 논리적 제약 조건을 만족하도록 보정합니다.
    - 규칙 1: 모든 제품은 최소 2개의 공급 경로를 갖도록 보장합니다.
    - 규칙 2: 고객의 DC 선택은, 해당 고객이 원하는 제품들을 공급받을 수 있는 DC 중에서 이루어지도록 재할당됩니다.
              만약 완벽한 DC가 없다면, 가장 많은 종류를 공급하는 DC를 선택하고 부족한 경로를 강제로 연결합니다.
    """
    was_numpy = isinstance(chrom, np.ndarray)
    # 계산 편의를 위해 파이썬 리스트로 변환
    chrom_list = chrom.tolist() if was_numpy else list(chrom)

    N_BINARY = num_plants * num_products * num_dcs
    num_customers = len(s_table)

    # ===================================================================
    # 규칙 1: 최소 공급 보장
    # 각 제품 k에 대해, 연결된 경로(kij=1)가 전혀 없다면 2개를 강제로 연결합니다.
    # ===================================================================
    for k in range(num_products):
        product_paths_indices = range(k * (num_plants * num_dcs), (k + 1) * (num_plants * num_dcs))

        # 해당 제품의 모든 경로가 0인지 확인
        if all(chrom_list[idx] == 0 for idx in product_paths_indices):
            # 2개의 경로를 무작위로 선택하여 1로 설정
            indices_to_open = np.random.choice(list(product_paths_indices), 2, replace=False)
            for idx in indices_to_open:
                chrom_list[idx] = 1

    # ===================================================================
    # 규칙 2를 위한 사전 준비: 각 DC가 어떤 제품들을 공급하는지 미리 계산
    # ===================================================================
    dc_supplies = [set() for _ in range(num_dcs)]
    for k in range(num_products):
        for i in range(num_plants):
            for j in range(num_dcs):
                idx = k * (num_plants * num_dcs) + i * num_dcs + j
                if chrom_list[idx] == 1:
                    dc_supplies[j].add(k)

    # ===================================================================
    # 규칙 2: 고객별 DC 선택 재할당
    # ===================================================================
    for r in range(num_customers):
        # 고객 r이 원하는 제품들의 집합 (0-based index)
        demanded_products = {k for k, wanted in enumerate(s_table[r]) if wanted == 1}

        if not demanded_products:
            continue  # 원하는 제품이 없으면 다음 고객으로

        # --- 2.1: 고객이 제품을 1종류만 원하는 경우 ---
        if len(demanded_products) == 1:
            k_needed = list(demanded_products)[0]
            # 해당 제품을 공급하는 DC들의 목록
            available_dcs = [j for j, supplied_ks in enumerate(dc_supplies) if k_needed in supplied_ks]

            if available_dcs:
                new_j = random.choice(available_dcs)
                chrom_list[N_BINARY + r] = new_j + 1  # 1-based로 다시 저장

        # --- 2.2: 고객이 제품을 2종류 이상 원하는 경우 ---
        else:
            # 모든 수요 제품을 공급하는 '완벽한' DC 목록
            perfect_dcs = [j for j, supplied_ks in enumerate(dc_supplies) if demanded_products.issubset(supplied_ks)]

            if perfect_dcs:
                # 완벽한 DC가 있으면 그 중 하나를 랜덤 선택
                new_j = random.choice(perfect_dcs)
                chrom_list[N_BINARY + r] = new_j + 1
            else:
                # 완벽한 DC가 없으면 '차선책'을 찾음
                # 1. 가장 많은 종류의 수요 제품을 공급하는 '최선'의 DC를 찾음
                scores = [(j, len(demanded_products.intersection(supplied_ks))) for j, supplied_ks in
                          enumerate(dc_supplies)]
                max_score = max(s for j, s in scores)
                best_effort_dcs = [j for j, s in scores if s == max_score]
                chosen_j = random.choice(best_effort_dcs)

                # 2. 고객의 DC를 '최선'의 DC로 먼저 할당
                chrom_list[N_BINARY + r] = chosen_j + 1

                # 3. 해당 DC에서 여전히 공급 못하는 제품들의 경로를 강제로 연결
                missing_products = demanded_products - dc_supplies[chosen_j]
                for k_miss in missing_products:
                    # k_miss 제품을 chosen_j DC로 보내는 5개 경로 중 하나를 랜덤으로 1로 설정
                    indices_to_fix = [k_miss * (num_plants * num_dcs) + i * num_dcs + chosen_j for i in
                                      range(num_plants)]
                    idx_to_open = random.choice(indices_to_fix)
                    chrom_list[idx_to_open] = 1

    # 원래 타입이 NumPy 배열이었다면 다시 변환하여 반환
    if was_numpy:
        return np.array(chrom_list, dtype=int)
    else:
        return chrom_list