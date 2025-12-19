# ===================================================================
# constraints_utils.py 파일 수정안
# ===================================================================

import random
from excel_reader import read_2d_dict
from typing import List, Union # Union 추가
import numpy as np # numpy 임포트 추가

def load_s_table(
    filepath: str = "data/CenterCust.xlsx",
    sheet_name: str = "17",
    cell_range: str = "A2:C51"
) -> List[List[int]]:
    # --- 이 함수는 수정할 필요 없음 ---
    """
    Excel 파일에서 s_table을 읽어 2차원 리스트로 반환합니다.

    Returns:
        s_table: shape (num_customers x num_products)의 리스트.
    """
    # Excel에서 (row, col) 키를 가지는 dict로 읽어들임
    s_dict = read_2d_dict(filepath, sheet_name, cell_range)
    # 행/열 크기 계산
    max_row = max(i for (i, _) in s_dict.keys()) + 1
    max_col = max(j for (_, j) in s_dict.keys()) + 1
    # 2차원 리스트로 변환
    s_table = [ [ int(s_dict.get((i, j), 0)) for j in range(max_col) ]
                for i in range(max_row) ]
    return s_table


def adjust_chromosome(
    # chrom의 타입 힌트를 List와 ndarray 모두 가능하도록 수정
    chrom: Union[List[int], np.ndarray],
    s_table: List[List[int]],
    num_plants: int = 3,
    num_products: int = 3,
    num_dcs: int = 10
) -> Union[List[int], np.ndarray]: # 반환 타입도 Union으로 수정
    """
    (Docstring은 수정 없음)
    """
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 시작 부분: NumPy 배열 -> 리스트 변환 & 원래 타입 기억 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    was_numpy = False
    if isinstance(chrom, np.ndarray):
        was_numpy = True
        chrom = chrom.tolist()
    # =================================================================================

    N_BINARY = num_plants * num_products * num_dcs
    num_customers = len(s_table)

    for r in range(num_customers):
        j_sel = chrom[N_BINARY + r] - 1  # 0-based index for DC 선택
        for k in range(num_products):
            if s_table[r][k] == 1:
                # 해당 k, j 구간의 x 인덱스 계산
                base = k * (num_plants * num_dcs)
                indices = [ base + i * num_dcs + j_sel for i in range(num_plants) ]
                # 모두 0이면 위반
                if all(chrom[idx] == 0 for idx in indices):
                    # 최소 하나 1이 될 때까지 랜덤 재할당
                    while True:
                        for idx in indices:
                            chrom[idx] = random.randint(0, 1)
                        if any(chrom[idx] == 1 for idx in indices):
                            break

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 종료 부분: 원래 NumPy 배열이었다면 다시 변환하여 반환 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if was_numpy:
        return np.array(chrom, dtype=int)
    else:
        return chrom
    # =================================================================================