import pandas as pd
import numpy as np

# excel_reader.py 가 같은 폴더에 있어야 합니다.
try:
    from excel_reader import read_1d_dict, read_2d_dict, read_3d_dict
except ImportError:
    print("!!! ERROR: 'excel_reader.py' 파일을 찾을 수 없습니다.")
    print("이 스크립트는 excel_reader.py와 같은 폴더에 있어야 합니다.")
    exit()


def verify_parameter(name, param, check_negative=False):
    """파라미터의 정보를 출력하고 검증하는 헬퍼 함수"""
    print(f"--- Verifying: {name} ---")
    try:
        print(f"  - Type: {type(param)}")

        if isinstance(param, (dict, list)):
            print(f"  - Length: {len(param)}")
            if isinstance(param, dict) and len(param) > 0:
                print("  - Sample (first 5 items):")
                for i, (k, v) in enumerate(param.items()):
                    if i >= 5: break
                    print(f"    {k}: {v}")

            if check_negative and isinstance(param, dict):
                # 값이 숫자형인 경우에만 음수 검사
                neg_vals = {k: v for k, v in param.items() if isinstance(v, (int, float)) and v < 0}
                if neg_vals:
                    print(f"  - !!! WARNING: Negative values found: {neg_vals}")
                else:
                    print("  - Sanity Check: No negative cost values found. OK.")

        elif isinstance(param, pd.DataFrame):
            print(f"  - Shape: {param.shape}")
            print("  - Sample (first 5 rows):")
            print(param.head().to_string())

        else:
            print(f"  - Value: {param}")

    except Exception as e:
        print(f"  !!! ERROR while verifying {name}: {e}")

    print("-" * 60 + "\n")


if __name__ == '__main__':
    parameter_data = "data/data1.xlsx"
    print(f"Starting data verification for: {parameter_data}\n")

    # 인덱스 설정 (evaluate.py와 동일하게)
    numP = 5;
    numVD = 20;
    numR = 100;
    Msize = 3;
    ProductNum = 3
    P, Vd, R, M, K = range(numP), range(numVD), range(numR), range(Msize), range(ProductNum)

    # === Fixed Cost Parameters ===
    fj_param = read_1d_dict(parameter_data, "13", "B2:B21")
    verify_parameter("fj_param (Ordering Cost)", fj_param, check_negative=True)

    fp_param = read_2d_dict(parameter_data, "9", "B2:F4")
    verify_parameter("fp_param (Plant Fixed Cost)", fp_param, check_negative=True)

    fd_param = read_1d_dict(parameter_data, "10", "B1:B20")
    verify_parameter("fd_param (DC Fixed Cost)", fd_param, check_negative=True)

    fb_param = read_3d_dict(parameter_data, "11", "D2:D301", (3, 5, 20))
    verify_parameter("fb_param (Route Fixed Cost p-d)", fb_param, check_negative=True)

    fc_param = read_2d_dict(parameter_data, "12", "A2:CV21")
    verify_parameter("fc_param (Route Fixed Cost d-c)", fc_param, check_negative=True)

    # === Capacity Parameters ===
    MP = read_2d_dict(parameter_data, sheet_name="7", cell_range="B2:F4")
    verify_parameter("MP (Max Production)", MP)

    MC = read_1d_dict(parameter_data, sheet_name="8", cell_range="B1:B20")
    verify_parameter("MC (DC Capacity)", MC)

    # === Demand Parameters ===
    df_hat = pd.read_excel(parameter_data, "5", header=None)
    hat_range = df_hat.iloc[1:101, 0:3].reset_index(drop=True)
    hat_range.columns = range(hat_range.shape[1])
    mu_hat = {(r, k): hat_range.iloc[r, k] for r in R for k in K}
    verify_parameter("mu_hat (Perturbed Demand)", mu_hat)

    df_bar = pd.read_excel(parameter_data, "6", header=None)
    bar_range = df_bar.iloc[1:101, 0:3].reset_index(drop=True)
    bar_range.columns = range(bar_range.shape[1])
    mu_bar = {(r, k): bar_range.iloc[r, k] for r in R for k in K}
    verify_parameter("mu_bar (Nominal Demand)", mu_bar)

    # === Variable Cost Parameters (가장 중요한 검증 대상) ===
    # D1: Distance factor 1
    D1 = read_3d_dict(parameter_data, "2", "D2:D301", (3, 5, 20))
    verify_parameter("D1 (Distance Factor 1)", D1, check_negative=True)

    # D2: Distance factor 2
    D2 = read_2d_dict(parameter_data, "3", "A2:CV21")
    verify_parameter("D2 (Distance Factor 2)", D2, check_negative=True)

    # h: Handling cost
    # 이전 검토에서 발견한 의심스러운 부분입니다. 'B'열이 맞는지 확인하세요.
    print(">>> Note: Checking 'h' from sheet 14, column 'B'. Please verify if this column is correct.")
    df_h = pd.read_excel(parameter_data, sheet_name="14", usecols="B", skiprows=1, nrows=20, header=None)
    h = df_h.iloc[:, 0].to_dict()
    verify_parameter("h (Holding Cost)", h, check_negative=True)

    # Fq: Production cost
    df_fq = pd.read_excel(parameter_data, sheet_name="15", header=None)
    fq_range = df_fq.iloc[1:4, 1:6].reset_index(drop=True)
    fq_range.columns = range(fq_range.shape[1])
    Fq = {(k, i): fq_range.iloc[k, i] for k in K for i in P}
    verify_parameter("Fq (Production Cost)", Fq, check_negative=True)

    print("=" * 60)
    print("All parameter checks are complete.")
    print("Please review the output above, especially for '!!! WARNING' messages.")
    print("=" * 60)