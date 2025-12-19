# Dataset Usage Guide

50개의 데이터셋과 4가지 DI 시나리오를 사용하는 방법입니다.

---

## 데이터셋 구조

### 생성된 파일
```
data/DH_data_full_seed1.pkl   # Dataset 1
data/DH_data_full_seed2.pkl   # Dataset 2
...
data/DH_data_full_seed50.pkl  # Dataset 50
```

### 각 데이터셋 내용
각 데이터셋에는 **4가지 DI 시나리오**가 포함되어 있습니다:

1. **HD (High Demand sensitivity)**: k ∈ [0.667, 1.000]
   - 신선식품, 의약품 등 배송 속도에 매우 민감
   - 빠른 배송 시 수요가 크게 증가

2. **MD (Medium Demand sensitivity)**: k ∈ [0.333, 0.667]
   - 일반 제품, 중간 민감도
   - 빠른 배송 시 수요가 중간 정도 증가

3. **LD (Low Demand sensitivity)**: k ∈ [0.000, 0.333]
   - 저가 제품, 배송 속도에 덜 민감
   - 빠른 배송 시 수요가 조금만 증가

4. **Mixed**: 제품별로 다른 민감도
   - Product 0: HD (High)
   - Product 1: MD (Medium)
   - Product 2: LD (Low)

---

## 실행 방법

### 기본 사용법

```bash
python3 DH_main.py [instance_type] [gamma] --seed [seed] --di [scenario]
```

### 파라미터 설명

- **instance_type**: `toy` 또는 `full`
- **gamma**: Uncertainty budget (0 ~ R)
- **--seed**: 데이터셋 번호 (1-50)
- **--di**: DI 시나리오 (`HD`, `MD`, `LD`, `Mixed`)

---

## 사용 예시

### 예시 1: Seed 1, HD scenario, Gamma=10
```bash
python3 DH_main.py full 10 --seed 1 --di HD
```

### 예시 2: Seed 5, Mixed scenario, Gamma=20
```bash
python3 DH_main.py full 20 --seed 5 --di Mixed
```

### 예시 3: Seed 10, LD scenario, Gamma=50
```bash
python3 DH_main.py full 50 --seed 10 --di LD
```

### 예시 4: Sensitivity analysis (전체 Gamma)
```bash
# --seed와 --di만 지정, gamma 생략 시 전체 sensitivity analysis 실행
python3 DH_main.py full --seed 1 --di HD
```

### 예시 5: 기본 데이터 사용
```bash
# --seed 없으면 기본 데이터 사용
python3 DH_main.py full 10 --di HD
```

---

## 논문 실험 계획

### 총 200개 실험 (50 datasets × 4 scenarios)

#### HD 시나리오: 50개 데이터셋
```bash
for seed in {1..50}; do
    python3 DH_main.py full 10 --seed $seed --di HD
done
```

#### MD 시나리오: 50개 데이터셋
```bash
for seed in {1..50}; do
    python3 DH_main.py full 10 --seed $seed --di MD
done
```

#### LD 시나리오: 50개 데이터셋
```bash
for seed in {1..50}; do
    python3 DH_main.py full 10 --seed $seed --di LD
done
```

#### Mixed 시나리오: 50개 데이터셋
```bash
for seed in {1..50}; do
    python3 DH_main.py full 10 --seed $seed --di Mixed
done
```

---

## 배치 실행 스크립트

### run_all_experiments.sh 생성

```bash
#!/bin/bash
# Run all 200 experiments (50 seeds × 4 scenarios)

GAMMA=10  # or any gamma value you want to test

for DI_SCENARIO in HD MD LD Mixed; do
    echo "Running $DI_SCENARIO scenario..."
    for SEED in {1..50}; do
        echo "  Seed $SEED / 50"
        python3 DH_main.py full $GAMMA --seed $SEED --di $DI_SCENARIO
    done
done

echo "All 200 experiments completed!"
```

### 병렬 실행 (서버에서)

```bash
# 4개 시나리오 동시 실행
python3 DH_main.py full 10 --seed 1 --di HD &
python3 DH_main.py full 10 --seed 1 --di MD &
python3 DH_main.py full 10 --seed 1 --di LD &
python3 DH_main.py full 10 --seed 1 --di Mixed &
wait

# 다음 seed로
python3 DH_main.py full 10 --seed 2 --di HD &
python3 DH_main.py full 10 --seed 2 --di MD &
# ...
```

---

## 결과 파일 형식

### 단일 실행 결과
```
result/DH_single_full_seed1_HD_gamma10_20251219_143000.csv
result/DH_single_full_seed1_Mixed_gamma20_20251219_144000.csv
```

### 파일 내용
```csv
Gamma,Seed,DI_Scenario,Converged,Iterations,Total_Time,Optimal_Value,LB,UB,Gap,Num_Scenarios
10,1,HD,True,5,1234.56,347286.96,347286.95,347286.96,0.000001,3
```

---

## DI 값 확인

### 특정 데이터셋의 DI 시나리오 확인

```python
from DH_data_gen import SupplyChainData

# Load dataset
data = SupplyChainData.load('data/DH_data_full_seed1.pkl')

# Print all scenarios
for scenario in ['HD', 'MD', 'LD', 'Mixed']:
    print(f"\n{scenario} Scenario:")
    for k in range(3):
        di_vec = data.DI_scenarios[scenario][k]
        k_param = data.DI_k_params[scenario][k]
        print(f"  Product {k}: {di_vec} (k={k_param:.3f})")
```

---

## DI 수식

각 제품의 DI는 다음 수식으로 생성됩니다:

```
DI_m = (3/2)^(k × m)  for m = 0, 1, 2
```

Where:
- m = 0 (Mode 0, slow): DI = 1.000 (항상)
- m = 1 (Mode 1, medium): DI = 1.5^k
- m = 2 (Mode 2, fast): DI = 1.5^(2k)

k parameter 범위:
- **HD**: k ~ Uniform[2/3, 1.0]  →  DI_2 ∈ [1.72, 2.25]
- **MD**: k ~ Uniform[1/3, 2/3]  →  DI_2 ∈ [1.31, 1.72]
- **LD**: k ~ Uniform[0.0, 1/3]  →  DI_2 ∈ [1.00, 1.31]

---

## 재현성 (Reproducibility)

### 동일한 결과를 재현하려면:

1. **Seed 지정**: 같은 seed 번호 사용
2. **DI 시나리오 지정**: 같은 시나리오 사용
3. **Gamma 지정**: 같은 gamma 값 사용

예시:
```bash
# 이 명령은 항상 동일한 결과를 생성합니다
python3 DH_main.py full 10 --seed 1 --di HD
```

### 새로운 데이터셋 생성

```bash
# 50개 데이터셋 재생성 (다른 random values)
python3 generate_multiple_datasets.py
```

---

## 통계 분석을 위한 팁

### 같은 시나리오 50개 결과 수집

```bash
# HD 시나리오 모든 seed 실행
for seed in {1..50}; do
    python3 DH_main.py full 10 --seed $seed --di HD
done

# 결과 파일 모두 result/ 폴더에 저장됨
ls result/DH_single_full_seed*_HD_gamma10_*.csv
```

### Python으로 결과 통합

```python
import pandas as pd
import glob

# HD 시나리오의 모든 결과 수집
hd_files = glob.glob('result/DH_single_full_seed*_HD_gamma10_*.csv')
hd_results = pd.concat([pd.read_csv(f) for f in hd_files])

# 통계 계산
print(f"Mean Optimal Value: {hd_results['Optimal_Value'].mean()}")
print(f"Std Optimal Value: {hd_results['Optimal_Value'].std()}")
print(f"Mean Iterations: {hd_results['Iterations'].mean()}")
print(f"Mean Time: {hd_results['Total_Time'].mean()}")
```

---

## 주의사항

1. **Seed 범위**: 1-50 (full instance)
2. **Gamma 범위**: 0-100 (full instance, R=100)
3. **DI 시나리오**: 반드시 대문자 사용 (HD, MD, LD, Mixed)
4. **디스크 공간**: 결과 파일이 많이 생성되므로 충분한 공간 확보
5. **실행 시간**: Full instance는 gamma 당 20분-2시간 소요 가능

---

## 문제 해결

### 데이터셋이 없다는 에러
```
Error: Data file not found: data/DH_data_full_seed51.pkl
Available seeds for full: 1-50
```
→ 1-50 범위 내의 seed 사용

### DI 시나리오 경고
```
Warning: DI scenario 'hd' not found in data. Using default DI values.
```
→ 대문자 사용: `--di HD` (not `--di hd`)

### 메모리 부족
→ Toy instance로 테스트: `python3 DH_main.py toy 5 --seed 1 --di HD`
