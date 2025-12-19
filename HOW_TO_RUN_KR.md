# 실행 방법 및 결과 확인 가이드

## 📂 결과 파일 위치

모든 결과는 **`result/`** 디렉토리에 저장됩니다:

```
result/
├── DH_sensitivity_toy_YYYYMMDD_HHMMSS.csv     # 결과 데이터 (CSV)
├── DH_sensitivity_toy_YYYYMMDD_HHMMSS.png     # 시각화 그래프 (PNG)
├── DH_sensitivity_full_YYYYMMDD_HHMMSS.csv    # Full 인스턴스 결과
└── DH_sensitivity_full_YYYYMMDD_HHMMSS.png    # Full 인스턴스 그래프
```

### 결과 CSV 파일 형식

| 컬럼명 | 설명 |
|--------|------|
| Gamma | 불확실성 예산 (Γ) |
| Converged | 수렴 여부 (True/False) |
| Iterations | C&CG 반복 횟수 |
| Total_Time | 총 실행 시간 (초) |
| Optimal_Value | 최적 목적함수 값 |
| LB | 하한값 (Lower Bound) |
| UB | 상한값 (Upper Bound) |
| Gap | 갭 (UB - LB) |
| Num_Scenarios | 추가된 시나리오 개수 |

## 🚀 Gamma 값 수정하여 실행하는 방법

### 방법 1: DH_main.py 사용 (권장)

**전체 Sensitivity Analysis 실행**

```bash
# Toy 인스턴스 (Γ = 0, 1, 2, 3, 4, 5)
python3 DH_main.py toy

# Full 인스턴스 (Γ = 0, 10, 20, ..., 100)
python3 DH_main.py full
```

**Gamma 범위를 수정하려면:**

`DH_main.py` 파일의 `run_sensitivity_analysis()` 함수를 수정:

```python
# 54번째 줄 근처
if instance_type == 'toy':
    gamma_values = [0, 1, 2, 3, 4, 5]  # ← 여기를 수정
elif instance_type == 'full':
    gamma_values = list(range(0, 101, 10))  # ← 여기를 수정
```

**예시: Gamma를 0부터 10까지 실행하려면:**

```python
gamma_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 또는
gamma_values = list(range(0, 11))  # 0부터 10까지
```

### 방법 2: 단일 Gamma 값 실행

특정 Gamma 값 하나만 테스트하려면:

```python
# test_single_gamma.py 파일 생성
from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_algo import CCGAlgorithm

# 설정
instance_type = 'toy'  # 또는 'full'
gamma_value = 5        # ← 원하는 Gamma 값

# 실행
config = ProblemConfig(instance_type=instance_type)
config.Gamma = gamma_value
data = SupplyChainData.load(config.data_file)

ccg = CCGAlgorithm(data, config)
result = ccg.run()

# 결과 출력
print(f"Gamma = {gamma_value}")
print(f"Optimal Value: {result['optimal_value']:.2f}")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Time: {result['total_time']:.2f}s")
```

실행:
```bash
python3 test_single_gamma.py
```

### 방법 3: DH_config.py에서 Gamma 범위 설정

`DH_config.py`에 클래스를 추가하여 사용:

```python
# DH_config.py 하단에 추가
class SensitivityConfig:
    """Gamma sensitivity analysis configuration."""

    def __init__(self, instance_type='toy'):
        if instance_type == 'toy':
            # Toy 인스턴스용 Gamma 값들
            self.gamma_values = [0, 1, 2, 3, 4, 5]
        elif instance_type == 'full':
            # Full 인스턴스용 Gamma 값들
            self.gamma_values = list(range(0, 101, 10))
```

## 📊 결과 확인 방법

### 1. CSV 파일 확인

```bash
# 최신 결과 파일 확인
ls -lt result/*.csv | head -1

# CSV 파일 내용 보기
cat result/DH_sensitivity_toy_20251218_222623.csv
```

### 2. 그래프 확인

PNG 파일을 열어서 시각화 확인:
- **Optimal Value vs Gamma**: 최적값이 Gamma에 따라 어떻게 변하는지
- **Scenarios vs Gamma**: 추가된 시나리오 개수
- **Iterations vs Gamma**: 수렴까지 필요한 반복 횟수
- **Time vs Gamma**: 계산 시간

### 3. Python으로 결과 분석

```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('result/DH_sensitivity_toy_20251218_222623.csv')

# 요약 통계
print(df.describe())

# 특정 Gamma 값의 결과 확인
gamma_3_result = df[df['Gamma'] == 3]
print(gamma_3_result)
```

## 📝 실행 예시

### 예시 1: Toy 인스턴스, Gamma = 0~10

```bash
# 1. DH_main.py 수정
# gamma_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 2. 실행
python3 DH_main.py toy

# 3. 결과 확인
ls -lh result/DH_sensitivity_toy_*.csv | head -1
cat result/DH_sensitivity_toy_*.csv
```

### 예시 2: Full 인스턴스, Gamma = 0, 25, 50, 75, 100

```bash
# 1. DH_main.py 수정
# gamma_values = [0, 25, 50, 75, 100]

# 2. 실행
python3 DH_main.py full

# 3. 결과 확인 (시간이 오래 걸림)
```

### 예시 3: 단일 Gamma 값 빠른 테스트

```bash
# test_single_gamma.py에서 gamma_value = 3 설정 후
python3 test_single_gamma.py
```

## ⚙️ DH_config.py에서 Gamma 기본값 설정

현재 Gamma 관련 설정:

```python
# DH_config.py
class SensitivityConfig:
    """Sensitivity analysis configuration."""

    def __init__(self, instance_type='toy'):
        if instance_type == 'toy':
            # Toy 인스턴스용 - 빠른 테스트
            self.gamma_values = [0, 1, 2, 3, 4, 5]
            self.gamma_min = 0
            self.gamma_max = 5
            self.gamma_step = 1

        elif instance_type == 'full':
            # Full 인스턴스용 - 전체 분석
            self.gamma_values = list(range(0, 101, 10))
            self.gamma_min = 0
            self.gamma_max = 100
            self.gamma_step = 10
```

## 🔍 디버깅 및 상세 분석

특정 Gamma 값에 대한 상세 분석:

```bash
# Gamma=3일 때 상세 디버깅
python3 DH_debug_gap.py toy 3

# Gamma=50일 때 상세 디버깅 (full 인스턴스)
python3 DH_debug_gap.py full 50
```

## 📌 주의사항

1. **Full 인스턴스는 시간이 오래 걸립니다**
   - Gamma=0: 약 20-30분
   - Gamma=100: 수 시간 소요 가능

2. **Gamma 값이 클수록 시간이 증가합니다**
   - 더 많은 시나리오 추가됨
   - 더 많은 C&CG 반복 필요

3. **결과 파일은 자동 저장됩니다**
   - 중간에 중단되어도 temp 파일 확인 가능
   - `result/DH_sensitivity_*_temp.csv`

## 🎯 Quick Reference

| 작업 | 명령어 |
|------|--------|
| Toy 전체 실행 | `python3 DH_main.py toy` |
| Full 전체 실행 | `python3 DH_main.py full` |
| 단일 Gamma 테스트 | `python3 test_single_gamma.py` (파일 수정 필요) |
| 결과 확인 | `ls -lh result/` |
| 최신 CSV 보기 | `cat result/DH_sensitivity_toy_*.csv \| tail -1` |
| 디버깅 | `python3 DH_debug_gap.py toy 3` |

---

**작성일**: 2025-12-18
