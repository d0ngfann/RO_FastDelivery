# Experiment Execution Plan — 코드 수정 및 실험 재실행 계획

> revision_checklist.md의 체크된 항목 중 **코드 수정 + 실험 실행**이 필요한 사항만 추출하여
> 구현 순서대로 정리한 실행 계획입니다.
> **실행 환경: 집 컴퓨터 (로컬).** GitHub에서 Revision_0326 폴더를 받아서 실행.

---

## 전체 요약

| 구분 | 실험 규모 | 비고 |
|------|----------|------|
| 기본 재실행 (밑수 변경 + ε 변경) | 4,800 runs | Exp1: 800 + Exp2: 4,000 |
| Breakeven Analysis | 500 runs | TC_2 변경, LD only, Optimal+FM2 |
| 선형 DI 비교 | 320 runs | Robustness check |
| 서비스 커버리지 SA | 480 runs | 3 coverage × 10 seeds × 4 DI × 4 policies |
| **총 실험 규모** | **~6,100 runs** | |

---

## Step 1: 코드 수정 (모든 실험 전에 완료)

> 모든 파라미터/모델 변경을 먼저 확정한 뒤 한 번에 재실행하는 것이 효율적.
> 아래 수정사항을 모두 반영한 후 Step 2로 진행.

### 1-1. DI 함수 밑수 변경: 1.5 → 1.265 (√1.6)

- [ ] **파일:** `DH_data_gen.py` — `generate_DI_vector()` 함수
  - 현재: `return [1.5 ** (k * m) for m in range(M)]`
  - 변경: `return [1.265 ** (k * m) for m in range(M)]`
  - 또는 `math.sqrt(1.6)` 사용으로 정확한 값 보장
- [ ] **파일:** `DH_config.py` — `DataParameters.DI_base` (backward compatibility용 디폴트)
  - 현재: `[1.0, 1.2, 1.5]` (하드코딩)
  - 변경: 조정된 밑수에 맞게 업데이트
- [ ] **확인:** `generate_DI_scenarios()` 함수의 docstring 업데이트 (1.5 → 1.265 관련 주석)
- [ ] **근거:** Marino & Zotteri (2018) 실증 상한(+60%)에 calibrate. HD에서 DI_2 = 1.60

### 1-2. 수렴 허용 오차 변경: ε = 100 고정

- [ ] **파일:** `DH_config.py` — `ProblemConfig.__init__()`
  - 기본값: `self.epsilon = 200` → `self.epsilon = 100`
  - full200: `self.epsilon = 800` → `self.epsilon = 100`
- [ ] **파일:** `DH_algo.py` — **adaptive convergence tiers 제거**
  - 현재: 1시간 후 ε=600, 2시간 후 ε=1000으로 자동 상향하는 로직 존재
  - 변경: adaptive tiers 완전 제거, ε=100 고정으로 운영
  - `_check_adaptive_mode()` 메서드 제거 또는 비활성화
  - `solve()` 루프에서 `_check_adaptive_mode()` 호출 제거
- [ ] **주의:** 수렴 시간 증가 가능. 일부 인스턴스가 시간 제한 내 수렴 못 하면 ε=200으로 상향 검토

### 1-3. 서비스 커버리지 제약 구현 (Master Problem 전처리)

> 핵심: C&CG 루프 코드 자체는 변경 불필요. Master Problem 빌드 시 α 변수 상한 고정만 추가.

- [ ] **파일:** `DH_master.py` — 커버리지 제약 메서드 추가
  - α[j,r,m] 변수 생성 후, D2[j,r] > D_bar[m]이면 `alpha[j,r,m].ub = 0`
  - 구현 방식: **방법 2 (변수 상한 고정)** — 가장 안전
  ```python
  def apply_coverage_constraint(self, D_bar):
      """Apply service coverage: alpha[j,r,m] = 0 if D2[j,r] > D_bar[m]"""
      count = 0
      for j in range(self.J):
          for r in range(self.R):
              for m in range(self.M):
                  if self.data.D2[(j, r)] > D_bar[m]:
                      self.alpha[(j, r, m)].ub = 0
                      count += 1
      print(f"  [COVERAGE] Fixed {count} alpha variables to 0")
  ```

- [ ] **파일:** `DH_master_fixed_mode.py` — 동일한 커버리지 메서드 추가
  - Fixed Mode에서도 커버리지 적용 필요 (FM2 infeasibility 발생 가능)

- [ ] **D_bar 파라미터 계산 로직 추가**
  - 각 seed의 D2 분포에서 percentile 기반으로 D_bar 계산
  ```python
  def compute_coverage_thresholds(data, scenario='moderate'):
      """Compute D_bar[m] based on D2 distance distribution percentiles."""
      import numpy as np
      all_D2 = list(data.D2.values())
      if scenario == 'tight':
          return {0: float('inf'), 1: np.percentile(all_D2, 50), 2: np.percentile(all_D2, 25)}
      elif scenario == 'moderate':
          return {0: float('inf'), 1: np.percentile(all_D2, 75), 2: np.percentile(all_D2, 50)}
      elif scenario == 'relaxed':
          return {0: float('inf'), 1: np.percentile(all_D2, 90), 2: np.percentile(all_D2, 75)}
      elif scenario == 'no_limit':
          return {0: float('inf'), 1: float('inf'), 2: float('inf')}
  ```

- [ ] **FM2 infeasibility 대응:** infeasible 케이스는 보고에서 명시하고 제외 (옵션 ii)

### 1-4. TC_2 파라미터 외부 주입 기능 (Breakeven Analysis용)

- [ ] **파일:** `DH_main.py`, `DH_main_fixed_mode.py`
  - `--tc2` 커맨드라인 인자 추가, 데이터 로드 후 `data.TC[2]` 오버라이드
  ```python
  parser.add_argument('--tc2', type=float, default=None, help='Override TC_2 for breakeven analysis')
  # 데이터 로드 후:
  if args.tc2 is not None:
      data.TC[2] = args.tc2
  ```

### 1-5. 선형 DI 함수 지원 (Robustness Check용)

- [ ] **파일:** `DH_data_gen.py` — 선형 DI 함수 추가
  ```python
  def generate_DI_vector_linear(k, M=3):
      """Linear DI: DI_m = 1 + 0.3 * k * m (anchored to same endpoint as exponential)"""
      return [1.0 + 0.3 * k * m for m in range(M)]
  ```
- [ ] **파일:** `DH_main.py`, `DH_main_fixed_mode.py` — `--di-func` 인자 추가
  - `exponential` (기본) 또는 `linear` 선택
  - 선형 선택 시: 데이터에서 k_params를 읽어서 선형 함수로 DI를 런타임 재계산
  - 데이터 재생성 불필요 (같은 .pkl, 같은 k_params, 다른 함수형)

### 1-6. 데이터 재생성

- [ ] 밑수 1.265로 코드 수정 후 데이터 재생성
  - `cd Revision_0326/codes && python3 generate_50_seeds.py full`
  - `cd Revision_0326/codes && python3 generate_50_seeds.py full200`
- [ ] 생성된 데이터: `Revision_0326/data/DH_data_full_seed{1-50}.pkl`, `DH_data_full200_seed{1-50}.pkl`
- [ ] `generate_50_seeds.py`의 data 저장 경로가 `Revision_0326/data/`를 가리키도록 확인/수정

---

## Step 2: 검증 (소규모 테스트)

> 코드 수정 후 전체 실행 전에 toy 인스턴스 또는 소규모로 검증.

- [ ] **2-1.** toy 인스턴스에서 밑수 1.265 + ε=100 실행 → 수렴 확인
- [ ] **2-2.** toy 인스턴스에서 커버리지 제약 적용 → α 고정 확인, 정상 수렴 확인
- [ ] **2-3.** full 인스턴스 seed 1, HD, gamma 10에서 테스트 실행
  - 밑수 변경 전후 DI 값 비교 출력
  - 수렴 속도 확인 (ε=100이 합리적 시간 내 수렴하는지)
- [ ] **2-4.** TC_2 오버라이드 테스트 (TC_2=0.40)
- [ ] **2-5.** 선형 DI 테스트

---

## Step 3: 기본 재실행 (4,800 runs)

> 밑수 1.265 + ε=100 적용. 커버리지 제약 없음 (No limit = 현재 모형과 동일).
> 이 결과가 논문의 메인 결과표가 됨.

### 3-1. Experiment 1 재실행: R=50, Gamma=10 (800 runs)

- [ ] 로컬 실행 스크립트 작성 (Python subprocess 또는 shell script)
  - 50 seeds × 4 DI × 4 policies = 800 runs
  - 순차 또는 멀티프로세스 실행
- [ ] 결과 디렉토리: `result/exp1/`
- [ ] 결과 수집 및 완료 확인 (800개 결과 파일)

### 3-2. Experiment 2 재실행: R=200, Gamma sensitivity (4,000 runs)

- [ ] 로컬 실행 스크립트 작성
  - 50 seeds × 4 DI × 4 policies × 5 Gamma(20,40,60,80,100) = 4,000 runs
- [ ] 결과 디렉토리: `result/exp2/`
- [ ] 결과 수집 및 완료 확인 (4,000개 결과 파일)

### 3-3. ε=100 수렴성 모니터링

- [ ] 초기 결과(seed 1-5)로 수렴률 확인
- [ ] 수렴 실패 다수 발생 시 → ε=200으로 상향 후 재실행
- [ ] seed 38 특이 인스턴스 별도 확인

---

## Step 4: Breakeven Analysis (500 runs)

> TC_2 변경 실험. LD 시나리오에서 FM2 우위가 역전���는 breakeven point 식별.
> 체크리스트 1-3, 3-3 항목.

### 실험 설계

| TC_2 | slow 대비 배율 | 해석 |
|------|---------------|------|
| 0.20 | 4x (현재값) | 기준선 |
| 0.40 | 8x | |
| 0.60 | 12x | |
| 0.80 | 16x | |
| 1.00 | 20x | 극단적 비용 |

- **DI 시나리오:** LD only (역전이 가장 먼저 일어나는 조건)
- **인스턴스:** full (R=50), Gamma=10
- **정책:** Optimal + FM2 (2정책)
- **규모:** 5 TC_2 × 50 seeds × 2 policies = **500 runs**

### 실행 계획

- [ ] 로컬 실행 스크립트 작성
  - 각 TC_2 값에 대해 `--tc2` 인자로 전달
- [ ] 결과 디렉토리: `result/breakeven/`
- [ ] 결과 분석: TC_2별 Optimal vs FM2 profit 비교 → breakeven point 식별

---

## Step 5: 선형 DI 비교 실험 (320 runs)

> DI 함수 형태에 대한 robustness check.
> 체크리스트 2-1 (추가 실험), 3-4 항목.

### 실험 설계

| 함수 | 수식 | m=2, κ=1 값 |
|------|------|------------|
| 지수 (조정) | (1.265)^(κ·m) | 1.60 |
| 선형 | 1 + 0.3·κ·m | 1.60 |

- **앵커 포인트 통일:** 두 함수의 끝점(m=0: 1.0, m=2/κ=1: 1.60) 동일
- **차이:** 중간 경로 — 지수 m=1에서 1.27, 선형 m=1에서 1.30

### 실행 계획

- [ ] 로컬 실행 스크립트 작성
  - `--di-func exponential` (기본) / `--di-func linear`
  - 2 함수 × 4 DI × 10 seeds × 4 policies = 320 runs
  - 인스턴스: full (R=50), Gamma=10
- [ ] 결과 디렉토리: `result/linear_di/`
- [ ] 결과 분석:
  - [ ] (a) Optimal vs FM2의 VoF가 함수형에 따라 변하는지
  - [ ] (b) 최적 모드 배분 비율이 달라지는지
  - [ ] (c) 선형에서 FM1 선택 비율 증가 여부

---

## Step 6: 서비스 커버리지 SA (480 runs)

> 커버리지 제한 시 VoF 격차 변화 분석.
> 체크리스트 3-1 항목. R2가 "논문 기여를 가장 크게 향상시킬 수 있는 단일 변경"으로 평가.

### 실험 설계

| 시나리오 | m=2 커버리지 | m=1 커버리지 | m=0 | 해석 |
|---------|-------------|-------------|-----|------|
| Tight | 25th percentile | 50th percentile | 무제한 | 빠른배송 매우 근거리만 |
| Moderate | 50th percentile | 75th percentile | 무제한 | 중간 수준 |
| Relaxed | 75th percentile | 90th percentile | 무제한 | 느슨한 제한 |
| No limit | 100% | 100% | 무제한 | 기준선 (Step 3 결과 재사용) |

- **No limit**은 Step 3의 기본 재실행 결과와 동일 → 새로 실행 불필요
- **실제 추가:** 3 커버리지 × 10 seeds × 4 DI × 4 policies = **480 runs**

### 실행 계획

- [ ] 로컬 실행 스크립트 작성
  - `--coverage tight` / `moderate` / `relaxed`
  - 인스턴스: full (R=50), Gamma=10, Seeds: 1-10
- [ ] 결과 디렉토리: `result/coverage/`
- [ ] **FM2 infeasibility 모니터링:**
  - [ ] Tight 시나리오에서 FM2가 infeasible한 seed 수 집계
  - [ ] infeasible 케이스는 결과에서 제외하되, 발생 비율을 보고
- [ ] 결과 분석:
  - [ ] (a) 커버리지별 Optimal vs FM2의 VoF% 변화 (0.2% → 5-15% 예상)
  - [ ] (b) 커버리지별 모드 배분 비율 (m=0,1,2 고객 비율)
  - [ ] (c) 커버리지별 평균 DC 개설 수 변화 (2.86에서 증가하는지)

---

## Step 7: 결과 분석 및 통합

- [ ] **7-1.** 기본 재실행 결과로 메인 테이블 갱신
  - 새 VoF 수치 (밑수 1.265 기준)
  - 새 Gamma sensitivity 결과
- [ ] **7-2.** 절대 VoF 표 작성 (VoF% 옆에 병렬 배치)
- [ ] **7-3.** Breakeven point 식별 및 그래프 생성
  - "TC_2가 X배 이상이면 FM2 비효율로 전환" 결론 도출
- [ ] **7-4.** 선형 DI 비교 결과 정리
  - 결론이 함수형에 robust한지 확인
- [ ] **7-5.** 커버리지 SA 결과 정리
  - VoF 변화 표/그래프
  - DC 개설 수 변화
- [ ] **7-6.** 다중 비교 보정 적용 (Bonferroni 또는 Holm) — 통계 검정 12개에 대해
- [ ] **7-7.** seed 38 비수렴 원인 분석

---

## 실행 순서 및 의존관계

```
[Step 1] 코드 수정 (모두 완료 후 진행)
    │
    ├── 1-1 밑수 변경 ──────┐
    ├── 1-2 ε 변경 + adaptive 제거 ┤
    ├── 1-3 커버리지 구현 ───┤ (병렬 수정 가능)
    ├── 1-4 TC_2 오버라이드 ─┤
    ├── 1-5 선형 DI 지원 ────┤
    └── 1-6 데이터 재생성 ───┘
                │
    [Step 2] 검증 (toy + full seed 1)
                │
    ┌───────────┼───────────┐
    │           │           │
[Step 3]    [Step 4]    [Step 5]     ← 병렬 실행 가능
기본 재실행  Breakeven   선형 DI 비교
4,800 runs  500 runs    320 runs
    │           │           │
    └───────────┼───────────┘
                │
          [Step 6]
          커버리지 SA
          480 runs
          (Step 3 기준선 필요)
                │
          [Step 7]
          결과 분석 통합
```

---

## 코드 파일별 수정 요약

| 파일 | 수정 내용 | Step |
|------|----------|------|
| `DH_data_gen.py` | 밑수 1.5→1.265, 선형 DI 함수 추가 | 1-1, 1-5 |
| `DH_config.py` | ε 변경 (전체 100으로 통일), DI_base 업데이트 | 1-2 |
| `DH_algo.py` | adaptive convergence tiers 제거, ε=100 고정 | 1-2 |
| `DH_master.py` | `apply_coverage_constraint()` 메서드 추가 | 1-3 |
| `DH_master_fixed_mode.py` | 동일 커버리지 메서드 추가 | 1-3 |
| `DH_main.py` | `--tc2`, `--di-func`, `--coverage` 인자 추가 | 1-4, 1-5, 1-3 |
| `DH_main_fixed_mode.py` | 동일 인자 추가 | 1-4, 1-5, 1-3 |
| `DH_sub.py` | 변경 없음 (Subproblem 구조 유지) | — |
| `generate_50_seeds.py` | data 경로 확인 (밑수 변경은 DH_data_gen.py에서 반영됨) | 1-6 |

### 새로 작성할 파일

| 파일 | 용도 |
|------|------|
| `run_exp1.py` (또는 .sh) | Step 3-1 로컬 실행 스크립트 |
| `run_exp2.py` (또는 .sh) | Step 3-2 로컬 실행 스크립트 |
| `run_breakeven.py` (또는 .sh) | Step 4 로컬 실행 스크립트 |
| `run_linear_di.py` (또는 .sh) | Step 5 로컬 실행 스크립트 |
| `run_coverage.py` (또는 .sh) | Step 6 로컬 실행 스크립트 |

---

## 리스크 및 대응

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| ε=100에서 수렴 실패 다수 | 중간 | ε=200으로 상향, seed별 모니터링 |
| 밑수 변경 후 VoF 수치 대폭 변동 | 높음 | 예상된 변화. 새 결과 기준으로 해석 재작성 |
| FM2 infeasibility (커버리지 Tight) | 높음 | 예상된 결과. infeasible 비율 자체가 의미 있는 발견 |
| 집 컴퓨터 실행 시간 | 중간 | 멀티프로세스 병렬 실행, full은 빠름(분 단위), full200은 시간 단위 |
| 선형 DI에서 결론 역전 | 낮음 | 앵커 포인트 통일로 큰 차이 없을 것으로 예상 |
