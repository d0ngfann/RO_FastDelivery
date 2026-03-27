# 역질문 답변 종합 보고서

**논문:** A Robust Optimization Framework for Logistics Management: Balancing Fast Delivery and Demand Uncertainty
**작성일:** 2026-03-21
**구성:** 4인 리뷰어 역질문 답변 + 교차 분석 + 통합 액션 플랜

---

# Part 1: 리뷰어별 답변 전문

---

## 1. Reviewer 1 (방법론 전문가) 답변

### RQ1-1: 수요 함수 대안 설계 — 구체적 함수식

공정한 비교를 위해, 모든 대안 함수가 다음 두 가지 **앵커 포인트(anchor points)**를 공유하도록 설정:
- $m=0$일 때: $DI = 1$ (기준선)
- $m=2$, $\kappa=1$ (HD 최대)일 때: $DI = (1.5)^2 = 2.25$ (현재 지수 함수의 최대값과 동일)

| 함수 형태 | 수식 | $m=1, \kappa=1$ 값 | 특성 |
|-----------|------|-------------------|------|
| 지수 (현재) | $(1.5)^{\kappa \cdot m}$ | 1.50 | 빠른 모드의 한계 효과 증가 |
| 선형 | $1 + \frac{(1.5^{2\kappa} - 1)}{2} \cdot m$ | 1.625 | 균등한 한계 효과. medium 모드 가치를 더 높게 평가 |
| 로그 | $1 + (1.5^{2\kappa} - 1) \cdot \frac{\ln(1+m)}{\ln(3)}$ | ~1.79 | 수확 체감(diminishing returns). $m=0 \to 1$ 전환 효과가 큼 |
| S-curve | $1 + (1.5^{2\kappa} - 1) \cdot \frac{1}{1 + e^{-3(m-1)}}$ | ~1.625 | $m=1$ 부근에서 가장 급격한 변화 |

**비교 실험 설계:** 4가지 함수형 x 4 DI 시나리오 x 10개 seed = **640회 실험**이면 충분.

**핵심 보고 지표:**
1. Optimal vs FM2의 VoF가 함수형에 따라 변하는지
2. 최적 모드 배분 비율($m=0,1,2$의 고객 비율)이 함수형에 따라 달라지는지
3. 선형/로그 함수에서 medium 모드 선택 비율 증가 여부 → 지수 함수의 편향 증거

---

### RQ1-2: $\epsilon$ 민감도 분석의 우선순위

**미수렴 처리:** 시간 제한 1시간. 미수렴 시 best feasible solution의 objective value와 현재 UB-LB gap을 함께 보고. 수렴/미수렴 인스턴스를 분리 보고.

**우선순위: $\epsilon$ 민감도 분석이 규모 확장보다 우선.**

이유:
1. 현재 가장 취약한 주장은 "Optimal > FM2"인데, VoF(945)와 $\epsilon$(800)이 비슷한 크기 → 이 신뢰성 확보가 최우선
2. 규모 확장은 "nice to have"이지만, 현재 규모의 결과 자체가 불확실하면 의미 제한적
3. $\epsilon$ 변경은 파라미터 하나만 수정하면 되므로 구현 비용 최소

**둘 다 할 여유 있으면:** $\epsilon$ 민감도 → 소규모($|R|=50$) 확장 순서. 소규모에서는 작은 $\epsilon$으로도 빠르게 수렴하므로 두 분석을 동시에 수행 가능.

---

### RQ1-3: SP/DRO 비교 설계

**권장: Wasserstein DRO**

이유:
1. Moment-based DRO는 budgeted RO와 차별성 약함
2. Wasserstein DRO가 최근 문헌의 주류
3. Wasserstein 반경 $\varepsilon_W$를 $\Gamma$와 비교 가능

**구체적 설정:**
- 경험적 분포: 50개 seed에서 base demand에 노이즈 추가하여 $N=100$개 시나리오 생성
- Wasserstein 반경: $\varepsilon_W \in \{0.1, 0.5, 1.0, 2.0\} \times \hat{\mu}_{avg}$
- SAA: 동일 100개 시나리오의 기대값 최적화 (DRO의 $\varepsilon_W = 0$)

**초점: Out-of-sample 성능이 핵심.**
- 50개 seed 중 30개로 모델을 풀고 (in-sample)
- 나머지 20개의 demand realization에서 실제 profit 평가 (out-of-sample)
- 보고 지표: (a) 평균 out-of-sample profit, (b) worst-case out-of-sample profit, (c) profit 변동성

**시간 제한적이면 SAA와의 비교만으로도 충분** — SAA는 구현 간단하고 RO 대비 가장 명확한 대조 제공.

---

### RQ1-4: Dual Derivation의 범위

**LP dual의 표준 결과를 전제하고, 문제 특화된 유도만 보여주면 충분.**

부록에 포함할 내용:
1. **출발점:** "2단계 문제는 A, u에 대해 선형이므로 strong duality 성립" — 한 문장. LP feasibility는 shortage 변수 $u_{rk}$로 자명.
2. **Dual 변수 대응표:** Primal 제약 ↔ Dual 변수 ↔ 의미
3. **Dual 목적함수 전개:** $\gamma_{rk}$에 $\tilde{d}_{rk}$가 곱해지는 항을 명확히 → bilinear term $\eta_{rk} \cdot \gamma_{rk}$ 발생 지점
4. **$\gamma_{rk}$ bounds 유도:** Dual feasibility로부터 $\gamma_{rk} \in [-(S+SC), S]$ 도출 과정 (2-3줄)
5. **McCormick 연결:** bilinear term linearization으로 자연스럽게 전환

**분량: 2-3페이지 부록.**

---

### RQ1-5: McCormick Tightness 개선의 Trade-off

**우선순위 낮은 개선.**

이유:
1. $(i,j,r,m)$-specific bounds 적용 시 McCormick 제약 수가 2,400 → 108,000으로 **45배 증가**
2. $\gamma_{rk}$는 모든 $(i,j,m)$ 조합에 걸쳐 하나의 변수 → variable-specific tightening의 실제 효과 제한적
3. 현재 C&CG가 평균 2.2회 반복으로 수렴 → McCormick tightness가 병목이 아님

**권장 조치:** 구현 대신, 논문에 다음 한 단락 추가:
> "$\gamma^U = S$는 보수적인 상한이며, 생산비와 운송비를 고려하면 더 tight한 bounds가 가능하다. 그러나 C&CG가 평균 2.2회 반복으로 수렴하므로, McCormick tightness가 알고리즘 성능의 병목이 아님을 시사한다."

---

### Reviewer 1 공통 질문 답변

**RQ-C1 (3가지 수정):**
1. Dual derivation 부록 추가 (재현성 문제 해결, 노력 대비 효과 최대)
2. $\epsilon$ 민감도 분석 (10개 seed x 4 $\epsilon$ x 4 정책 = 160 runs 추가)
3. 1단계/2단계 분류 심층 논의 (텍스트 보강만으로 가능)

**RQ-C2 (출판 이유):**
배송 속도-수요 관계를 강건 최적화 안에서 계산 가능한 형태로 정식화한 최초의 시도. 4,800회 실험과 99.75% 수렴률은 실용성 입증.

**RQ-C3 (외적 타당성):**
3단계 전략 — (1) 문헌 기반 파라미터 정당화 [필수, 텍스트만], (2) 규모 다양화 실험 ($|R| \in \{50, 200, 500\}$), (3) 공개 벤치마크(OR-Library 등) 활용.

---

## 2. Reviewer 2 (도메인 전문가) 답변

### RQ2-1: DI 함수 실증 보정 방법론

**Step 1: 실증 결과를 배송 시간-수요 탄성으로 변환**

Fisher et al. (2019) 기반 (7일→5일→3일 매핑):
| 모드 | DI (Fisher 기반) | DI (Marino 기반) | DI (현재 모형, $\kappa=1$) |
|------|-----------------|-----------------|--------------------------|
| m=0 (slow) | 1.000 | 1.000 | 1.000 |
| m=1 (medium) | 1.029 | ~1.25 | 1.500 |
| m=2 (fast) | 1.059 | ~1.60 | 2.250 |

**핵심 발견: 현재 모형이 수요 탄성을 수십 배 과대추정하고 있음** (Fisher 기반 대비). Marino 기반이라 해도 현재 값보다 상당히 작음.

**Step 2: 구체적 권고**
1. 4-5개 실증 논문의 추정치를 표로 정리
2. 각 논문의 배송 시간 단축 → 수요 증가 효과를 이산적 모드(m=0,1,2)에 매핑
3. 이 범위의 하한, 중앙값, 상한으로 3개 DI 시나리오 구성
4. 현재 HD/MD/LD를 실증 기반 시나리오로 대체하거나 병렬 비교

**함수 형태:** 지수함수를 고수하되 밑수를 실증 기반으로 조정하거나, 선형 DI 함수로 한 세트 추가 실험하여 함수형태 강건성 확인.

---

### RQ2-2: DC 수 고정 현상 원인

**파라미터 설정의 문제**일 가능성 높음 (모형 구조적 한계 아님).

**원인:** DC 용량 $\mathcal{U}(6000, 7200)$이 총 수요(~10,000-13,000) 대비 충분. DC 3개면 커버 가능. DC 고정비가 상대적으로 높아 추가 개설의 한계 편익 < 한계 비용.

**DC가 모드에 반응하도록 하는 조정 (효과 순):**
1. **DC 용량 축소** → $\mathcal{U}(3000, 4000)$ [가장 효과적]
2. DC 고정비 축소 (마이크로 풀필먼트 반영)
3. 후보 DC 수 증가 ($|J|=5$ → $|J|=10$)
4. 거리 기반 비용 구조 강화

---

### RQ2-3: 서비스 커버리지 제약의 영향 — 핵심 답변

**결론이 상당히 달라질 것으로 예상.**

| 효과 | 설명 |
|------|------|
| Optimal vs FM2 격차 확대 | FM2는 원거리 고객에도 fast 강제 → 비용 폭증. Optimal은 거리별 차별화. **VoF가 0.2% → 5-15%로 증가 가능** |
| DC-모드 상호작용 활성화 | fast 배송 고객 근처에 DC 추가 배치 필요 → DC 수가 모드에 반응 시작 |
| 유연성 가치 극대화 | 모드 선택이 비균질적(heterogeneous)이 되어 단일 모드 정책의 비효율성 부각 |

**구현:** $\alpha_{jrm} = 0 \text{ if } D_{jr}^2 > \bar{D}_m$ — 간단한 제약 추가. 모형 구조 변경 불필요.

**Reviewer 2 평가: 이 수정이 논문의 기여도를 가장 크게 향상시킬 수 있는 단일 변경.**

---

### RQ2-4: 누락 문헌 우선순위

| 우선순위 | 문헌 카테고리 | 구체적 논문 | 처리 방법 |
|---------|-------------|-----------|----------|
| 최우선 | 서비스 수준-수요 내생적 모형 | So & Song (1998), Fattahi et al. (2018) 확장 | 문헌 리뷰에 체계적 정리 |
| 차선 | DRO vs BRO 정당화 | 기존 강건 최적화 소절에 1-2 문단 | 삽입 |
| 선택적 | 옴니채널/대기행렬 | 1-2편 각주 또는 간략 언급 | 과도한 범위 확대 방지 |

**메타휴리스틱 섹션:** 완전 삭제보다 **대폭 축소**(1.5페이지 → 0.5페이지). 핵심 2-3편만 남기고 절약된 공간을 서비스-수요 모형으로 대체. 섹션 제목을 "Solution approaches for supply chain optimization"으로 변경.

---

### RQ2-5: 양방향 불확실성 분석 보고 형태

**기본 보고 (반드시 포함):**
1. 최악 시나리오에서 $\eta_{rk}^+ = 1$ (수요 증가) vs $\eta_{rk}^- = 1$ (수요 감소) 고객 수의 **평균 비율** — 정책별·DI 시나리오별
2. 수요 증가/감소가 **어떤 고객에 집중**되는지: fast 모드 배정 고객 vs slow 모드 배정 고객

**심화 보고:** $\Gamma$ 변화에 따른 worst-case 구조 변화 (수요 감소 방향 활성화 증가 여부)

**핵심 가치:** 양방향 불확실성이 단순 이론적 일반화가 아닌, 실제로 최적 해에 영향을 미치는 메커니즘임을 입증.

---

### Reviewer 2 공통 질문 답변

**RQ-C1 (3가지 수정):**
1. **서비스 커버리지 제약 추가** (구현 쉽고, VoF 격차 확대로 핵심 기여 극적 강화)
2. **DI 함수의 실증적 앵커링** (표 하나 + 논의 2-3문단)
3. **문헌 검토 재구조화** (메타휴리스틱 축소 + 서비스-수요 모형 추가 + BRO vs DRO 정당화)

**RQ-C2 (출판 이유):**
1. 문제 정의의 독창성 (수요-운영 피드백 루프 포착의 최초 체계적 시도)
2. 실험 규모와 통계적 엄격성 (4,800회, 비모수 검정 병행)
3. 비자명한 통찰 (불확실성↑ → 유연성 가치↑)

**RQ-C3 (외적 타당성):**
1. 실제 지리 데이터 기반 인스턴스 (Census Bureau, 통계청)
2. 실증 문헌 파라미터 앵커링
3. 산업별 시나리오 (식료품 저마진·고빈도, 가전 고마진·저빈도, 의류 중간)

---

## 3. Advisor (지도교수) 답변

### RQ3-1: EJOR 최적화 제목

**비유적 표현은 피하는 것이 안전.** EJOR은 기술적 정체성이 강한 저널.

**최종 제목 추천:**
> "Two-Stage Robust Supply Chain Design with Delivery-Speed-Dependent Demand: A Column-and-Constraint Generation Approach"

이유: (1) "Two-Stage Robust"가 방법론을 즉시 전달, (2) "Delivery-Speed-Dependent Demand"가 novelty 정확 전달, (3) "C&CG"가 OR 키워드.

---

### RQ3-2: "Endogenous" 유지 전략

**"Endogenous"를 유지하되, 명확한 정의를 제공하는 방식이 전략적으로 유리.**

Introduction에서 처음 사용 시 삽입할 정의:
> "We use 'endogenous demand' in the operations management sense: demand is a function of the firm's own decision variables—specifically, transportation mode selection—rather than being exogenously fixed. This is analogous to the endogenous demand treatment in pricing models where demand depends on the firm's price decision (e.g., Petruzzi and Dada, 1999), though here the decision lever is delivery speed rather than price."

**추가 규칙:** "endogenously links" 표현을 논문 전체에서 **3-4회 이상 반복하지 말 것**. Introduction, Contribution, Conclusion에서 각 한 번.

---

### RQ3-3: 실증 데이터 없이 EJOR 수용 가능성

**수용 가능하다.** EJOR은 순수 방법론 논문도 다수 게재. 단, 조건 필요.

**전략 1 (가장 효과적): 실증 문헌 파라미터 직접 차용**

"Parameter Calibration" 서브섹션 추가:

| Mode transition | 현재 모형 ($\kappa=1$) | Fisher et al. (2019) | Marino & Zotteri (2018) |
|---|---|---|---|
| Slow→Medium | +50% | — | — |
| Slow→Fast | +125% | — | — |
| 2-day→7-day equiv. | — | -7.25% (5×1.45%) | -37.5% |

**완벽하게 맞지 않아도 됨.** "같은 order of magnitude"임을 보여주면 충분.

**전략 2: Sensitivity analysis 확장** (functional form + cost ratio 변경 실험)

**전략 1이 전략 2보다 훨씬 중요.** 1-2페이지 추가로 핵심 우려 해소.

---

### RQ3-4: Contribution 톤 조절 — 구체적 리프레이밍

**Contribution 1 (수요-배송속도 연결):**
> "This study extends the supply chain network design literature by incorporating delivery-speed-dependent demand into a profit-maximizing framework. While prior models treat demand as fixed regardless of service level, our formulation captures the empirically documented phenomenon that faster delivery stimulates additional demand, enabling joint optimization of network structure and delivery strategy."

**Contribution 2 (Two-stage robust model):**
> "The proposed two-stage robust optimization model distinguishes between committed strategic decisions and adaptive operational responses, providing a more realistic representation of supply chain decision-making under uncertainty than single-stage robust counterparts."

**Contribution 3 (C&CG):**
> "To solve the resulting model, we develop a tailored C&CG decomposition that exploits the problem's structure through McCormick linearization and binary uncertainty decomposition. Computational experiments on 4,800 instances demonstrate a 99.75% convergence rate with 89.7% of runs completing in under one minute, confirming the approach's practical tractability."

---

### RQ3-5: 제출 전략

**명확하게 (A) Major Revision 후 제출 추천.**

| 이유 | 설명 |
|------|------|
| 첫인상 효과 | Editor/reviewer의 첫 판단이 전체 심사 지배 |
| Desk reject 위험 | EJOR desk reject율 40-50%. 현재 abstract 길이와 "endogenous" 과대 포장이 사유 가능 |
| R&R 횟수 제한 | 대부분 2회까지. (A)는 1회 R&R 가능, (B)는 2회 소진 위험 |
| 총 시간 비용 | (A)에서 1-2개월 투자가 총 소요 시간에서 오히려 유리 |

**예외:** 졸업/연구비 deadline 있으면 (B) 가능. 그 경우에도 최소 초록 축소 + 제목 변경 + endogenous 정의 추가는 필수.

---

### Advisor 공통 질문 답변

**RQ-C1 (3가지 수정):**
1. **초록 축소 + 제목 변경** (desk reject 방지, 작업량 최소·효과 최대)
2. **Parameter Calibration 섹션 추가** (1-2페이지로 가장 큰 약점 해소)
3. **"Endogenous" 정의 명시 + Contribution 톤 조절** (공격 표면 사전 차단)

→ 이 3가지만으로 **reject 확률 30-40% 감소** 가능.

**RQ-C2 (출판 이유):**
1. 실무적 중요성 (배송 속도 경쟁에서 수요 탄력성을 robust하게 다룬 OR 연구 거의 부재)
2. 실험 규모/엄밀성 (4,800건, 다층 통계 검정)
3. 비자명한 통찰 (불확실성↑ → 유연성 가치↑, LD에서만 mode mixing 유의미)

**RQ-C3 (외적 타당성):**
1. [필수] 실증 문헌 기반 파라미터 보정
2. [권장] 공개 데이터셋 (Instacart, JD.com, 통계청 온라인쇼핑동향)
3. [이상적] Industry report 인용 (McKinsey, Capgemini)

---

## 4. Questioner (비판적 질문자) 답변

### RQ4-1: Tautology 해결 — 세 가지 탐색 축

**(1) 운송 비용 비율 극단화 [가장 추천]**
- 현재 $TC_2/TC_0 = 4$배. 이를 8배, 16배, 32배로 확대
- $TC_2 \in \{0.20, 0.40, 0.60, 0.80, 1.00\}$에 대해 LD 시나리오 실험
- **50 seeds x 5 costs = 250 runs 추가**로 충분
- **Breakeven point** 식별: "운송 비용이 X배를 초과하면 빠른 배송의 수익성이 역전"
- 기존 코드에서 $TC_2$ 값만 변경 → 구현 비용 최소

**(2) 용량 제약 강화**
- Plant/DC capacity를 50%, 30%, 20%로 축소
- 빠른 배송의 수요 증가를 충족 불가 → shortage cost 급증 → FM2 우위 소멸
- **통찰:** "충분한 생산 용량이 빠른 배송 전략의 전제 조건"

**(3) 수요 함수 Concave 변형**
- $(3/2)^{\kappa m}$ → $1 + \kappa \cdot \log(1+m)$ 또는 $1 + \kappa \cdot m/(1+m)$
- Concave에서는 m=1과 m=2 차이 축소 → FM1이 최적인 영역 발생 가능

---

### RQ4-2: "Value of Flexibility" 정당화 방법

**방법 A: Concave 수요 함수** — 체감 효과로 일부 고객은 FM1, 다른 고객은 FM2가 최적 → 진정한 "flexibility"

**방법 B: 제품별 차별적 가격** — $S_k \in \{50, 150, 300\}$으로 차별화 → 저가 제품은 느린 배송, 고가는 빠른 배송이 최적

**현 상태 타협안 — 프레이밍 이원화:**
- **Primary finding:** "Value of Fast Delivery" — FM2 vs FM0/FM1 비교 (82.7%)
- **Secondary finding:** "Value of Flexibility" — Optimal vs FM2 비교, 특히 LD 시나리오 (0.2%~0.6%)

→ 82.7%를 "flexibility"의 증거로 쓰지 않고, LD 결과를 더 정직하게 사용.

---

### RQ4-3: 분모 효과 검증

**절대 VoF 계산 결과 (Table 10 데이터 기반):**

| $\Gamma$ | Optimal | FM0 | 절대 VoF | VoF% |
|----------|---------|-----|---------|------|
| 20 | 964,641 | 558,348 | 406,293 | 72.8% |
| 40 | 904,847 | 495,565 | 409,282 | 82.6% |
| 60 | 852,801 | 440,942 | 411,859 | 93.4% |
| 80 | 808,673 | 394,733 | 413,940 | 104.9% |
| 100 | 773,279 | 357,651 | 415,628 | 116.2% |

**핵심 발견: 절대 VoF도 $\Gamma$와 함께 증가** (406K → 416K, +2.3%). 분모 효과만이 아님을 입증.

**추가 권장 지표:**
1. VoF / 총비용 비율 (분모가 비용 기준)
2. 순위 기반 Spearman 상관계수

**핵심 권고:** 절대 VoF 표를 Table 11 옆에 병렬 배치하는 것이 가장 간단하고 효과적.

---

### RQ4-4: 경쟁 반응의 간접적 반영

**학술적으로 허용 가능.** $DI_m^k$를 "경쟁사 대비 상대적 서비스 수준에서 오는 수요 재분배 효과"로 재해석.

Problem Definition에 추가할 문단:
> "The demand increase factor $DI_m^k$ should be interpreted as the net demand effect relative to the prevailing market service level, rather than an absolute increase. In markets where competitors also offer fast delivery, the effective DI would be lower (corresponding to our LD scenario), while in markets where fast delivery provides a competitive differentiation, the effective DI would be higher (corresponding to our HD scenario)."

→ 4개 DI 시나리오가 **다양한 경쟁 강도를 대리(proxy)**하는 것으로 재해석. Game-theoretic 확장 불필요.

---

### RQ4-5: 실무적 기여 재구성

**"고객별 모드 선택 기준"만으로는 부족** (Optimal과 FM2 차이가 0.2%이므로 "거의 없다"가 답이 됨).

**3가지로 재구성:**
1. **Breakeven Analysis Tool** — "운송 비용이 X배 초과 시 빠른 배송 비효율" 정량 기준
2. **불확실성 하 투자 정당화** — "불확실한 시장일수록 빠른 배송 투자가 가치 있다" (반직관적 통찰)
3. **네트워크 설계 가이드라인** — HD 시나리오에서 추가 plant 개설 필요 (3.74 vs 3.04) → "빠른 배송 전략 시 생산 용량 확장 동반 필요"

→ Conclusion의 managerial implications를 이 3가지로 구조화.

---

### Questioner 공통 질문 답변

**RQ-C1 (3가지 수정):**
1. **운송 비용 비율 민감도 (Breakeven Analysis)** — 250 runs 추가로 tautology 해결 + 실무적 기여
2. **절대 VoF 표 추가 + 프레이밍 조정** — 분모 효과 차단 + VoF 해석 이원화
3. **경쟁 환경 해석 문단 추가** — 2-3 문단으로 필수 지적 선제 대응

**RQ-C2 (출판 이유):**
1. 방법론적 기여 (수요-내생적 RO 프레임워크 — 후속 연구의 출발점)
2. 알고리즘적 기여 (McCormick + Binary Decomposition C&CG, 99.75% 수렴)
3. 실증적 기여 (절대 VoF도 $\Gamma$와 함께 증가 → 분모 효과 차감 후에도 유의미)

**RQ-C3 (외적 타당성):**
1. 공개 벤치마크 (Solomon VRPTW, 실제 도시 지리 데이터)
2. **파라미터 캘리브레이션 테이블 [가장 추천]** — 추가 실험 없이 텍스트만으로 가능
3. Out-of-Sample Robustness Test (40개 학습, 10개 검증)

---

# Part 2: 교차 분석 — 4명의 답변 비교

---

## 공통 질문 RQ-C1: "3가지만 수정한다면?" 교차 비교

| 순위 | Reviewer 1 | Reviewer 2 | Advisor | Questioner |
|------|-----------|-----------|---------|-----------|
| 1 | Dual derivation 부록 | **서비스 커버리지 제약** | 초록 축소 + 제목 변경 | **Breakeven analysis** |
| 2 | $\epsilon$ 민감도 분석 | DI 함수 실증 앵커링 | **Parameter calibration** | 절대 VoF 표 + 프레이밍 |
| 3 | Here-and-now 논의 | 문헌 검토 재구조화 | Endogenous 정의 + 톤 조절 | 경쟁 환경 해석 |

### 합의 영역 (3명 이상 동의):
- **DI 함수의 실증적 정당화/캘리브레이션** — R1(함수 대안 실험), R2(실증 앵커링), Advisor(Parameter Calibration), Questioner(캘리브레이션 테이블) → **전원 동의**
- **프레이밍/톤 조절** — Advisor(endogenous 정의+contribution), Questioner(VoF 이원화+경쟁 해석) → 3명 동의

### 독자적 제안 (1명만 주장하나 영향력 큰 것):
- **서비스 커버리지 제약 추가** (R2) — VoF를 0.2% → 5-15%로 확대 가능. 논문 기여 극적 강화
- **Breakeven analysis** (Questioner) — Tautology 문제 직접 해결. 250 runs 추가
- **Dual derivation 부록** (R1) — 재현성 문제. 저널 게재의 최소 요건

---

## 공통 질문 RQ-C2: "출판 이유" 교차 비교

**4명 전원 동의하는 고유 기여:**
> 배송 속도-수요 관계를 강건 최적화 프레임워크에서 계산 가능한 형태로 정식화한 최초의 체계적 시도

**추가 동의 사항:**
- 4,800회 실험의 통계적 엄밀성 (4명 전원)
- C&CG 99.75% 수렴률의 실용성 (R1, R2, Advisor)
- "불확실성↑ → 유연성 가치↑"의 비자명성 (Advisor, Questioner, R2)

---

## 공통 질문 RQ-C3: "외적 타당성" 교차 비교

**전원 1순위 동의:**
> 실증 문헌의 파라미터를 차용한 캘리브레이션 테이블 — 추가 실험 없이 텍스트만으로 가능

| 조치 | R1 | R2 | Advisor | Questioner | 합의 |
|------|----|----|---------|-----------|------|
| 문헌 기반 파라미터 캘리브레이션 | O | O | O | O | **전원** |
| 규모 다양화 실험 | O | O | - | - | 2명 |
| 공개 벤치마크/데이터셋 | O (OR-Library) | O (Census) | O (Instacart) | O (Solomon) | **전원** |
| Industry report 인용 | - | - | O | - | 1명 |
| Out-of-sample test | - | - | - | O | 1명 |
| 산업별 시나리오 | - | O | - | - | 1명 |

---

# Part 3: 통합 액션 플랜 — 우선순위별

4명의 답변을 종합하여, **실행 가능성**과 **논문 개선 효과**를 기준으로 최종 액션 플랜을 제시합니다.

---

## Tier 1: 필수 수정 (텍스트만으로 가능, 추가 실험 불필요)

| # | 액션 | 근거 | 예상 작업량 |
|---|------|------|-----------|
| T1-1 | **초록 250→150단어로 축소** | Advisor: desk reject 방지 | 30분 |
| T1-2 | **제목 변경** → "Two-Stage Robust Supply Chain Design with Delivery-Speed-Dependent Demand: A C&CG Approach" | Advisor: EJOR 최적화 | 10분 |
| T1-3 | **"Endogenous" 정의 명시** (pricing 문헌 analogy + 사용 빈도 3회 이내) | Advisor: 공격 표면 제거 | 30분 |
| T1-4 | **Parameter Calibration 서브섹션 추가** (Fisher, Marino, Park 수치와 DI 비교표) | 전원 동의. 가장 큰 약점 해소 | 2-3시간 |
| T1-5 | **Contribution 톤 조절** (Advisor의 리프레이밍 적용) | Advisor: 과대 포장 제거 | 1시간 |
| T1-6 | **경쟁 환경 해석 문단** 추가 (DI를 경쟁 대비 순 효과로 재해석) | Questioner: 경쟁 부재 비판 선제 대응 | 30분 |
| T1-7 | **절대 VoF 표** 추가 (Table 11 옆에 병렬) | Questioner: 분모 효과 차단 | 1시간 |
| T1-8 | **VoF 프레이밍 이원화** (Primary: Value of Fast Delivery / Secondary: Value of Flexibility) | Questioner: 정직한 해석 | 1시간 |
| T1-9 | **Dual derivation 부록** 추가 (2-3페이지) | R1: 재현성 필수 요건 | 3-4시간 |
| T1-10 | **문헌 검토 재구조화** (메타휴리스틱 1.5p→0.5p + 서비스-수요 모형 추가 + BRO vs DRO 정당화) | R2, Advisor, R1 | 3-4시간 |
| T1-11 | **McCormick tightness 한 단락** 추가 (병목 아님을 언급) | R1: 간단히 해소 | 15분 |
| T1-12 | **Here-and-now 분류 논의 보강** (적합한 산업 맥락 열거 + limitations 명시) | R1: 텍스트만으로 대응 | 1시간 |
| T1-13 | **Managerial implications 3가지로 구조화** (Breakeven/불확실성 투자/네트워크 설계) | Questioner | 1시간 |

**Tier 1 총 예상 작업량: 약 2-3일**

---

## Tier 2: 핵심 추가 실험 (코드 수정 최소, 파라미터 변경만)

| # | 액션 | 근거 | 추가 실험 규모 |
|---|------|------|-------------|
| T2-1 | **운송 비용 Breakeven Analysis** ($TC_2 \in \{0.20, 0.40, 0.60, 0.80, 1.00\}$, LD 시나리오) | Questioner: tautology 해결. 가장 강력한 실무적 기여 | 250 runs |
| T2-2 | **$\epsilon$ 민감도 분석** ($\epsilon \in \{10, 100, 400, 800\}$, 10 seeds) | R1: Optimal vs FM2 신뢰성 확보 | 160 runs |
| T2-3 | **DI 함수 대안 실험** (선형 + 로그, 앵커 포인트 통일, 10 seeds) | R1: 함수형태 강건성 | 320 runs |
| T2-4 | **서비스 커버리지 제약** ($\alpha_{jrm} = 0$ if $D_{jr}^2 > \bar{D}_m$) | R2: VoF 격차 확대, 논문 기여 극적 강화 | 기존 실험 재실행 |

**Tier 2 총 추가 실험: 약 730 runs + 기존 재실행**

---

## Tier 3: 선택적 보강 (시간 여유 있을 때)

| # | 액션 | 근거 |
|---|------|------|
| T3-1 | 규모 확장 실험 ($|R| \in \{50, 500\}$) | R1: scalability 검증 |
| T3-2 | SAA와의 out-of-sample 비교 | R1: RO의 가치 입증 |
| T3-3 | 양방향 불확실성 구조 분석 ($\eta^+$ vs $\eta^-$ 패턴) | R2: 양방향 불확실성의 실질적 영향 |
| T3-4 | 용량 제약 강화 실험 (capacity 50%, 30%) | Questioner: tautology 추가 해결 |
| T3-5 | 공개 벤치마크 기반 인스턴스 (Solomon/실제 도시) | 전원: 외적 타당성 |
| T3-6 | 산업별 시나리오 (비용/가격 차별화) | R2: 일반화 가능성 |

---

## 최종 권장 실행 순서

```
Week 1: Tier 1 전체 (텍스트 수정)
Week 2: T2-1 (Breakeven) + T2-2 (epsilon)
Week 3: T2-3 (DI 대안) + T2-4 (커버리지 제약)
Week 4: 결과 통합 + 최종 교정
→ 제출
```

**최소 버전 (1주일 내 제출 필요 시):**
T1-1 ~ T1-6 + T1-9 만 수행 → desk reject 방지 + 핵심 약점 텍스트 대응
