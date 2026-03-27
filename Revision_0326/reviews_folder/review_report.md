# 논문 사전 심사 종합 보고서

**논문:** A Robust Optimization Framework for Logistics Management: Balancing Fast Delivery and Demand Uncertainty
**저자:** Donghwan Kim (Yonsei University)
**심사일:** 2026-03-21
**심사 구성:** 4인 독립 리뷰 (방법론 리뷰어, 도메인 리뷰어, 지도교수 역할, 비판적 질문자)

---

# Part 1: 종합 리뷰 보고서

---

## 1. 전체 판정 요약

| 리뷰어 | 역할 | 판정 |
|--------|------|------|
| Reviewer 1 (Methodology) | 수리 모델링, 알고리즘, 계산 실험 | Major Revision |
| Reviewer 2 (Domain) | 공급망, 라스트마일, 수요 모델링 | Major Revision |
| Advisor | 전략적 포지셔닝, 글쓰기, 프레이밍 | Major Revision 1회 후 제출 권장 |
| Questioner | 가정 도전, 논리적 허점, 근본 질문 | 세 가지 근본 약점 지적 |

**종합 판정: Major Revision 필수. 현 상태로 제출 시 R&R 또는 Reject 가능성 높음.**

---

## 2. 긍정적 평가 (강점)

### 2.1 연구 주제의 시의적절성과 실무적 중요성

- 배송 속도를 단순 비용 요소가 아닌 **수요 창출의 전략적 수단**으로 모델링한 관점은 기존 문헌 대비 의미 있는 기여이다. (Reviewer 1, 2, Advisor 공통)
- 서론에서 제기하는 연구 갭 — "기존 모델이 수요를 고정으로 취급한다" — 은 설득력 있고, 실무적으로 중요하다. (Advisor)

### 2.2 대규모 실험 설계와 통계적 엄밀성

- **4,800회의 대규모 수치 실험**(50 datasets × 4 DI scenarios × 4 policies + Gamma sensitivity)은 결과의 신뢰성을 높인다. (Reviewer 1, 2)
- Friedman test, Wilcoxon signed-rank test, paired t-test 등 **다층적 통계 검정**을 수행하여 결과의 통계적 유의성을 철저히 검증했다. (Reviewer 1)
- 4개 DI 시나리오(LD, MD, HD, VHD)와 5개 Gamma 수준으로 **체계적인 sensitivity analysis**를 수행했다. (Reviewer 2)

### 2.3 알고리즘 성능

- C&CG 알고리즘의 **99.75% 수렴률**은 인상적이다. (Advisor, Reviewer 1)
- **89.7%의 인스턴스가 1분 이내**에 수렴하여, 실무적 적용 가능성을 보여준다. (Advisor)
- McCormick linearization과 binary uncertainty decomposition을 문제에 맞게 tailoring한 점은 기술적 기여이다. (Reviewer 1)

### 2.4 문제 정의와 수식 전개

- Section 3(Preliminaries)의 표기법 표가 매우 상세하고 체계적이다. (Advisor)
- Section 4(Formulation)의 수식 전개가 논리적이며 잘 구성되어 있다. (Advisor)
- 연구 갭과 기여의 구분이 명확하고, 서론-결론 간 일관성이 유지된다. (Advisor)

### 2.5 Gamma Sensitivity Analysis

- 불확실성 예산($\Gamma$) 변화에 따른 유연성 가치 변화를 체계적으로 분석한 점은 강건 최적화 연구에서 중요한 기여이다. (Reviewer 1, 2)
- "유연성의 가치가 불확실성과 함께 증가한다"는 발견은 실무적으로 의미 있는 통찰이다. (Advisor)

---

## 3. 부정적 평가 (약점) — 우선순위별 정리

---

### 3.1 [Critical] 수요 증가 함수 $DI_m^k = (1.5)^{\kappa \cdot m}$의 정당화 부재

**지적자:** Reviewer 1, Reviewer 2, Advisor, Questioner (4/4 전원)

**문제:**
- 이 함수 형태(지수함수)를 선택한 이론적/실증적 근거가 전혀 제시되지 않았다.
- 서론에서 인용된 실증 연구들은 지수적 관계를 지지하지 않는다:
  - Fisher et al. (2019): "영업일 1일당 1.45% 매출 감소" → **선형 관계**에 가까움
  - Marino & Zotteri (2018): "2일→7일 시 37.5% 수요 감소" → 선형/감소적 관계 시사
- 밑수(base) 1.5의 선택 근거도 불분명하다.
- $\kappa$ 파라미터([0, 1] 범위)가 어떤 실증 데이터에 기반하여 보정(calibrate)된 것인지 불분명하다.
- HD 시나리오에서 $DI_2^k = (1.5)^{1.0 \times 2} = 2.25$, 즉 **125% 수요 증가**인데, 이것이 실증적으로 지지되는 수준인가? (Questioner)

**Advisor:** "endogenous"라고 부르지만 실제로는 exogenous parameter를 곱하는 것이다. 진정한 endogenous demand는 균형 가격이나 게임 이론적 상호작용이 있어야 한다.

**Reviewer 1:** sensitivity analysis에서 $\kappa$는 변화시키지만, **함수형 자체**에 대한 robustness check이 없다.

**필요 조치:**
1. 대안적 수요-속도 함수(선형, 로그, piecewise linear, S-curve)에 대한 sensitivity analysis 수행
2. Fisher et al.의 실증 수치와 모형 DI 파라미터가 생성하는 수요 변화 폭을 직접 정량 비교
3. "endogenous demand" 표현을 "delivery-speed-dependent demand" 또는 "demand-responsive"로 완화
4. 실제 산업 데이터로 보정 가능성에 대한 논의 강화

---

### 3.2 [Critical] 순환 논증 / 결론의 자명성 (Tautology)

**지적자:** Questioner, Advisor, Reviewer 2 (3/4)

**문제:**
- 모형 자체가 "빠른 배송 → 수요 증가" 메커니즘을 내장하고 있으므로, "빠른 배송이 최적"이라는 결론은 가정에서 거의 자동으로 도출된다.
- "빠른 배송이 좋다"는 것을 4,800번의 최적화로 증명한 것인가? 단순 비용-편익 분석으로도 동일한 결론에 도달할 수 있지 않은가? (Questioner)
- Optimal과 FM2의 이익 차이가 **평균 0.2% (절대값 945)**에 불과하다. 이는 "Value of Flexibility"가 아니라 사실상 "Value of Fast Delivery"를 보여주는 것이다. (Questioner, Reviewer 2)
- FM0 대비 82.7% 차이는 유연성이 아니라 **단순히 빠른 배송 모드의 효과**일 뿐이다. (Questioner)

**Advisor:** 핵심 insight는 "flexibility의 가치가 불확실성이 클수록 증가한다"는 것과 "LD scenario에서만 mode mixing이 의미 있다"는 것 — 이 두 가지가 non-trivial하므로 더 강조해야 한다.

**Questioner:** 더 가치 있는 연구 질문은 **"언제 빠른 배송이 최적이 아닌가?"**이다.

**필요 조치:**
1. 빠른 배송이 최적이 **아닌** 조건을 명시적으로 탐색 (높은 운송비, 낮은 수요 민감도, tight capacity)
2. LD 시나리오에서의 선택적 다운그레이드 패턴 심층 분석
3. $\Gamma$ 증가 시 유연성 가치 증가의 메커니즘 해부
4. VoF의 주요 비교 기준을 FM0 → FM1 또는 FM2로 변경 검토
5. 프레이밍 재검토: "Value of Flexibility" vs "Value of Fast Delivery"

---

### 3.3 [Critical] 실증 데이터 / 외적 타당성 부재

**지적자:** Advisor, Reviewer 1, Reviewer 2, Questioner (4/4 전원)

**문제:**
- **Advisor:** Reject 사유 1순위. 수요-속도 관계가 가정에 기반하며, 실제 데이터로 calibration되지 않았다.
- **Reviewer 1:** 모든 실험이 $|K|=3, |I|=3, |J|=5, |R|=200$의 **단일 규모**에서 수행됨. Scalability 주장 불가.
- **Reviewer 2:** 합성 데이터(Uniform, Gaussian 분포)가 실제 공급망의 공간적 분포, 비용 구조, 수요 패턴을 대표하는지 의문.
- **Questioner:** 82.7% 이익 개선이 현실적인가, 모형 파라미터의 산물(artifact)인가?

**추가 현실성 문제 (Reviewer 2):**
- DC 수가 모든 정책에서 약 2.86~2.88개로 사실상 고정 → DC 입지 결정이 배송 모드 선택과 독립적
- 단일 제품 가격($S = 150$)은 제품 카테고리 간 수익성 차이를 무시
- Gaussian 분포 기반 고객 위치는 도시-교외-농촌의 실제 인구 분포 패턴을 반영하지 못함

**필요 조치:**
1. 실제 데이터 case study 1개라도 추가 (가장 효과적인 개선)
2. 최소한 규모 확장 실험 ($|R| = 50, 500, 1000$) 추가
3. DI 파라미터와 실증 문헌 수치의 정량적 대조표 제시
4. Limitations에 실증 데이터 부���를 솔직하게 명시

---

### 3.4 [Major] C&CG 수렴 허용 오차 $\epsilon = 800$ 문제

**지적자:** Reviewer 1, Reviewer 2, Questioner (3/4)

**문제:**
- $\epsilon = 800$은 절대 갭 기준으로, Optimal vs FM2의 평균 VoF(945)와 비교하면 매우 크다.
- VoF의 상당 부분이 수렴 오차 범위 내에 있음을 의미한다.
- 음의 VoF(-308) 발생은 단순 수치 정밀도 문제라기보다 dual formulation이나 McCormick 구현의 잠재적 오류 가능성도 시사한다.
- seed 38에서 비수렴이 집중 발생하는 이유에 대한 진단 부재. (Reviewer 1)

**필요 조치:**
1. 대표 인스턴스(10개)에서 $\epsilon \in \{10, 100, 400, 800\}$으로 sensitivity analysis 수행
2. Optimal vs FM2 비교에서 수렴 오차의 영향을 솔직히 인정
3. seed 38 인스턴스의 특성 분석

---

### 3.5 [Major] Dual Reformulation 유도 과정 누락

**지적자:** Reviewer 1

**문제:**
- "내부 최대화 문제의 dual을 취한다"고 언급하지만, 실제 이중 문제의 목적함수, 이중 변수의 정의, 이중 가능 영역의 제약조건이 명시적으로 제시되지 않았다.
- $\gamma_{rk}$가 어떤 제약조건에 대응하는 이중 변수인지, 상하한($\gamma^L = -(S+SC)$, $\gamma^U = S$)이 어떻게 도출되는지의 유도 과정이 필요하다.
- McCormick linearization은 올바르게 적용된 것으로 보이나, 이중 문제 자체가 제시되지 않아 검증 불가.
- 재현성(reproducibility) 측면에서 심각한 문제.

**필요 조치:**
1. 부록에 이중 변수 정의, 목적함수, 제약조건을 포함한 완전한 dual derivation 추가
2. McCormick 변수 $p_{rk}^+$, $p_{rk}^-$의 도입 과정을 단계별로 제시

---

### 3.6 [Major] SP/DRO와의 비교 실험 부재

**지적자:** Reviewer 1, Reviewer 2, Questioner (3/4)

**문제:**
- 문헌 리뷰에서 DRO(Gao et al., 2024)와 SP 접근법들을 언급하면서도, 실험적 비교가 전혀 없다.
- RO는 worst-case에 초점을 맞추는데, 실제 기업은 기대값(expected profit)을 최적화하는 경우가 더 많다. (Questioner)
- Worst-case 시나리오에서의 최적 전략이 평균적으로도 최적인지 알 수 없다.
- 왜 BRO를 DRO보다 선호하는지에 대한 명시적 정당화가 없다. (Reviewer 2)

**필요 조치:**
1. 동일 인스턴스에서 SAA(Sample Average Approximation)와의 out-of-sample 성능 비교
2. RO를 선택한 이유(예: 수요 분포를 모른다)를 더 명확히 정당화

---

### 3.7 [Major] 배송 모드의 Here-and-Now 분류 정당성

**지적자:** Reviewer 1, Reviewer 2, Questioner (3/4)

**문제:**
- 배송 모드 선택($\alpha_{jrm}$, $\beta_{rm}$)을 1단계(here-and-now) 결정으로 분류했으나, 현대 물류에서는 주문별 동적 결정이 일반적 (Amazon, Coupang).
- "계약적 물류 계약" 때문이라는 정당화는 불충분하다.
- 2단계(wait-and-see) 변수로 전환하면 모델 구조와 결과가 근본적으로 달라질 수 있다. (Reviewer 1)
- 이 가정이 "유연성의 가치"를 인위적으로 높이거나 낮출 수 있다. (Questioner)

**필요 조치:**
1. 이 가정의 현실적 근거를 강화하거나, wait-and-see 모드 선택과의 비교 제시
2. 본문에서 이 가정이 결과 해석에 미치는 영향을 논의

---

### 3.8 [Major] McCormick 완화의 Tightness 분석 부재

**지적자:** Reviewer 1

**문제:**
- McCormick envelope는 변수의 상하한에 의존하며, 경계가 느슨할수록 완화도 느슨해진다.
- $\gamma^U = S$에 대해 "판매 가격이 수요 한 단위의 최대 한계 가치"라고 정당화하지만, 생산비/운송비를 고려하면 더 tight한 bounds를 얻을 수 있다.
- McCormick gap이 수렴 속도와 해의 질에 미치는 영향에 대한 분석 부재.

**필요 조치:**
1. 대안적 bounds를 사용한 실험으로 tightness 영향 분석
2. McCormick gap과 수렴 속도의 관계 보고

---

### 3.9 [Major] 운송비 구조의 비현실적 단순화

**지적자:** Reviewer 1, Reviewer 2

**문제:**
- 라스트마일 운송비가 $D_{jr}^2 \cdot TC_m \cdot \alpha_{jrm} \cdot A_{jr}^k$로, **배송량에 비례**하게 모델링됨.
- 실제 라스트마일에서는 **고정비(차량 파견 비용)**가 크고, 규모의 경제가 존재.
- **차량 라우팅(VRP)**이 고려되지 않아 동일 경로 순회 효과 미반영.
- 운송 모드별 **용량 제약** 미반영: 오토바이, 드론 등은 적재 용량 제한적.
- **서비스 커버리지** 가정 비현실적: 모든 모드가 모든 고객에게 이용 가능.
- **시간 차원의 부재:** 배송 "속도"를 모델링하면서 실제 배송 시간(시간/일 단위)이 모형에 명시적으로 포함되지 않음.

---

### 3.10 [Major] 3계층 모형과 라스트마일 초점 간 불일치

**지적자:** Reviewer 1, Reviewer 2

**문제:**
- 서론에서 "라스트마일 배송"의 중요성을 강조하면서, 실제 모형은 공장-DC-고객 3계층 전체를 포괄.
- 배송 모드 선택은 DC-고객 구간에만 적용되지만, 1계층(공장-DC) 결정이 라스트마일 모드 선택의 효과를 희석할 수 있다.
- Reviewer 1: "three-echelon"이라 하지만 실제로는 plants → DCs → customers 구조이며, 전통적 의미의 3단계보다는 **2단계 허브 네트워크**에 가깝다.

**필요 조치:**
1. 라스트마일 배송 비용이 전체 공급망 비용에서 차지하는 비율을 분해 보고
2. 용어 사용의 정확성 재검토 (three-echelon vs two-stage hub network)

---

### 3.11 [Moderate] 누락된 관점들

**지적자:** Questioner, Reviewer 2

| 누락된 관점 | 설명 | 영향 |
|------------|------|------|
| **경쟁 반응** | 서론에서 "competitive redistribution of demand" 언급하면서 모형에는 경쟁 미반영. 경쟁사도 빠른 배송 제공 시 수요 증가 효과 상쇄 | 핵심 전제 훼손 가능 |
| **반품/고객서비스 비용** | 빠른 배송 → 반품률 증가, 배송 품질 저하 가능. 숨겨진 비용이 순이익 감소시킬 수 있음 | 이익 과대 추정 가능 |
| **환경적 영향** | Literature review에서 환경 연구 다수 인용하면서 자체 모형에는 탄소 배출 미포함 | 문헌 리뷰와 불일치 |
| **과잉 공급 비용** | Revenue가 $S \cdot (\tilde{d}_{rk} - u_{rk})$로 정의. 과잉 생산/배송의 폐기/보유 비용 미모형화 | 빠른 배송에 유리하게 왜곡 가능 |
| **용량 제약의 실질적 바인딩** | 거의 모든 경우 빠른 배송이 최적 → 용량 제약이 느슨하여 증가된 수요를 항상 수용 가능하기 때문일 수 있음 | 결론의 일반화 제한 |
| **신선식품 부패성** | 서론에서 신선식품 분야 관련성 언급하면서 제품 부패성/온도 관리 비용 미포함 | 주장과 모형 간 불일치 |

---

### 3.12 [Moderate] VoF% 증가의 분모 효과 (Denominator Effect)

**지적자:** Questioner

**문제:**
- VoF%는 $\frac{Z^*_{Optimal} - Z^*_{FM}}{|Z^*_{FM}|} \times 100$으로 정의.
- $\Gamma$ 증가 시 FM0의 이익이 급격히 감소(분모 축소) → 절대적 VoF 증가 없이도 VoF%가 증가할 수 있음.
- "유연성의 가치가 불확실성과 함께 증가한다"는 결론이 **분모 효과의 산물**일 수 있다.

**필요 조치:**
1. 절대적 VoF(금액)의 $\Gamma$ 민감도를 함께 보고
2. VoF%와 절대 VoF를 병행 분석

---

### 3.13 [Moderate] FM0 비교 기준의 적절성

**지적자:** Questioner

**문제:**
- FM0의 이익이 모든 DI 시나리오에서 정확히 동일한 값(495,565).
- 모형 설계상 당연한 결과(느린 배송은 DI 영향 없음)이지만, FM0을 비교 기준으로 사용하면 VoF가 인위적으로 커지는 효과.

**필요 조치:**
- VoF의 주요 비교 기준을 FM0이 아닌 FM1이나 FM2로 설정하는 것이 더 공정
- 또는 FM0을 "���현실적 baseline"으로 명시적으로 규정

---

## 4. 글쓰기 / 구조 문제

### 4.1 초록 (Abstract)

**지적자:** Advisor

- 현재 약 250단어 → **150-180단어로 축소** 필요.
- 실험 설계 세부사항("comprising 50 datasets, four demand-sensitivity scenarios, and four mode-selection policies at a fixed uncertainty budget (Experiment 1, 800 runs) plus a sensitivity analysis over five uncertainty budget levels (Experiment 2, 4000 runs)") 삭제.
- 핵심 결과(82.7%, 30.3%, gamma sensitivity)만 유지.

### 4.2 서론 (Introduction)

**지적자:** Advisor

- 너무 길다 (3페이지 이상 → **2페이지 이내**로 압축).
- 같은 메시지("faster delivery increases demand")를 여러 번 다른 문헌으로 반복.
- Research gaps와 Contributions를 subsection으로 두는 구조는 좋으나, 본문 자체가 과도.

### 4.3 문헌 리뷰 (Literature Review)

**지적자:** Advisor, Reviewer 1, Reviewer 2 (3/4)

- **Metaheuristic 섹션(2.3) 불필요**: 본 논문은 exact method(C&CG) 사용. 한 페이지 이상 할애 → 논문 초점 흐림.
- 대신 C&CG/Benders decomposition 문헌, 수요 내생성(endogenous demand) OR 문헌 보강 필요.
- 서비스 수준-수요 모형, 가격-수요 탄성 모형 관련 문헌 검토 부족 (Reviewer 2).

### 4.4 소제목 문제

**지적자:** Advisor

- Section 4.3.1 "Critical: correct revenue formulation" → 코드 문서화 스타일이지 논문 스타일이 아님.
- "Revenue formulation" 또는 유사한 학술적 표현으로 변경.

### 4.5 Table 수 과다

**지적자:** Advisor

- 12개 이상의 테이블 → 본문에 **8개 이내**, 나머지는 appendix로 이동.

### 4.6 제목과 키워드

**지적자:** Advisor

**현재 제목:** "A Robust Optimization Framework for Logistics Management: Balancing Fast Delivery and Demand Uncertainty" → **너무 일반적**. "Logistics Management"가 너무 넓고, 핵심인 transportation mode selection과 demand uplift가 드러나지 않음.

**대안 제목 제안:**
- "Delivery Speed as a Demand Lever: A Two-Stage Robust Optimization Approach for Last-Mile Mode Selection"
- "Balancing Speed and Uncertainty in Last-Mile Delivery: A Robust Supply Chain Design with Demand-Responsive Mode Selection"

**현재 키워드:** Robust optimization, Fast delivery, Demand uncertainty, Logistics management

**권장 키워드:** Robust optimization, Last-mile delivery, Transportation mode selection, Demand uncertainty, Column-and-constraint generation

### 4.7 기타 글쓰기 문제

| 항목 | 설명 |
|------|------|
| 시간 차원 불명확 | 단일 기간 vs 다기간, 주문 비용이 기간별인지 일회성인지 모호 (Reviewer 1) |
| 다중 비교 보정 미적용 | 12개 개별 검정에 Bonferroni/Holm 보정 필요 (Reviewer 2) |
| Three-echelon 용어 부정확 | 실질적으로 2단계 허브 네트워크에 가까움 (Reviewer 1) |
| Table 1 자기 홍보 | "This paper"를 다른 논문과 동일 행에 배치하는 것이 self-promotional (Reviewer 1) |
| 영어 교정 | 전반적으로 양호하나 일부 문장 과도하게 길고 복잡 (Advisor) |
| bibliographystyle | cas-model2-names는 Elsevier CAS 템플릿용. 타겟 저널 스타일 확인 필요 (Advisor) |
| 단독 저자 | Reviewer에게 더 엄격한 심사를 받는 경향. 공동연구자 추가 고려 (Advisor) |

### 4.8 인용 오류

**지적자:** Reviewer 2

| 문제 | 설명 |
|------|------|
| Park et al. 연도 불일치 | bib에서 출판연도 2025이나 본문에서 \citep{park2024targeting} 사용 → 연도 불일치 발생 가능 |
| Fotouhi 재인용 | "57-69% of Generation Z consumers" 통계가 원문에서 Cheng (2018)에서 재인용된 것. 원저자 직접 인용 권장 |
| 분야 불일치 인용 | tan2022two, zhang2023exploiting은 전력 시스템/에너지 분야 논문. 공급망 맥락 관련성 불분명 |
| bib 미사용 항목 | salari2022, oyama2024 등이 bib에 있으나 본문 미인용. DI 함수 정당화에 활용 가능 |

---

## 5. 누락된 문헌 / 문헌 검토 공백

**지적자:** Reviewer 2

| 분야 | 추천 문헌/방향 |
|------|---------------|
| 서비스 수준-수요 내생적 모형 | So & Song (1998, Management Science); Li & Lee (1994) 등 가격-서비스 경쟁 모형 |
| 옴니채널 풀필먼트 | Bayram & Baykal-Gursoy (2023); Acimovic & Graves (2015) |
| 대기행렬 기반 수요-속도 모형 | Cachon & Harker (2002); Allon & Federgruen (2007) |
| 내생적 불확실성 OR 문헌 | Goel & Grossmann (2006) (Reviewer 1 추가 제안) |
| 배송 시간 약속 실증 연구 | Salari et al. (2022), Oyama et al. (2024) — bib에 있으나 미인용 |

---

## 6. 타겟 저널 권장

**지적자:** Advisor

| 순위 | 저널 | 적합 이유 | 현 상태 가능성 |
|------|------|----------|---------------|
| 1순위 | European Journal of Operational Research (EJOR) | 모델링 + 실험 중심, RO + SC 조합 적합 | Major revision 후 가능 |
| 2순위 | Computers & Operations Research | 알고리즘 기여 있으므로 적합 | Major revision 후 가능 |
| 3순위 | Transportation Research Part E | 라스트마일 배송, 배송 속도 주제 적합 | Major revision 후 가능 |
| 부적합 | Management Science, Operations Research | 이론적 기여 부족, 실증 데이터 없음, novelty incremental | 현 상태로는 어려움 |

---

## 7. Reviewer가 Reject할 가능성이 높은 이유 Top 4

**지적자:** Advisor

1. **실증적 검증 부재:** 수요-속도 관계가 가정에 기반. "이 모델이 실제로 의미 있는가?"
2. **결과의 자명성(triviality):** "빠른 배송이 수요를 늘리니 빠른 배송이 최적이다" → 모델 없이도 예측 가능
3. **Endogenous demand 과대 포장:** 단순 multiplicative factor를 "endogenous"라고 부르는 것에 대한 비판
4. **문제 규모의 한계:** $|R|=200$이 실제 supply chain에 비해 작음

---

# Part 2: 리뷰어에게 역으로 질문할 사항

---

## Reviewer 1 (방법론 전문가)에게

### RQ1-1: 수요 함수 대안 설계
> DI 함수의 대안적 형태로 선형, 로그, S-curve를 제안하셨습니다. 이들 각각의 구체적 함수식을 제안해주실 수 있습니까? 특히 파라미터를 어떻게 설정해야 현재 지수 함수와 공정한 비교가 가능할까요? (예: 동일한 $m=2$에서의 수요 증가 범위를 맞추는 방식?)

### RQ1-2: $\epsilon$ 민감도 분석의 우선순위
> $\epsilon \in \{10, 100, 400, 800\}$ 실험을 제안하셨는데, 작은 $\epsilon$에서 수렴 시간이 크게 증가할 수 있습니다. 시간 제한(예: 1시간)을 두고 수렴하지 않는 경우는 어떻게 처리하는 것이 적절할까요? 또한 $\epsilon$ 민감도 분석과 규모 확장 실험 중 어느 것이 논문의 기여를 더 강화할까요?

### RQ1-3: SP/DRO 비교의 범위
> SAA 또는 DRO와의 비교를 제안하셨습니다. 공정한 비교를 위해 DRO의 ambiguity set을 어떻게 설정하는 것이 적절할까요? Moment-based, Wasserstein, 또는 다른 형태를 권장하시나요? 그리고 비교 시 in-sample 성능과 out-of-sample 성능 중 어느 쪽에 초점을 맞춰야 할까요?

### RQ1-4: Dual Derivation의 범위
> 부록에 완전한 dual derivation을 추가하라고 하셨는데, 어느 수준까지 상세하게 제시해야 할까요? Strong duality 조건 확인부터 시작해야 하나요, 아니면 LP dual의 표준 결과를 전제하고 문제 특화된 유도만 보여주면 충분할까요?

### RQ1-5: McCormick Tightness 개선
> $\gamma^U = S$보다 tighter한 bound로 $S - F_{ki} - \text{transport cost} - h_j/2$를 제안하셨습니다. 이 bound가 $(i, j, r, m)$ 조합에 따라 달라지면, 변수별로 다른 McCormick envelope을 적용해야 합니다. 이것이 구현 복잡도와 수렴 속도에 미치는 trade-off를 어떻게 평가하시나요?

---

## Reviewer 2 (도메인 전문가)에게

### RQ2-1: 현실적 DI 함수 보정
> DI 함수의 실증적 보정(calibration)을 권장하셨습니다. 실제 산업 데이터에 접근하기 어려운 상황에서, 기존 실증 논문(Fisher et al., Marino & Zotteri, Park et al.)의 결과를 활용하여 DI 함수를 보정하는 구체적인 방법론을 제안해주실 수 있습니까? 예를 들어, Fisher의 "1일당 1.45%"를 이 모형의 이산적 모드 선택($m \in \{0, 1, 2\}$)에 어떻게 매핑할 수 있을까요?

### RQ2-2: DC 수 고정 현상의 해석
> 모든 정책에서 DC 수가 2.86~2.88개로 고정적이라고 지적하셨습니다. 이것이 파라미터 설정(DC 용량, 고정비)의 문제인지, 아니면 모형 구조적 한계인지 어떻게 판단하시나요? DC 수가 배송 모드에 반응하도록 하려면 어떤 파라미터를 조정해야 할까요?

### RQ2-3: 서비스 커버리지 제약 도입
> 모든 모드가 모든 고객에게 이용 가능하다는 가정의 비현실성을 지적하셨습니다. 거리 기반 서비스 커버리지 제약(예: 당일배송은 DC에서 30km 이내)을 추가하면 모형의 결론이 근본적으로 달라질 것으로 예상하시나요? 이 경우 "유연성의 가치"가 더 두드러질까요?

### RQ2-4: 누락 문헌의 우선순위
> 여러 문헌 공백을 지적해주셨습니다 (서비스-수요 모형, 옴니채널, 대기행렬 기반 등). 지면 제약 하에서 가장 우선적으로 추가해야 할 문헌 카테고리는 무엇이며, 기존 메타휴리스틱 섹션을 대체해야 할까요?

### RQ2-5: 양방향 불확실성 분석 방법
> 최악 시나리오에서 $\eta_{rk}$가 어떤 방향으로 실현되는지 분석하라고 제안하셨습니다. 구체적으로 어떤 형태의 보고를 기대하시나요? (예: 수요 증가/감소 고객 비율, 지리적 패턴, 제품별 패턴 등?)

---

## Advisor (지도교수)에게

### RQ3-1: 제목 변경의 전략적 고려
> 두 가지 대안 제목을 제안해주셨습니다. EJOR을 1순위 타겟으로 할 때, "Demand Lever" 같은 비유적 표현이 OR 저널에서 긍정적으로 받아들여질까요? 아니면 더 기술적인 제목이 적절할까요?

### RQ3-2: "Endogenous" 표현 대체 전략
> "Endogenous demand"를 "delivery-speed-dependent demand"로 완화하라고 하셨습니다. 하지만 이렇게 하면 논문의 novelty claim이 약해질 수 있습니다. "Endogenous"를 유지하되 명확한 정의를 제공하는 방식과, 완전히 다른 용어를 쓰는 방식 중 어느 쪽이 전략적으로 더 유리할까요?

### RQ3-3: 실증 데이터 없이 수용 가능성
> Case study 추가를 가장 효과적인 개선으로 제안하셨습니다. 만약 실제 데이터 확보가 현실적으로 불가능한 경우, 실증 데이터 없이도 EJOR 수준에서 수용될 수 있는 대안적 전략이 있을까요? (예: 더 광범위한 sensitivity analysis, 실증 문헌의 파라미터를 직접 차용 등)

### RQ3-4: Contribution 톤 조절의 구체적 방향
> "First study to..." 대신 "This study contributes by..."로 톤 조절을 권장하셨습니다. 현재 3개의 contribution bullet 중 각각을 어떻게 리프레이밍하면 겸손하면서도 기여를 효과적으로 전달할 수 있을까요?

### RQ3-5: Major Revision 후 제출 vs 현 상태 제출 + R&R
> "Major revision 1회 후 제출 권장"이라고 하셨습니다. 전략적으로, (A) 모든 major concern을 해결한 후 제출하는 것과 (B) 현 상태로 제출하여 실제 reviewer 피드백을 받은 후 수정하는 것 중, 어느 쪽이 최종 accept 확률을 더 높일까요?

---

## Questioner (비판적 질문자)에게

### RQ4-1: Tautology 해결 방안
> 순환 논증 문제를 가장 날카롭게 지적하셨습니다. "빠른 배송이 최적이 아닌 조건"을 탐색하라고 하셨는데, 구체적으로 어떤 파라미터 공간을 탐색해야 이 tautology를 깰 수 있을까요? (예: $TC_m$ 비율을 극단적으로 높이기? capacity를 tight하게 설정? 수요 증가 함수를 concave로 변경?)

### RQ4-2: "Value of Flexibility" 재정의
> Optimal vs FM2 차이가 0.2%이므로 "Value of Flexibility"가 아닌 "Value of Fast Delivery"라고 하셨습니다. 만약 저자가 "Value of Flexibility"를 유지하고 싶다면, 이를 정당화할 수 있는 실험 설계나 시나리오가 있을까요? (예: 수요 함수가 concave여서 최적이 중간 속도일 때?)

### RQ4-3: 분모 효과 검증 방법
> VoF%의 $\Gamma$ 민감도가 분모 효과일 수 있다고 지적하셨습니다. 이를 검증하기 위해 절대 VoF 외에 어떤 지표를 추가로 보고하면 좋을까요? (예: VoF/총비용, VoF/총수익 등?)

### RQ4-4: 경쟁 반응의 간접적 반영
> 경쟁 모형의 부재를 지적하셨습니다. Game-theoretic 확장 없이도, 경쟁의 영향을 간접적으로 반영할 수 있는 방법이 있을까요? (예: DI 함수의 수요 증가 효과를 "경쟁 환경에서의 순 효과"로 해석하는 것이 학술적으로 허용 가능할까요?)

### RQ4-5: 실무적 기여의 구체화
> "어떤 실무적 의사결정이 이 논문 덕분에 달라지는가?"라는 질문을 하셨습니다. 만약 저자가 "어떤 유형의 고객에게 어떤 모드를 선택해야 하는지의 구체적 기준"을 도출해서 제시한다면, 이것이 실무적 기여로 충분할까요? 아니면 더 근본적인 프레이밍 변경이 필요할까요?

---

## 전체 리뷰어에게 공통 질문

### RQ-C1: 수정 우선순위
> 지적하신 모든 사항을 한번에 해결하기는 어렵습니다. 현실적으로 **3가지만** 수정하여 제출한다면, 어떤 3가지를 선택하시겠습니까?

### RQ-C2: 이 논문의 "죽일 수 없는 강점"
> 모든 약점에도 불구하고, 이 논문이 반드시 출판되어야 하는 이유가 있다면 무엇일까요? 어떤 점이 기존 문헌에서 이 논문만이 제공하는 고유한 기여입니까?

### RQ-C3: 최소한의 실증적 검증
> 실제 기업 데이터에 접근할 수 없는 상황에서, 논문의 외적 타당성을 높이기 위해 취할 수 있는 **가장 현실적인 조치**는 무엇입니까?
