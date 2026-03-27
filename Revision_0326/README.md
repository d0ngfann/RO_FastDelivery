# Revision_0326 — Experiment Code

## Setup (집 컴퓨터에서)

### 1. Python 환경
```bash
pip install -r requirements.txt
```

### 2. Gurobi 라이선스
```bash
# 1) https://www.gurobi.com/academia/academic-program-and-licenses/ 에서 학교 이메일로 라이선스 발급
# 2) 학교 VPN (GlobalProtect, portal-palo.pitt.edu) 연결
# 3) 라이선스 활성화
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# 4) 활성화 후 VPN 꺼도 됨
```

### 3. 데이터 확인
`data/` 폴더에 100개 `.pkl` 파일이 있어야 합니다 (full 50 + full200 50).
없으면 재생성:
```bash
cd codes
python generate_50_seeds.py full
python generate_50_seeds.py full200
```

## 실행

모든 실행은 `codes/` 디렉토리에서 합니다.

```bash
cd codes
```

### 단일 테스트 (동작 확인용)
```bash
# Optimal 정책
python main.py full 10 --seed 1 --di HD

# FM2 (고정 모드)
python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2
```

### 전체 실험 실행
```bash
# Exp1: R=50 기본 재실행 (800 runs, ~1시간)
python run_exp1.py

# Exp2: R=200 Gamma sensitivity (4,000 runs, ~15시간 병렬4)
python run_exp2.py

# Breakeven Analysis (500 runs, ~40분)
python run_breakeven.py

# 선형 DI 비교 (320 runs, ~30분)
python run_linear_di.py

# 커버리지 SA (480 runs, ~40분)
python run_coverage.py
```

## 파일 구조

```
Revision_0326/
├── codes/
│   ├── config.py              # 설정 & 파라미터
│   ├── data_gen.py            # 데이터 생성, DI 함수
│   ├── master.py              # Master Problem (1단계)
│   ├── sub.py                 # Subproblem (최악 시나리오)
│   ├── algo.py                # C&CG 알고리즘 루프
│   ├── main.py                # 실행 (Optimal 정책)
│   ├── master_fixed_mode.py   # MP 고정모드 (master.py 상속)
│   ├── algo_fixed_mode.py     # C&CG 고정모드 (algo.py 상속)
│   ├── main_fixed_mode.py     # 실행 (FM0/FM1/FM2)
│   ├── generate_50_seeds.py   # 데이터셋 생성
│   ├── run_exp1.py            # Exp1 실행 스크립트
│   ├── run_exp2.py            # Exp2 실행 스크립트
│   ├── run_breakeven.py       # Breakeven 실행 스크립트
│   ├── run_linear_di.py       # 선형 DI 실행 스크립트
│   └── run_coverage.py        # 커버리지 SA 실행 스크립트
├── data/                      # 데이터셋 (.pkl)
├── result/                    # 실험 결과 (.csv)
├── reviews_folder/            # 리뷰 문서
├── requirements.txt
└── README.md
```

## 새 커맨드라인 인자 (이번 revision에서 추가)

| 인자 | 파일 | 설명 |
|------|------|------|
| `--tc2 0.60` | main.py, main_fixed_mode.py | TC_2 오버라이드 (breakeven용) |
| `--di-func linear` | main.py, main_fixed_mode.py | 선형 DI 함수 (robustness check) |
| `--coverage moderate` | main.py, main_fixed_mode.py | 서비스 커버리지 제약 (tight/moderate/relaxed) |
