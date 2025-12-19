# 서버 업로드 가이드

## 업로드할 파일 목록

### 필수 Python 파일 (6개)
```
DH_main.py          # 메인 실행 파일
DH_config.py        # 설정
DH_data_gen.py      # 데이터 생성/로딩
DH_algo.py          # C&CG 알고리즘
DH_master.py        # Master Problem
DH_sub.py           # Subproblem
requirements.txt    # 라이브러리 목록
```

### 서버 스크립트 (5개)
```
setup_server.sh                  # 초기 환경 설정
run_server.sh                    # 실행 스크립트
run_background.sh                # 백그라운드 실행
submit_slurm.sh                  # SLURM 배치 작업
generate_multiple_datasets.py    # 데이터 생성 (필요시)
```

### 데이터 파일 (50개)
```
data/DH_data_full_seed1.pkl
data/DH_data_full_seed2.pkl
...
data/DH_data_full_seed50.pkl
```

---

## 자동 업로드 (권장)

### 1. 스크립트 실행 권한 부여
```bash
chmod +x upload_to_server.sh
```

### 2. 업로드 실행
```bash
./upload_to_server.sh
```

이 스크립트가 자동으로:
- 필요한 Python 파일 업로드
- 서버 스크립트 업로드
- data/ 디렉토리 생성 및 50개 데이터 업로드
- result/ 디렉토리 생성
- 실행 권한 설정

---

## 수동 업로드 (선택사항)

### 1. Python 파일 업로드
```bash
scp DH_main.py DH_config.py DH_data_gen.py DH_algo.py DH_master.py DH_sub.py requirements.txt \
    dok99@h2p.crc.pitt.edu:~/firstpaper/
```

### 2. 서버 스크립트 업로드
```bash
scp setup_server.sh run_server.sh run_background.sh submit_slurm.sh generate_multiple_datasets.py \
    dok99@h2p.crc.pitt.edu:~/firstpaper/
```

### 3. 서버에 디렉토리 생성
```bash
ssh dok99@h2p.crc.pitt.edu "mkdir -p ~/firstpaper/data ~/firstpaper/result"
```

### 4. 데이터 파일 업로드
```bash
scp data/DH_data_full_seed*.pkl dok99@h2p.crc.pitt.edu:~/firstpaper/data/
```

### 5. 실행 권한 설정
```bash
ssh dok99@h2p.crc.pitt.edu "chmod +x ~/firstpaper/*.sh"
```

---

## 업로드 확인

### 서버에 접속
```bash
ssh dok99@h2p.crc.pitt.edu
```

### 파일 확인
```bash
cd firstpaper

# Python 파일 확인
ls -lh *.py

# 데이터 파일 개수 확인
ls data/*.pkl | wc -l
# 출력: 50

# 디렉토리 구조 확인
tree -L 2
# 또는
ls -R
```

---

## 서버에서 실행

### 1. 환경 설정 (최초 1회)
```bash
bash setup_server.sh
```

### 2. Gurobi 라이센스 설정 (최초 1회)
```bash
conda activate ROFP
grbgetkey [라이센스키]
```

### 3. 테스트 실행
```bash
python3 DH_main.py toy 5 --seed 1 --di HD
```

### 4. 본 실행
```bash
# 포그라운드 (터미널 유지 필요)
python3 DH_main.py full 10 --seed 1 --di HD

# 백그라운드 (터미널 종료 가능)
bash run_background.sh

# SLURM 배치 작업
sbatch submit_slurm.sh
```

---

## 파일 크기 및 전송 시간 예상

### 파일 크기
- Python 파일들: ~200KB
- 서버 스크립트들: ~20KB
- 데이터 파일 50개: ~10MB (206KB × 50)
- **총 크기: ~10-11MB**

### 전송 시간 (네트워크 속도별)
- 10 Mbps: ~10초
- 100 Mbps: ~1초
- 1 Gbps: 즉시

---

## 업로드되지 않는 파일들 (불필요)

다음 파일/폴더는 서버에 업로드하지 **않습니다**:

```
.DS_Store           # macOS 메타데이터
__pycache__/        # Python 캐시
.idea/              # IDE 설정
Unused/             # 사용하지 않는 코드
Algorithm_Comparison/
Transportation_Method_Comparison/
result/             # 로컬 결과 (서버에서 새로 생성)
*.md                # 문서 파일들 (실행에 불필요)
old_data_generation.py
test_*.py
check_*.py
analyze_*.py
DH_debug_gap.py
DH_fix_convergence.py
```

---

## 문제 해결

### SSH 연결 실패
```bash
# SSH 키 확인
ssh-add -l

# 새 연결 시도
ssh -v dok99@h2p.crc.pitt.edu
```

### 권한 거부 에러
```bash
# 로컬에서 실행
chmod 600 ~/.ssh/id_rsa
```

### 파일 전송 실패
```bash
# 한 개씩 업로드해서 어느 파일이 문제인지 확인
scp DH_main.py dok99@h2p.crc.pitt.edu:~/firstpaper/
scp DH_config.py dok99@h2p.crc.pitt.edu:~/firstpaper/
# ...
```

### 데이터 파일이 너무 많아서 오래 걸릴 때
```bash
# 압축해서 업로드
cd data
tar -czf DH_data_full_seeds.tar.gz DH_data_full_seed*.pkl
cd ..
scp data/DH_data_full_seeds.tar.gz dok99@h2p.crc.pitt.edu:~/firstpaper/data/

# 서버에서 압축 해제
ssh dok99@h2p.crc.pitt.edu "cd ~/firstpaper/data && tar -xzf DH_data_full_seeds.tar.gz"
```

---

## 빠른 시작 (Quick Start)

```bash
# 로컬에서
chmod +x upload_to_server.sh
./upload_to_server.sh

# 서버에서
ssh dok99@h2p.crc.pitt.edu
cd firstpaper
bash setup_server.sh
conda activate ROFP
grbgetkey [라이센스키]
python3 DH_main.py full 10 --seed 1 --di HD
```

---

## 추가 팁

### rsync 사용 (더 빠른 동기화)
```bash
rsync -avz --progress \
    --include='*.py' \
    --include='*.sh' \
    --include='*.txt' \
    --include='data/DH_data_full_seed*.pkl' \
    --exclude='*' \
    . dok99@h2p.crc.pitt.edu:~/firstpaper/
```

### 특정 seed만 업로드
```bash
# Seed 1-10만 업로드
scp data/DH_data_full_seed{1..10}.pkl dok99@h2p.crc.pitt.edu:~/firstpaper/data/
```

### 업로드 후 자동 실행
```bash
./upload_to_server.sh && ssh dok99@h2p.crc.pitt.edu "cd ~/firstpaper && bash run_background.sh"
```
