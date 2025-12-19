# 학교 서버 설정 및 실행 가이드

학교 서버에서 RO_FastDelivery 프로젝트를 실행하기 위한 완전한 가이드입니다.

---

## 1. 초기 설정

### 1.1 프로젝트 업로드

서버에 프로젝트 파일을 업로드합니다:

```bash
# 로컬에서 서버로 파일 전송
scp -r /path/to/FirstPaper username@server.edu:/home/username/

# 또는 git 사용
ssh username@server.edu
git clone [repository-url]
cd FirstPaper
```

### 1.2 자동 설정 실행

```bash
# 서버에 SSH 접속 후
cd FirstPaper

# 실행 권한 부여
chmod +x setup_server.sh run_server.sh run_background.sh

# 자동 설정 스크립트 실행
bash setup_server.sh
```

이 스크립트가 자동으로:
- Python 모듈 로드
- `ROFP` 가상환경 생성
- 필요한 라이브러리 설치
- Gurobi 설치 확인

### 1.3 수동 설정 (자동 설정 실패 시)

```bash
# Anaconda 모듈 로드
module load anaconda3/2025.7.0-2-python_3.11

# 가상환경 생성
conda create -n ROFP python=3.11 -y

# 가상환경 활성화
conda activate ROFP

# pip 업그레이드
pip install --upgrade pip

# 라이브러리 설치
pip install -r requirements.txt
```

---

## 2. Gurobi 라이센스 설정

### 옵션 1: 학교 서버에 라이센스가 이미 있는 경우

```bash
# 라이센스 파일 위치 확인
echo $GRB_LICENSE_FILE

# 또는 일반적인 위치 확인
ls /opt/gurobi*/gurobi.lic
ls /usr/local/gurobi*/gurobi.lic

# 라이센스가 있으면 환경변수 설정
export GRB_LICENSE_FILE="/path/to/gurobi.lic"
```

### 옵션 2: 개인 학술 라이센스 사용 (권장)

1. **라이센스 발급**:
   - https://www.gurobi.com/academia/academic-program-and-licenses/ 접속
   - 학교 이메일로 로그인/가입
   - "Academic License" 클릭
   - 라이센스 키 복사 (예: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

2. **서버에서 라이센스 설치**:
   ```bash
   # 서버에 SSH 접속한 상태에서
   source activate ROFP
   grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```

3. **환경변수 영구 설정** (선택사항):
   ```bash
   # .bashrc에 추가
   echo 'export GRB_LICENSE_FILE="$HOME/gurobi.lic"' >> ~/.bashrc
   source ~/.bashrc
   ```

### 라이센스 테스트

```bash
source activate ROFP
python3 gurobitest.py
```

성공하면 "Optimize a model with..." 메시지가 출력됩니다.

---

## 3. 데이터 생성

```bash
source activate ROFP
python3 DH_data_gen.py
```

출력:
- `data/DH_data_toy.pkl`
- `data/DH_data_full.pkl`

---

## 4. 실행 방법

### 4.1 테스트 실행 (Toy Instance)

```bash
source activate ROFP
python3 DH_main.py toy
```

예상 실행 시간: 1-2분

### 4.2 Full Instance 실행

#### 방법 1: 포그라운드 실행 (터미널 유지 필요)

```bash
bash run_server.sh
```

#### 방법 2: 특정 Gamma 값만 실행

```bash
bash run_server.sh 10    # Gamma=10만 실행
bash run_server.sh 50    # Gamma=50만 실행
```

#### 방법 3: 백그라운드 실행 (터미널 종료 가능) ⭐ 권장

```bash
# 전체 실행
bash run_background.sh

# 특정 gamma만
bash run_background.sh 10

# 진행 상황 확인
tail -f logs/run_full_*.log
```

#### 방법 4: SLURM 배치 작업 (서버가 SLURM 사용시)

```bash
# SLURM 설정 파일 수정 (필요시)
nano submit_slurm.sh
# - partition 이름 확인/수정
# - CPU, 메모리, 시간 제한 조정

# 작업 제출
sbatch submit_slurm.sh

# 작업 상태 확인
squeue -u $USER

# 로그 확인
tail -f logs/slurm_*.out
```

---

## 5. 결과 확인

### 결과 파일 위치

```bash
ls -lh result/
```

출력 파일:
- `DH_sensitivity_full_YYYYMMDD_HHMMSS.csv`: 수치 결과
- `DH_sensitivity_full_YYYYMMDD_HHMMSS.png`: 그래프

### 결과 다운로드

```bash
# 로컬 컴퓨터에서 실행
scp username@server.edu:/home/username/FirstPaper/result/*.csv ./
scp username@server.edu:/home/username/FirstPaper/result/*.png ./
```

---

## 6. 유용한 명령어

### 가상환경 관리

```bash
# 가상환경 활성화
source activate ROFP

# 가상환경 비활성화
conda deactivate

# 설치된 패키지 확인
pip list

# 가상환경 삭제 (재설치 필요시)
conda env remove -n ROFP
```

### 프로세스 관리

```bash
# 실행 중인 Python 프로세스 확인
ps aux | grep DH_main

# 프로세스 종료
kill [PID]

# 강제 종료
kill -9 [PID]
```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/run_full_*.log

# 최근 50줄만 보기
tail -50 logs/run_full_*.log

# 에러 검색
grep -i "error" logs/run_full_*.log
```

### 디스크 사용량 확인

```bash
# 현재 디렉토리 용량
du -sh .

# 폴더별 용량
du -h --max-depth=1
```

---

## 7. 문제 해결

### 문제 1: Gurobi 라이센스 에러

```
GurobiError: No Gurobi license found
```

**해결**:
```bash
# 라이센스 파일 확인
ls ~/gurobi.lic

# 환경변수 확인
echo $GRB_LICENSE_FILE

# 다시 설정
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
grbgetkey [라이센스키]
```

### 문제 2: 메모리 부족

```
MemoryError: Unable to allocate array
```

**해결**:
- SLURM 사용 시 메모리 증가: `#SBATCH --mem=32G`
- 또는 Toy instance로 테스트

### 문제 3: 모듈 없음 에러

```
ModuleNotFoundError: No module named 'numpy'
```

**해결**:
```bash
source activate ROFP
pip install -r requirements.txt
```

### 문제 4: 너무 오래 걸림

**예상 실행 시간**:
- Toy instance: 1-2분
- Full instance, 하나의 Gamma: 10-60분
- Full instance, 전체 (11개 Gamma): 2-10시간

**해결**:
- 백그라운드 실행 사용
- 또는 특정 Gamma만 먼저 테스트: `bash run_server.sh 10`

---

## 8. 서버별 설정 차이

### 일반 Linux 서버

```bash
bash setup_server.sh
bash run_background.sh
```

### SLURM 클러스터

```bash
bash setup_server.sh
sbatch submit_slurm.sh
```

### PBS/Torque 시스템

`submit_slurm.sh`를 PBS 형식으로 수정 필요:

```bash
#!/bin/bash
#PBS -N ROFP_full
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l mem=16gb

cd $PBS_O_WORKDIR
module load python/ondemand-jupyter-python3.10
source activate ROFP
python3 DH_main.py full
```

---

## 9. 체크리스트

실행 전 확인사항:

- [ ] Python 모듈 로드됨
- [ ] ROFP 가상환경 생성됨
- [ ] 라이브러리 설치 완료
- [ ] Gurobi 라이센스 설정 완료 (`gurobitest.py` 성공)
- [ ] 데이터 파일 존재 (`data/DH_data_full.pkl`)
- [ ] `result/` 폴더 존재 (자동 생성됨)
- [ ] 충분한 디스크 공간 (최소 1GB)

---

## 10. 빠른 시작 (Quick Start)

완전 초보자용 단계별 가이드:

```bash
# 1. 서버 접속
ssh username@server.edu

# 2. 프로젝트 폴더로 이동
cd FirstPaper

# 3. 초기 설정 (최초 1회만)
bash setup_server.sh

# 4. Gurobi 라이센스 설정 (최초 1회만)
conda activate ROFP
grbgetkey [라이센스키]

# 5. 데이터 생성 (최초 1회만)
python3 DH_data_gen.py

# 6. 테스트 실행
python3 DH_main.py toy

# 7. 본 실행 (백그라운드)
bash run_background.sh

# 8. 진행 확인
tail -f logs/run_full_*.log

# 9. (몇 시간 후) 결과 확인
ls -lh result/
```

---

## 연락처 및 지원

문제 발생 시:
1. 로그 파일 확인 (`logs/` 폴더)
2. `gurobitest.py` 실행으로 Gurobi 확인
3. `test_all.py` 실행으로 전체 시스템 확인
