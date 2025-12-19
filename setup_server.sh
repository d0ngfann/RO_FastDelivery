#!/bin/bash
# setup_server.sh
# 학교 서버 환경 설정 스크립트
# 사용법: bash setup_server.sh

echo "=========================================="
echo "RO_FastDelivery 서버 환경 설정"
echo "=========================================="

# 1. Anaconda 모듈 로드
echo "Step 1: Anaconda 모듈 로드"
module load anaconda3/2025.7.0-2-python_3.11

# 2. Conda 가상환경 생성
echo "Step 2: Conda 가상환경 생성 (ROFP)"
conda create -n ROFP python=3.11 -y

# 3. 가상환경 활성화
echo "Step 3: 가상환경 활성화"
conda activate ROFP

# 4. pip 업그레이드
echo "Step 4: pip 업그레이드"
pip install --upgrade pip

# 5. 필수 라이브러리 설치
echo "Step 5: 필수 라이브러리 설치"
pip install -r requirements.txt

# 6. Gurobi 라이센스 확인
echo "Step 6: Gurobi 라이센스 확인"
echo "----------------------------------------"
echo "Gurobi 라이센스 설정이 필요합니다."
echo ""
echo "옵션 1: 학교 서버에 Gurobi 라이센스가 이미 있는 경우"
echo "  - GRB_LICENSE_FILE 환경변수 확인: echo \$GRB_LICENSE_FILE"
echo "  - 또는 /opt/gurobi/gurobi.lic 같은 곳에 있을 수 있음"
echo ""
echo "옵션 2: 개인 학술 라이센스 사용"
echo "  1) https://www.gurobi.com/academia/academic-program-and-licenses/ 접속"
echo "  2) Academic License 발급"
echo "  3) 서버에서 'grbgetkey [라이센스키]' 실행"
echo "  4) 라이센스 파일 경로를 ~/.bashrc에 추가:"
echo "     export GRB_LICENSE_FILE=\"/home/[username]/gurobi.lic\""
echo ""
echo "Gurobi 테스트 실행:"
echo "  python3 gurobitest.py"
echo "----------------------------------------"

# 7. 테스트 실행
echo "Step 7: Gurobi 설치 테스트"
python3 -c "import gurobipy as gp; print(f'Gurobi version: {gp.gurobi.version()}')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Gurobi 설치 성공!"
else
    echo "⚠️  Gurobi 라이센스 설정이 필요합니다."
    echo "    위의 옵션 1 또는 2를 따라 라이센스를 설정하세요."
fi

# 8. 데이터 파일 확인
echo "Step 8: 데이터 파일 확인"
if [ -f "data/DH_data_full.pkl" ]; then
    echo "✅ Full instance 데이터 존재"
else
    echo "⚠️  데이터 파일이 없습니다. 생성 필요:"
    echo "    python3 DH_data_gen.py"
fi

echo ""
echo "=========================================="
echo "설정 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. Gurobi 라이센스 설정 (필요시)"
echo "2. 데이터 생성: python3 DH_data_gen.py"
echo "3. 테스트 실행: python3 DH_main.py toy"
echo "4. Full 실행: bash run_server.sh"
echo ""
echo "가상환경 활성화 명령:"
echo "  source activate ROFP"
echo ""
