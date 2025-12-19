#!/bin/bash
# run_server.sh
# 서버에서 Full Instance 실행 스크립트
# 사용법: bash run_server.sh [gamma_value]
# 예시: bash run_server.sh 10        (gamma=10만 실행)
#       bash run_server.sh            (전체 sensitivity analysis)

echo "=========================================="
echo "RO_FastDelivery 실행 시작"
echo "=========================================="

# 1. Anaconda 모듈 로드
module load anaconda3/2025.7.0-2-python_3.11

# 2. 가상환경 활성화
conda activate ROFP

# 3. 현재 시간 기록
START_TIME=$(date +%s)
echo "시작 시간: $(date)"
echo ""

# 4. 실행
if [ -z "$1" ]; then
    # 인자 없으면 전체 sensitivity analysis
    echo "전체 Sensitivity Analysis 실행 (Gamma = 0, 10, 20, ..., 100)"
    python3 DH_main.py full
else
    # 특정 gamma 값만 실행
    echo "Gamma = $1 실행"
    python3 DH_main.py full $1
fi

# 5. 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "실행 완료!"
echo "=========================================="
echo "종료 시간: $(date)"
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"
echo ""
echo "결과 파일 위치: result/ 폴더"
ls -lh result/*.csv result/*.png 2>/dev/null | tail -5
echo ""
