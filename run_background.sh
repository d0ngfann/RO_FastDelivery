#!/bin/bash
# run_background.sh
# 백그라운드로 실행 (터미널 종료해도 계속 실행)
# 사용법: bash run_background.sh [gamma_value]

echo "백그라운드 실행 시작..."

# 로그 파일 이름 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$1" ]; then
    LOG_FILE="logs/run_full_${TIMESTAMP}.log"
    GAMMA_ARG=""
else
    LOG_FILE="logs/run_gamma${1}_${TIMESTAMP}.log"
    GAMMA_ARG="$1"
fi

# 로그 디렉토리 생성
mkdir -p logs

# nohup으로 백그라운드 실행
nohup bash run_server.sh $GAMMA_ARG > $LOG_FILE 2>&1 &

PID=$!

echo "백그라운드 프로세스 시작됨"
echo "PID: $PID"
echo "로그 파일: $LOG_FILE"
echo ""
echo "진행 상황 확인:"
echo "  tail -f $LOG_FILE"
echo ""
echo "프로세스 종료:"
echo "  kill $PID"
echo ""
echo "실행 중인 프로세스 확인:"
echo "  ps aux | grep DH_main"
echo ""
