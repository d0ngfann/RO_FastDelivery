#!/bin/bash
#SBATCH --job-name=ROFP_full
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=normal

# submit_slurm.sh
# SLURM 배치 작업 제출 스크립트 (학교 서버가 SLURM 사용시)
# 사용법: sbatch submit_slurm.sh
#
# 주의: 학교 서버 환경에 맞게 #SBATCH 옵션 수정 필요
# - partition: 사용 가능한 파티션 이름으로 변경
# - time: 예상 실행 시간 (최대 24시간)
# - cpus-per-task: CPU 코어 수 (Gurobi threads 설정과 일치)
# - mem: 메모리 (GB)

echo "=========================================="
echo "SLURM Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 환경 설정
module load anaconda3/2025.7.0-2-python_3.11
conda activate ROFP

# 작업 정보 출력
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo ""

# Gurobi 설정 확인
echo "Gurobi license file: $GRB_LICENSE_FILE"
python3 -c "import gurobipy as gp; print(f'Gurobi version: {gp.gurobi.version()}')"
echo ""

# 메인 실행
echo "Running Full Instance Sensitivity Analysis..."
python3 DH_main.py full

# 완료
echo ""
echo "=========================================="
echo "SLURM Job Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved in: result/"
ls -lh result/*.csv result/*.png 2>/dev/null | tail -5
