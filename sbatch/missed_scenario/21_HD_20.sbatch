#!/bin/bash
#SBATCH --job-name=ROFP_21_HD
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=ROFP_21_HD_20-%j.out
#SBATCH --error=ROFP_21_HD_20-%j.err

module purge
# 1. 아나콘다 로드
module load anaconda3/2025.7.0-2-python_3.11
# 2. 구로비 모듈 로드 (라이선스 경로 설정을 위해 반드시 필요!)
module load gurobi/12.0.3

SEED=21
DI=HD
GAMMA=20

# Change to project root directory (sbatch/ 폴더 밖으로 이동)
cd $SLURM_SUBMIT_DIR/../..

# 가상환경 실행
conda run -n ROFP python DH_main.py full $GAMMA --seed $SEED --di $DI
