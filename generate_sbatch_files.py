#!/usr/bin/env python3
"""
Generate 200 sbatch files for SLURM job submission.
50 seeds × 4 DI scenarios = 200 jobs
"""

import os

# Parameters
seeds = range(1, 51)  # 1 to 50
di_scenarios = ['HD', 'MD', 'LD', 'Mixed']
gamma = 10  # Fixed gamma value (reduced from 20 for faster computation)

# sbatch template
sbatch_template = """#!/bin/bash
#SBATCH --job-name=ROFP_{seed}_{di}
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=ROFP_{seed}_{di}_{gamma}-%j.out
#SBATCH --error=ROFP_{seed}_{di}_{gamma}-%j.err

module purge
# 1. 아나콘다 로드
module load anaconda3/2025.7.0-2-python_3.11
# 2. 구로비 모듈 로드 (라이선스 경로 설정을 위해 반드시 필요!)
module load gurobi/12.0.3

SEED={seed}
DI={di}
GAMMA={gamma}

# Change to project root directory (sbatch/ 폴더 밖으로 이동)
cd $SLURM_SUBMIT_DIR/..

# 가상환경 실행
conda run -n ROFP python DH_main.py full $GAMMA --seed $SEED --di $DI
"""

# Create sbatch directory if not exists
os.makedirs('sbatch', exist_ok=True)

# Generate all sbatch files
file_count = 0
for seed in seeds:
    for di in di_scenarios:
        # Create filename
        filename = f"sbatch/{seed}_{di}_{gamma}.sbatch"

        # Fill template
        content = sbatch_template.format(
            seed=seed,
            di=di,
            gamma=gamma
        )

        # Write file
        with open(filename, 'w') as f:
            f.write(content)

        file_count += 1

        # Make executable
        os.chmod(filename, 0o755)

print(f"Generated {file_count} sbatch files in 'sbatch/' directory")
print(f"\nFiles: {seeds.start}_{di_scenarios[0]}_{gamma}.sbatch to {seeds.stop-1}_{di_scenarios[-1]}_{gamma}.sbatch")
print(f"\nTo submit all jobs, run:")
print(f"  cd sbatch && for f in *.sbatch; do sbatch $f; done")
