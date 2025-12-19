import subprocess
import sys
import os
import time

# --- 설정 ---
# 실행할 세트 번호 범위
START_SET = 20
END_SET = 20  # 1부터 20까지의 세트를 실행

# 파일 이름 접두사 및 접미사
prefixes = ['All']
suffixes = ['HD', 'LD', 'MD']
# --- 설정 끝 ---

# NumPy 등 라이브러리의 병렬 처리 충돌 방지를 위한 환경 변수 설정 (안정성 확보)
my_env = os.environ.copy()
my_env["OMP_NUM_THREADS"] = "1"
my_env["MKL_NUM_THREADS"] = "1"
my_env["OPENBLAS_NUM_THREADS"] = "1"
my_env["VECLIB_MAXIMUM_THREADS"] = "1"
my_env["NUMEXPR_NUM_THREADS"] = "1"

# 전체 실행 파일 카운터
total_scripts_to_run = len(prefixes) * len(suffixes) * (END_SET - START_SET + 1)
executed_count = 0
has_failed = False

print("=" * 60)
print("Sequential script execution started.")
print(f"Target: Up to {total_scripts_to_run} scripts across {END_SET - START_SET + 1} sets.")
print("=" * 60)

# 전체 시작 시간 기록
total_start_time = time.time()

try:
    for i in range(START_SET, END_SET + 1):
        print(f"\n{'='*15} Starting Set {i}/{END_SET} {'='*15}")
        for suffix in suffixes:
            for prefix in prefixes:
                executed_count += 1
                script_name = f"{prefix}_{i}_{suffix}.py"

                # 실행할 파일이 실제로 존재하는지 확인
                if not os.path.exists(script_name):
                    print(f"--- [SKIP] '{script_name}' not found. ---")
                    continue

                print(f"\n>>> [{executed_count}/{total_scripts_to_run}] Running '{script_name}'...")
                script_start_time = time.time()

                # 자식 스크립트를 실행하고 끝날 때까지 대기합니다.
                # 자식 스크립트의 모든 출력은 실시간으로 터미널에 표시됩니다.
                result = subprocess.run([sys.executable, script_name], env=my_env)

                script_end_time = time.time()
                script_duration = script_end_time - script_start_time

                # 실행 결과(returncode)를 확인하여 성공/실패를 판단합니다.
                if result.returncode == 0:
                    print(f">>> Success! '{script_name}' finished in {script_duration:.2f} seconds.")
                else:
                    # 스크립트 실행 중 오류 발생 시
                    print(f"!!! ERROR: '{script_name}' failed with exit code {result.returncode}.")
                    print("!!! Halting further execution.")
                    has_failed = True
                    break  # 가장 안쪽 루프 중단

            if has_failed:
                break  # 중간 루프 중단
        if has_failed:
            break  # 바깥쪽 루프 중단

except KeyboardInterrupt:
    print("\n\n!!! User interrupted the execution (Ctrl+C). Halting. !!!")
    has_failed = True

# 최종 결과 출력
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print("\n" + "=" * 60)
if has_failed:
    print("Tasks HALTED due to an error or user interruption.")
else:
    print("All tasks finished successfully.")

print(f"Total execution time: {total_duration:.2f} seconds.")
print("=" * 60)

# 윈도우에서 실행 후 창이 바로 닫히는 것을 방지
if sys.platform == "win32":
    os.system("pause")