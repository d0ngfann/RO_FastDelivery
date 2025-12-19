import os
import re


# ----------------------------------------------------------------------------
# 1. 'set_seeds' 함수만 정확히 추가하는 함수 (최종 안정화 버전)
# ----------------------------------------------------------------------------
def add_set_seeds_function(file_path):
    """
    주어진 파이썬 파일에서 'def create_random_individual():' 라인을 찾아
    그 바로 위에 'set_seeds' 함수 정의를 추가합니다.
    - 함수가 이미 존재하면 건너뜁니다.
    - 더 안정적인 코드 삽입 기준점을 사용합니다.

    :param file_path: 수정할 파이썬 파일 경로
    :return: (성공 여부, 상태 메시지) 튜플 ('success', 'skipped', 'error')
    """
    print(f"--- 처리 시작: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"-> 오류: 파일을 찾을 수 없습니다.")
        return False, "error"
    except Exception as e:
        print(f"-> 오류: 파일을 읽는 중 문제가 발생했습니다: {e}")
        return False, "error"

    # 1. 함수가 이미 파일에 존재하는지 확인
    if "def set_seeds(" in content:
        print("-> 정보: 'set_seeds' 함수가 이미 존재합니다. 건너뜁니다.")
        return True, "skipped"

    # 2. 파일에 추가할 함수 정의 코드
    set_seeds_func_str = """def set_seeds(seed_value):
    \"\"\"전역 시드를 설정하여 결과의 재현성을 보장합니다.\"\"\"
    random.seed(seed_value)
    np.random.seed(seed_value)
"""
    # 3. 삽입 기준점(def create_random_individual)을 찾는 정규식
    # 이 함수 정의 바로 '앞'에 코드를 삽입할 것입니다.
    pattern = re.compile(r'def\s+create_random_individual\(\):')

    # 4. 패턴이 존재하는지 확인 후 코드 삽입
    if not pattern.search(content):
        print(f"-> 오류: 삽입 기준점('def create_random_individual')을 찾을 수 없습니다.")
        return False, "error"

    # 찾은 패턴(함수 정의) 앞에 set_seeds 함수 코드를 삽입
    # \n\n를 추가하여 적절한 줄 간격을 유지합니다.
    replacement = set_seeds_func_str + "\n\n\n" + r'def create_random_individual():'
    modified_content = pattern.sub(replacement, content, 1)

    # 5. 변경된 내용을 파일에 다시 쓰기
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print("-> 성공: 'set_seeds' 함수를 파일에 추가했습니다.")
        return True, "success"
    except Exception as e:
        print(f"-> 오류: 파일을 쓰는 중 문제가 발생했습니다: {e}")
        return False, "error"


# ----------------------------------------------------------------------------
# 2. 메인 실행 로직 (이전과 동일)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    TARGET_FOLDER = '.'
    part1_options = ["All", "one", "two", "three"]
    part2_options = range(1, 21)
    part3_options = ["HD", "LD", "MD"]

    success_count = 0
    skipped_count = 0
    error_count = 0

    print("=" * 50)
    print("파이썬 파일 대상, 'set_seeds' 함수 추가 작업 (최종 버전)")
    print("=" * 50)

    for p1 in part1_options:
        for p2 in part2_options:
            for p3 in part3_options:
                filename = f"{p1}_{p2}_{p3}.py"
                file_path = os.path.join(TARGET_FOLDER, filename)

                result, status = add_set_seeds_function(file_path)
                if status == 'success':
                    success_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                else:  # 'error'
                    error_count += 1

    print("=" * 50)
    print("모든 작업 완료!")
    print(f"- 성공적으로 추가: {success_count}개 파일")
    print(f"- 이미 존재하여 건너뜀: {skipped_count}개 파일")
    print(f"- 오류 발생/건너뜀: {error_count}개 파일")
    print("=" * 50)