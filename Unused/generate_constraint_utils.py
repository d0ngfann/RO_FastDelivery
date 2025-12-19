import os

# --- 설정 ---
# 생성할 파일의 총 개수
TOTAL_FILES_TO_CREATE = 20

# 원본이 되는 파일 이름
SOURCE_FILENAME = "constraint_utils_numpy_1.py"

# 원본 파일에서 찾아 바꿀 코드 라인
# 기본 인자 값을 수정하는 것이므로, 함수 정의 라인 전체를 찾아 바꿉니다.
LINE_TO_FIND = 'filepath: str = "data/data1.xlsx",'
# --- 설정 끝 ---


def create_files():
    """
    원본 파일을 기반으로 지정된 개수만큼 새 파이썬 스크립트 파일을 생성합니다.
    """
    # 1. 원본 파일이 현재 위치에 있는지 확인합니다.
    if not os.path.exists(SOURCE_FILENAME):
        print(f"오류: 원본 파일 '{SOURCE_FILENAME}'을 찾을 수 없습니다.")
        print("이 스크립트를 원본 파일과 같은 폴더에 놓고 실행해주세요.")
        return

    # 2. 원본 파일의 내용을 한 번만 읽어옵니다.
    try:
        with open(SOURCE_FILENAME, 'r', encoding='utf-8') as f:
            original_content = f.read()
        print(f"원본 파일 '{SOURCE_FILENAME}'을 성공적으로 읽었습니다.\n")
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 3. 설정된 개수만큼 반복하여 파일을 생성합니다.
    for i in range(1, TOTAL_FILES_TO_CREATE + 1):
        # 새로운 파일 이름과 데이터 경로를 만듭니다.
        new_filename = f"constraint_utils_numpy_{i}.py"
        new_data_path_line = f'filepath: str = "data/data_{i}.xlsx",'

        # 원본 내용에서 해당 라인을 찾아 새로운 라인으로 교체합니다.
        modified_content = original_content.replace(LINE_TO_FIND, new_data_path_line)

        # 수정된 내용을 새로운 파일에 씁니다.
        try:
            with open(new_filename, 'w', encoding='utf-8') as f_new:
                f_new.write(modified_content)
            print(f"✅ {new_filename} 파일이 생성되었습니다.")
        except Exception as e:
            print(f"❌ {new_filename} 파일 생성 중 오류 발생: {e}")

    print(f"\n총 {TOTAL_FILES_TO_CREATE}개의 파일 생성이 완료되었습니다.")


# 스크립트 실행
if __name__ == "__main__":
    create_files()