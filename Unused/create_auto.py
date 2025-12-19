import os

# 파일이 있는 폴더 경로 (스크립트와 파일이 같은 폴더에 있다면 비워두거나 '.'으로 설정)
TARGET_FOLDER = '.'

# 파일명 조합을 위한 리스트
part1_options = ["All", "one", "two", "three"]
part2_options = range(1, 21)
part3_options = ["HD", "LD", "MD"]

# 변경할 내용 정의
original_text = 'tree_method="gpu_hist"'
replacement_text = 'tree_method="auto"'

# 카운터 초기화
files_changed = 0
files_not_found = 0

# 모든 파일명 조합을 만들며 작업 수행
for p1 in part1_options:
    for p2 in part2_options:
        for p3 in part3_options:
            # 최종 파일명 생성
            filename = f"{p1}_{p2}_{p3}.py"
            filepath = os.path.join(TARGET_FOLDER, filename)

            try:
                # 1. 파일 읽기
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                # 2. 내용 변경
                # 변경할 내용이 있는지 확인 후 교체
                if original_text in content:
                    new_content = content.replace(original_text, replacement_text)

                    # 3. 변경된 내용으로 파일 다시 쓰기
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)

                    print(f"✅ [수정 완료] {filename}")
                    files_changed += 1
                else:
                    print(f"⚠️ [내용 없음] {filename} 파일에는 변경할 내용이 없습니다.")

            except FileNotFoundError:
                # print(f"❌ [파일 없음] {filename} 파일을 찾을 수 없습니다.")
                files_not_found += 1

print("\n--- 작업 완료 ---")
print(f"총 {files_changed}개의 파일이 수정되었습니다.")
if files_not_found > 0:
    print(f"총 {files_not_found}개의 파일을 찾지 못했습니다.")