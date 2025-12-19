import ast
import importlib
import sys
import re
from typing import List


def load_chromosome_from_file(filepath: str) -> List[int]:
    """
    .txt 파일에서 'Best Chromosome:' 라인을 찾아, 그 뒤에 오는 리스트를 파싱합니다.
    리스트가 여러 줄에 걸쳐 있고 파일에 다른 내용이 있어도 안정적으로 동작합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            full_content = f.read()

        # 1. "Best Chromosome:" 문자열을 찾습니다.
        keyword = "Best Chromosome:"
        start_index = full_content.find(keyword)

        if start_index == -1:
            print(f"오류: 파일에서 '{keyword}' 키워드를 찾을 수 없습니다 - {filepath}")
            return None

        # 2. 키워드 바로 뒷부분부터 시작하는 새로운 문자열을 만듭니다.
        content_after_keyword = full_content[start_index + len(keyword):]

        # 3. 해당 문자열에서 대괄호 '[]'로 둘러싸인 리스트 부분만 정확히 추출합니다.
        match = re.search(r'\[.*\]', content_after_keyword, re.DOTALL)

        if not match:
            print(f"오류: 파일 내용에서 유효한 리스트 형식 '[]'를 찾을 수 없습니다.")
            return None

        # 4. 추출된 순수한 리스트 문자열을 파싱합니다.
        list_str = match.group(0)
        chromosome = ast.literal_eval(list_str)
        return chromosome

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {filepath}")
        return None
    except Exception as e:
        print(f"오류: 파일 파싱 중 문제가 발생했습니다 - {e}")
        return None


def main():
    EVAL_ID = 1 # 몇번 시나리오 인지 수정 필요
    TRANSNUM = "one" # all, one , two, three  도 수정 가능
    COST = "LC"   # cost 도 수정 필요
    chromosome_file = f"result/{EVAL_ID}_{TRANSNUM}_{COST}_2500.txt"
    evaluation_module_name = f"evaluate_{COST}_{EVAL_ID}"

    print(f"▶ Chromosome 파일 로딩: {chromosome_file}")
    chromosome = load_chromosome_from_file(chromosome_file)

    if chromosome is None:
        sys.exit(1)

    print(f"▶ 평가 모듈 로딩: {evaluation_module_name}.py")
    try:
        evaluation_module = importlib.import_module(evaluation_module_name)
    except ImportError:
        print(f"오류: '{evaluation_module_name}.py' 모듈을 찾거나 임포트할 수 없습니다.")
        sys.exit(1)

    print("\n▶ 평가 함수 실행 중...")

    # evaluate 함수가 이제 objective_value와 상세 정보 딕셔너리를 반환합니다.
    objective_value, detailed_results = evaluation_module.evaluate(chromosome, return_vars=True)

    print("=" * 50)
    print("                평가 결과 요약")
    print("=" * 50)

    if detailed_results:
        # 반환된 딕셔너리에서 비용 상세 내역을 추출합니다.
        breakdown = detailed_results.get("breakdown", {})
        total_revenue = breakdown.get("Total Revenue", 0)

        # 총 비용 계산
        total_cost = sum(v for k, v in breakdown.items() if k != "Total Revenue")

        print(f"\n📈 총 매출 (Total Revenue): {total_revenue:,.2f}")
        print(f"📉 총 비용 (Total Cost): {total_cost:,.2f}")
        print(f"💰 최종 이익 (Profit): {-objective_value:,.2f}\n")
        print("--- 비용 상세 내역 및 매출 대비 비율 ---")

        if total_revenue > 0:
            for cost_name, cost_value in breakdown.items():
                if cost_name == "Total Revenue":
                    continue
                percentage = (cost_value / total_revenue) * 100
                print(f"- {cost_name:<25}: {cost_value:15,.2f} ({percentage:5.2f} %)")
        else:
            print("매출이 0이므로 비율을 계산할 수 없습니다.")

        # --- Primal Variables 값만 필터링하여 출력 ---
        variables = detailed_results.get("variables", {})
        if variables:
            print("\n--- Primal Variables (0이 아닌 값) ---")
            primal_prefixes = ('Aij', 'Ajr', 'u', 'v', 'eta', 'tau')
            for name, value in variables.items():
                if name.startswith(primal_prefixes):
                    print(f"{name}: {value}")
    else:
        print("최적 해를 찾지 못했습니다.")

    print("=" * 50)


if __name__ == "__main__":
    main()