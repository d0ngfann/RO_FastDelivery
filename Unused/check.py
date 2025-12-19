from docplex.mp.model import Model

# 간단한 모델 생성
model = Model(name='cplex_test')

# 변수 정의
x = model.continuous_var(name='x', lb=0)
y = model.continuous_var(name='y', lb=0)

# 제약조건
model.add_constraint(x + y >= 8)
model.add_constraint(2*x + y >= 10)

# 목적함수
model.minimize(5*x + 4*y)

print("모델 생성 완료")
print("CPLEX 런타임 확인 중...")

# 실제 해결 시도 - 여기서 CPLEX API 필요
try:
    solution = model.solve()
    if solution:
        print("✅ CPLEX 런타임이 정상적으로 설치되어 있습니다!")
        print(f"해결책: x = {x.solution_value}, y = {y.solution_value}")
        print(f"목적함수 값: {solution.objective_value}")
    else:
        print("❌ 모델을 해결할 수 없습니다.")
except Exception as e:
    print(f"❌ CPLEX 런타임 오류: {e}")
    if "no CPLEX runtime found" in str(e):
        print("CPLEX Optimization Studio가 설치되지 않았거나")
        print("setup.py 실행이 필요합니다.")
