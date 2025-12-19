import gurobipy as gp
from gurobipy import GRB

# 가장 단순한 모델
m = gp.Model("test")

# 변수 1개
x = m.addVar(lb=0, name="x")

# 목적함수: x 최대화
m.setObjective(x, GRB.MAXIMIZE)

# 제약식: x <= 1
m.addConstr(x <= 1)

# 최적화
m.optimize()

# 결과 출력
print("Status:", m.Status)
print("x =", x.X)
print("Objective =", m.ObjVal)
