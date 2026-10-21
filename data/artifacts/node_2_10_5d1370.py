import operator

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    B_T = list(zip(*B))
    return [[sum(map(operator.mul, row_A, col_B)) for col_B in B_T] for row_A in A]