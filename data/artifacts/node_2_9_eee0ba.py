def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    B_T = list(zip(*B))
    return [[sum(x * y for x, y in zip(row, col)) for col in B_T] for row in A]