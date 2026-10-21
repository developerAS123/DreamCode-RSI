def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    p = len(B[0])
    # Transpose B to make column accesses sequential
    B_T = list(zip(*B))
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        row_A = A[i]
        for j in range(p):
            col_B = B_T[j]
            C[i][j] = sum(a * b for a, b in zip(row_A, col_B))
    return C