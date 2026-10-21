def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        row_A = A[i]
        row_C = C[i]
        for k in range(m):
            a_ik = row_A[k]
            row_B = B[k]
            for j in range(p):
                row_C[j] += a_ik * row_B[j]
    return C