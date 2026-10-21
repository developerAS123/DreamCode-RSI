def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    flat_C = [0.0] * (n * p)
    for i in range(n):
        row_A = A[i]
        i_p = i * p
        for k in range(m):
            aik = row_A[k]
            row_B = B[k]
            for j in range(p):
                flat_C[i_p + j] += aik * row_B[j]
    return [flat_C[i * p:(i + 1) * p] for i in range(n)]