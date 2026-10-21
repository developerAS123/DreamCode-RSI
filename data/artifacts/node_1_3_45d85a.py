def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    rem = p % 4
    limit = p - rem
    for i in range(n):
        a_row = A[i]
        c_row = C[i]
        for k in range(m):
            a_val = a_row[k]
            b_row = B[k]
            for j in range(0, limit, 4):
                c_row[j] += a_val * b_row[j]
                c_row[j+1] += a_val * b_row[j+1]
                c_row[j+2] += a_val * b_row[j+2]
                c_row[j+3] += a_val * b_row[j+3]
            for j in range(limit, p):
                c_row[j] += a_val * b_row[j]
    return C