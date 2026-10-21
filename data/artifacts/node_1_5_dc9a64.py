def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    # Threshold to switch to ikj
    if n <= 16:
        m, p = len(A[0]), len(B[0])
        C = [[0.0] * p for _ in range(n)]
        for i, a_row in enumerate(A):
            c_row = C[i]
            for k, a_val in enumerate(a_row):
                b_row = B[k]
                for j in range(p):
                    c_row[j] += a_val * b_row[j]
        return C
    
    # Pad to power of 2 if needed or standard ikj fallback
    B_T = list(zip(*B))
    return [[sum(a * b for a, b in zip(row_A, col_B)) for col_B in B_T] for row_A in A]