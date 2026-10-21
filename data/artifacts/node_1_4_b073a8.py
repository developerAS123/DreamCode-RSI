def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    BLOCK = 16
    for ii in range(0, n, BLOCK):
        i_end = min(ii + BLOCK, n)
        for kk in range(0, m, BLOCK):
            k_end = min(kk + BLOCK, m)
            for jj in range(0, p, BLOCK):
                j_end = min(jj + BLOCK, p)
                for i in range(ii, i_end):
                    a_row = A[i]
                    c_row = C[i]
                    for k in range(kk, k_end):
                        a_ik = a_row[k]
                        b_row = B[k]
                        for j in range(jj, j_end):
                            c_row[j] += a_ik * b_row[j]
    return C