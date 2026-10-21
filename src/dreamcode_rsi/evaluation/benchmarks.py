from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
from pydantic import BaseModel


class BenchmarkTask(BaseModel):
    task_id: str
    name: str
    category: str  # "algorithm", "math", "gpu", "debug"
    description: str
    function_signature: str
    entrypoint: str
    baseline_code: str
    test_harness_template: str
    baseline_runtime_ms: float = 10.0


# --- Task 1: Matrix Multiplication Optimization (Algorithm / Compute) ---
MATMUL_BASELINE = '''
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Naive O(N^3) standard triple loop."""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C
'''

MATMUL_HARNESS = '''
import time
import json
import random

{candidate_code}

def run_tests_and_benchmark():
    # 1. Correctness Checks
    # Edge Case: 1x1
    assert matmul([[2.0]], [[3.0]]) == [[6.0]], "1x1 failed"
    
    # 2x2 Identity
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    M2 = [[3.0, 5.0], [7.0, 11.0]]
    res = matmul(M2, I2)
    assert abs(res[0][0] - 3.0) < 1e-5 and abs(res[1][1] - 11.0) < 1e-5, "Identity failed"

    # Deterministic 4x4
    A4 = [[float(i + j) for j in range(4)] for i in range(4)]
    B4 = [[float(i * j + 1) for j in range(4)] for i in range(4)]
    res4 = matmul(A4, B4)
    # Verification against known element
    assert abs(res4[0][0] - 6.0) < 1e-4, f"4x4 element [0][0] mismatch: {res4[0][0]}"

    # 2. Performance Benchmark on 48x48 matrix
    random.seed(42)
    N = 48
    A_bench = [[random.random() for _ in range(N)] for _ in range(N)]
    B_bench = [[random.random() for _ in range(N)] for _ in range(N)]

    # Warmup
    _ = matmul(A_bench, B_bench)

    # Benchmark 3 runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = matmul(A_bench, B_bench)
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = sorted(times)[1]
    
    output = {
        "correct": True,
        "runtime_ms": median_ms,
        "error": None
    }
    print(json.dumps(output))

if __name__ == "__main__":
    try:
        run_tests_and_benchmark()
    except AssertionError as ae:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"AssertionError: {str(ae)}"}))
    except Exception as exc:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"{type(exc).__name__}: {str(exc)}"}))
'''

# --- Task 2: Longest Increasing Subsequence (Algorithm / DP) ---
LIS_BASELINE = '''
def lis(nums: list[int]) -> int:
    """O(N^2) dynamic programming."""
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
'''

LIS_HARNESS = '''
import time
import json
import random

{candidate_code}

def run_tests_and_benchmark():
    assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis([0, 1, 0, 3, 2, 3]) == 4
    assert lis([7, 7, 7, 7, 7]) == 1
    assert lis([]) == 0

    random.seed(42)
    bench_data = [random.randint(0, 500) for _ in range(400)]
    
    # Warmup
    _ = lis(bench_data)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = lis(bench_data)
        times.append((time.perf_counter() - t0) * 1000.0)
    
    median_ms = sorted(times)[1]
    print(json.dumps({"correct": True, "runtime_ms": median_ms, "error": None}))

if __name__ == "__main__":
    try:
        run_tests_and_benchmark()
    except AssertionError as ae:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"AssertionError: {str(ae)}"}))
    except Exception as exc:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"{type(exc).__name__}: {str(exc)}"}))
'''

# --- Task 3: LayerNorm Kernel Emulation (GPU / Numerical Kernel) ---
LAYERNORM_BASELINE = '''
import math

def layernorm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    """Row-wise naive LayerNorm in pure Python."""
    result = []
    for row in x:
        mean = sum(row) / len(row)
        var = sum((val - mean) ** 2 for val in row) / len(row)
        std = math.sqrt(var + eps)
        result.append([(val - mean) / std for val in row])
    return result
'''

LAYERNORM_HARNESS = '''
import time
import json
import random
import math

{candidate_code}

def run_tests_and_benchmark():
    test_row = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = layernorm([test_row])
    mean_out = sum(out[0]) / len(out[0])
    assert abs(mean_out) < 1e-4, f"Normalized mean should be 0, got {mean_out}"

    random.seed(42)
    bench_matrix = [[random.uniform(-5.0, 5.0) for _ in range(128)] for _ in range(64)]

    # Warmup
    _ = layernorm(bench_matrix)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = layernorm(bench_matrix)
        times.append((time.perf_counter() - t0) * 1000.0)
    
    median_ms = sorted(times)[1]
    print(json.dumps({"correct": True, "runtime_ms": median_ms, "error": None}))

if __name__ == "__main__":
    try:
        run_tests_and_benchmark()
    except AssertionError as ae:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"AssertionError: {str(ae)}"}))
    except Exception as exc:
        print(json.dumps({"correct": False, "runtime_ms": None, "error": f"{type(exc).__name__}: {str(exc)}"}))
'''


BENCHMARKS: dict[str, BenchmarkTask] = {
    "matrix_multiply": BenchmarkTask(
        task_id="matrix_multiply",
        name="Matrix Multiplication Optimization",
        category="algorithm",
        description="Optimize matrix multiplication beyond naive O(N^3) through loop interchange, tiling, caching, and pre-allocation.",
        function_signature="def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:",
        entrypoint="matmul",
        baseline_code=MATMUL_BASELINE,
        test_harness_template=MATMUL_HARNESS,
        baseline_runtime_ms=18.0,
    ),
    "lis": BenchmarkTask(
        task_id="lis",
        name="Longest Increasing Subsequence",
        category="algorithm",
        description="Find length of the longest strictly increasing subsequence in O(N log N) using binary search / patience sorting.",
        function_signature="def lis(nums: list[int]) -> int:",
        entrypoint="lis",
        baseline_code=LIS_BASELINE,
        test_harness_template=LIS_HARNESS,
        baseline_runtime_ms=12.0,
    ),
    "layernorm": BenchmarkTask(
        task_id="layernorm",
        name="LayerNorm Kernel Optimization",
        category="gpu",
        description="Vectorized / cache-friendly row-wise normalization for transformer attention blocks.",
        function_signature="def layernorm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:",
        entrypoint="layernorm",
        baseline_code=LAYERNORM_BASELINE,
        test_harness_template=LAYERNORM_HARNESS,
        baseline_runtime_ms=8.0,
    ),
}


def get_benchmark(task_id: str = "matrix_multiply") -> BenchmarkTask:
    if task_id not in BENCHMARKS:
        raise KeyError(f"Benchmark task '{task_id}' not found. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[task_id]

