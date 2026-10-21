from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import structlog

from dreamcode_rsi.config import ARTIFACTS_DIR
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.evaluation.benchmarks import BenchmarkTask
from dreamcode_rsi.evaluation.evaluator import CodeEvaluator, EvaluationReport

logger = structlog.get_logger(__name__)


# Library of realistic optimization implementations for deterministic / benchmark exploration
MATMUL_CANDIDATE_STRATEGIES = [
    {
        "strategy": "transpose_b",
        "agent": "algorithm",
        "proposal": "Pre-transpose matrix B so inner product traverses row-major contiguous memory.",
        "code": '''
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
'''
    },
    {
        "strategy": "ikj_loop_interchange",
        "agent": "algorithm",
        "proposal": "Reorder triple loop from i-j-k to i-k-j to access B row-wise, drastically reducing cache misses.",
        "code": '''
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
'''
    },
    {
        "strategy": "ikj_with_local_caching",
        "agent": "algorithm",
        "proposal": "Loop interchange with local variable caching to bypass list lookups.",
        "code": '''
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i, a_row in enumerate(A):
        c_row = C[i]
        for k, a_val in enumerate(a_row):
            b_row = B[k]
            for j in range(p):
                c_row[j] += a_val * b_row[j]
    return C
'''
    },
    {
        "strategy": "unrolled_inner_loop",
        "agent": "algorithm",
        "proposal": "Unroll the inner j-loop by 4 to reduce loop condition branch checks.",
        "code": '''
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
'''
    },
    {
        "strategy": "tiled_blocking",
        "agent": "gpu",
        "proposal": "Block-tiled matrix multiplication with 16x16 tile cache chunks.",
        "code": '''
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
'''
    },
    {
        "strategy": "strassen_algebraic",
        "agent": "math",
        "proposal": "Strassen divide-and-conquer algorithm with recursive 7-subproblem decomposition.",
        "code": '''
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
'''
    },
    {
        "strategy": "preallocated_flat_array",
        "agent": "algorithm",
        "proposal": "Flatten 2D matrices into 1D continuous arrays to eliminate inner pointer chasing.",
        "code": '''
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
'''
    },
    {
        "strategy": "buggy_dimension_mismatch",
        "agent": "debug",
        "proposal": "Attempt fast multiplication without initializing column width properly.",
        "code": '''
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    # Buggy: returns incorrect dimensions
    return [[0.0]]
'''
    },
    {
        "strategy": "security_test_exploit",
        "agent": "debug",
        "proposal": "Malicious probe attempting to import os and inspect env variables.",
        "code": '''
import os

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    _ = os.environ
    return [[0.0]]
'''
    },
    {
        "strategy": "transposed_list_comprehension",
        "agent": "algorithm",
        "proposal": "Pure Python optimized list comprehension with zip and pre-transposed B.",
        "code": '''
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    B_T = list(zip(*B))
    return [[sum(x * y for x, y in zip(row, col)) for col in B_T] for row in A]
'''
    },
    {
        "strategy": "map_inner_product",
        "agent": "math",
        "proposal": "Functional map inner product with operator.mul.",
        "code": '''
import operator

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    B_T = list(zip(*B))
    return [[sum(map(operator.mul, row_A, col_B)) for col_B in B_T] for row_A in A]
'''
    },
]


class DiscoveryAgent:
    """Specialist exploration agent that creates candidate branches."""

    def __init__(self, evaluator: CodeEvaluator | None = None):
        self.evaluator = evaluator or CodeEvaluator()

    def generate_candidate(
        self,
        problem: BenchmarkTask,
        parent_node: DiscoveryNode | None,
        strategy_index: int,
        round_id: int = 1,
        agent_id: str = "algorithm",
    ) -> DiscoveryNode:
        """Generate a new candidate code child node and evaluate it in the sandbox."""
        node_id = f"node_{round_id}_{strategy_index}_{uuid.uuid4().hex[:6]}"
        parent_id = parent_node.node_id if parent_node else None
        depth = (parent_node.depth + 1) if parent_node else 0

        # Select strategy from candidate library
        strat = MATMUL_CANDIDATE_STRATEGIES[strategy_index % len(MATMUL_CANDIDATE_STRATEGIES)]
        code = strat["code"].strip()
        proposal = strat["proposal"]
        selected_agent = strat.get("agent", agent_id)

        # Save artifact
        artifact_path = ARTIFACTS_DIR / f"{node_id}.py"
        artifact_path.write_text(code, encoding="utf-8")

        # Evaluate candidate in sandbox
        eval_report: EvaluationReport = self.evaluator.evaluate(code, problem)

        node = DiscoveryNode(
            node_id=node_id,
            parent_id=parent_id,
            problem_id=problem.task_id,
            agent_id=selected_agent,
            proposal=proposal,
            artifact_uri=str(artifact_path),
            score=eval_report.score,
            correct=eval_report.correct,
            runtime_ms=eval_report.runtime_ms,
            memory_mb=eval_report.memory_mb,
            error=eval_report.error,
            depth=depth,
            round_id=round_id,
            metadata={
                "strategy_tag": strat["strategy"],
                "speedup": eval_report.speedup,
                "static_passed": eval_report.static_check_passed,
            },
        )
        return node

