from __future__ import annotations

import pytest
from dreamcode_rsi.evaluation.benchmarks import get_benchmark
from dreamcode_rsi.evaluation.evaluator import CodeEvaluator
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.discovery.agent import DiscoveryAgent, MATMUL_CANDIDATE_STRATEGIES
from dreamcode_rsi.discovery.history import TreeHistoryManager


def test_evaluator_security_rejection():
    evaluator = CodeEvaluator()
    task = get_benchmark("matrix_multiply")
    malicious_code = """
import os

def matmul(A, B):
    os.listdir('/')
    return []
"""
    report = evaluator.evaluate(malicious_code, task)
    assert not report.correct
    assert report.score == 0.0
    assert not report.static_check_passed
    assert "SecurityViolation" in (report.error or "")


def test_evaluator_correctness_and_baseline_scoring():
    evaluator = CodeEvaluator()
    task = get_benchmark("matrix_multiply")
    report = evaluator.evaluate(task.baseline_code, task)
    assert report.correct
    assert report.score > 0.0
    assert report.runtime_ms is not None
    assert report.error is None


def test_evaluator_speedup_higher_score():
    evaluator = CodeEvaluator()
    task = get_benchmark("matrix_multiply")
    fast_code = """
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    B_T = list(zip(*B))
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in B_T] for row_a in A]
"""
    fast_report = evaluator.evaluate(fast_code, task)
    base_report = evaluator.evaluate(task.baseline_code, task)

    assert fast_report.correct
    assert fast_report.runtime_ms is not None
    assert base_report.runtime_ms is not None
    assert fast_report.runtime_ms < base_report.runtime_ms
    assert fast_report.score > base_report.score


def test_tree_invariants():
    tree = DiscoveryTree(problem_id="matrix_multiply", round_id=1)

    root = DiscoveryNode(
        node_id="root",
        parent_id=None,
        problem_id="matrix_multiply",
        agent_id="baseline",
        proposal="Baseline naive triple loop",
        artifact_uri="uri://baseline",
        score=0.45,
        correct=True,
        runtime_ms=18.0,
    )
    tree.add_node(root)
    assert tree.root_id == "root"
    assert tree.size() == 1
    assert tree.get_node("root").depth == 0

    child = DiscoveryNode(
        node_id="child_1",
        parent_id="root",
        problem_id="matrix_multiply",
        agent_id="algorithm",
        proposal="Loop interchange ikj",
        artifact_uri="uri://child_1",
        score=0.75,
        correct=True,
        runtime_ms=4.0,
    )
    tree.add_node(child)
    assert tree.size() == 2
    assert tree.get_node("child_1").depth == 1
    assert tree.best_node().node_id == "child_1"

    # Invariant: non-existent parent must raise KeyError
    with pytest.raises(KeyError):
        tree.add_node(DiscoveryNode(
            node_id="invalid",
            parent_id="non_existent",
            problem_id="matrix_multiply",
            agent_id="algorithm",
            proposal="test",
            artifact_uri="uri://test",
            score=0.5,
            correct=True,
        ))


def test_online_discovery_loop_generates_recorded_nodes():
    task = get_benchmark("matrix_multiply")
    evaluator = CodeEvaluator()
    agent = DiscoveryAgent(evaluator=evaluator)
    history_mgr = TreeHistoryManager()

    tree = DiscoveryTree(problem_id=task.task_id, round_id=1)

    # 1. Add root baseline node
    root_report = evaluator.evaluate(task.baseline_code, task)
    root_node = DiscoveryNode(
        node_id="root",
        parent_id=None,
        problem_id=task.task_id,
        agent_id="baseline",
        proposal="Naive O(N^3) standard baseline",
        artifact_uri="uri://baseline",
        score=root_report.score,
        correct=root_report.correct,
        runtime_ms=root_report.runtime_ms,
        memory_mb=root_report.memory_mb,
    )
    tree.add_node(root_node)

    # 2. Run online exploration generating 10 candidate nodes across branches
    # Depth 1: open multiple strategy branches from root
    parent_ids = ["root"]
    for i in range(len(MATMUL_CANDIDATE_STRATEGIES)):
        current_parent_id = parent_ids[i % len(parent_ids)]
        parent_node = tree.get_node(current_parent_id)
        child = agent.generate_candidate(
            problem=task,
            parent_node=parent_node,
            strategy_index=i,
            round_id=1,
        )
        tree.add_node(child)
        if child.correct and child.score > 0.5:
            parent_ids.append(child.node_id)

    assert tree.size() >= 11
    # Save tree to disk
    save_path = history_mgr.save_tree(tree, experiment_id="test_week1_discovery")
    assert save_path.exists()

    # Reload from disk and verify integrity
    loaded_tree = history_mgr.load_tree(save_path)
    assert loaded_tree.size() == tree.size()
    assert loaded_tree.best_node().score >= root_node.score
    print("\nWeek 1 Tree Summary:\n", loaded_tree.render_ascii())

