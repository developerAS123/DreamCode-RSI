from __future__ import annotations

import pytest
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.fixed import FixedPolicy, RandomPolicy
from dreamcode_rsi.policies.greedy import GreedyPolicy
from dreamcode_rsi.replay.scoring import compute_replay_score
from dreamcode_rsi.replay.simulator import ReplaySimulator
from dreamcode_rsi.replay.runner import ReplayRunner


def build_sample_discovery_tree() -> DiscoveryTree:
    """Build a deterministic multi-branch tree to test replay trajectories."""
    tree = DiscoveryTree(problem_id="matrix_multiply", round_id=1)

    # Root
    root = DiscoveryNode(
        node_id="root",
        parent_id=None,
        problem_id="matrix_multiply",
        agent_id="baseline",
        proposal="Naive baseline",
        artifact_uri="uri://root",
        score=0.50,
        correct=True,
        runtime_ms=20.0,
    )
    tree.add_node(root)

    # Level 1 branches
    branch_a = DiscoveryNode(
        node_id="A_transpose",
        parent_id="root",
        problem_id="matrix_multiply",
        agent_id="algorithm",
        proposal="Transpose B",
        artifact_uri="uri://A",
        score=0.85,
        correct=True,
        runtime_ms=6.0,
    )
    branch_b = DiscoveryNode(
        node_id="B_interchange",
        parent_id="root",
        problem_id="matrix_multiply",
        agent_id="algorithm",
        proposal="Loop interchange ikj",
        artifact_uri="uri://B",
        score=0.65,
        correct=True,
        runtime_ms=12.0,
    )
    branch_c = DiscoveryNode(
        node_id="C_buggy",
        parent_id="root",
        problem_id="matrix_multiply",
        agent_id="debug",
        proposal="Buggy index attempt",
        artifact_uri="uri://C",
        score=0.0,
        correct=False,
    )
    tree.add_node(branch_a)
    tree.add_node(branch_b)
    tree.add_node(branch_c)

    # Level 2 sub-branches
    branch_a_sub = DiscoveryNode(
        node_id="A1_map_zip",
        parent_id="A_transpose",
        problem_id="matrix_multiply",
        agent_id="algorithm",
        proposal="Map mul with pretransposed B",
        artifact_uri="uri://A1",
        score=0.92,
        correct=True,
        runtime_ms=4.0,
    )
    branch_b_sub = DiscoveryNode(
        node_id="B1_unroll",
        parent_id="B_interchange",
        problem_id="matrix_multiply",
        agent_id="algorithm",
        proposal="Unroll inner loop 4x",
        artifact_uri="uri://B1",
        score=0.70,
        correct=True,
        runtime_ms=10.0,
    )
    tree.add_node(branch_a_sub)
    tree.add_node(branch_b_sub)

    return tree


def test_replay_scoring_formula():
    # quality=0.85, attempts=5, rounds=2
    # parallel_eff = 5 / 2 = 2.5
    # score = 0.85 - (0.01 * 5) + (0.05 * 2.5) = 0.85 - 0.05 + 0.125 = 0.925
    breakdown = compute_replay_score(
        best_quality=0.85,
        attempts=5,
        rounds=2,
        beta_cost=0.01,
        beta_parallel=0.05,
    )
    assert breakdown.best_quality == 0.85
    assert breakdown.attempts == 5
    assert breakdown.rounds == 2
    assert breakdown.parallel_efficiency == 2.5
    assert breakdown.cost_penalty == 0.05
    assert breakdown.parallel_bonus == 0.125
    assert breakdown.total_score == 0.925


def test_core_milestone_different_policies_produce_different_trajectories():
    """
    CRITICAL RESEARCH MILESTONE:
    The same recorded tree must produce different trajectories and scores under different policies.
    """
    tree = build_sample_discovery_tree()
    sim = ReplaySimulator(tree)

    fixed_policy = FixedPolicy(name="fixed_fifo")
    greedy_policy = GreedyPolicy(name="greedy_best")
    random_policy = RandomPolicy(seed=123, name="random_seed_123")

    # Worker count 1 forces sequential selection choices
    res_fixed = sim.run(fixed_policy, max_rounds=3, workers=1)
    res_greedy = sim.run(greedy_policy, max_rounds=3, workers=1)
    res_random = sim.run(random_policy, max_rounds=3, workers=1)

    print("\n--- Milestone Check: Same Tree, Different Policies ---")
    print(f"Fixed Policy Trajectory:  {res_fixed.trajectory} -> Score: {res_fixed.final_score}")
    print(f"Greedy Policy Trajectory: {res_greedy.trajectory} -> Score: {res_greedy.final_score}")
    print(f"Random Policy Trajectory: {res_random.trajectory} -> Score: {res_random.final_score}")

    # Trajectories must NOT be identical
    assert res_fixed.trajectory != res_greedy.trajectory or res_fixed.final_score != res_greedy.final_score
    # Greedy policy must prioritize high scoring branch A_transpose and reach highest score
    assert res_greedy.best_score >= res_fixed.best_score
    assert "A_transpose" in res_greedy.trajectory[:2]

    # Verify causal consistency: A1_map_zip cannot be revealed before A_transpose
    if "A1_map_zip" in res_greedy.trajectory:
        assert res_greedy.trajectory.index("A1_map_zip") > res_greedy.trajectory.index("A_transpose")


def test_replay_runner_pool_comparison():
    tree1 = build_sample_discovery_tree()
    tree2 = build_sample_discovery_tree()

    runner = ReplayRunner(simulator_pool=[tree1, tree2])
    policies = [
        FixedPolicy(name="fixed_fifo"),
        GreedyPolicy(name="greedy_best"),
        RandomPolicy(seed=42, name="random_p"),
    ]

    report = runner.compare_policies(policies, max_rounds=4, workers=2)
    print("\n" + report.format_text_report())

    assert report.n_trees_evaluated == 2
    assert report.best_policy_name in ["greedy_best", "fixed_fifo"]
    assert "greedy_best" in report.summaries
    assert report.summaries["greedy_best"].mean_best_quality >= 0.85

