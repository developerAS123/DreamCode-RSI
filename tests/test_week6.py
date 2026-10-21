from __future__ import annotations

import pytest
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.uncertainty import UncertaintyAwarePolicy
from dreamcode_rsi.evaluation.ablations import AblationEngine, AblationStudyResult


def create_mock_tree() -> DiscoveryTree:
    tree = DiscoveryTree(problem_id="matrix_multiply", round_id=1)
    root = DiscoveryNode(
        node_id="root",
        parent_id=None,
        problem_id="matrix_multiply",
        agent_id="baseline",
        proposal="Naive baseline",
        artifact_uri="uri://root",
        score=0.50,
        correct=True,
    )
    tree.add_node(root)

    for i in range(5):
        tree.add_node(DiscoveryNode(
            node_id=f"branch_{i}",
            parent_id="root",
            problem_id="matrix_multiply",
            agent_id=["algorithm", "gpu", "math", "debug", "algorithm"][i],
            proposal=f"Proposal {i}",
            artifact_uri=f"uri://{i}",
            score=0.6 + (i * 0.08),
            correct=True,
            metadata={"strategy_tag": f"strategy_{i}"},
        ))
    return tree


def test_uncertainty_aware_policy_priority_computation():
    policy = UncertaintyAwarePolicy(w1=1.0, w2=0.5, w3=0.3, w4=0.5, w5=0.2)
    tree = create_mock_tree()

    available = tree.unrevealed_children({"root"})
    assert len(available) == 5

    # Compute priority for first unvisited node
    prio_1 = policy.compute_node_priority(available[0], total_available=5)
    assert isinstance(prio_1, float)
    assert prio_1 > 0.0

    # Select batch of 2
    batch = policy.select_batch(available, workers=2)
    assert len(batch) == 2
    # Verify visit counts tracked
    assert sum(policy.agent_visit_counts.values()) == 2


def test_ablation_engine_all_six_conditions():
    tree = create_mock_tree()
    engine = AblationEngine()
    results = {
        "A": engine.run_ablation_a([tree]),
        "B": engine.run_ablation_b([tree]),
        "C": engine.run_ablation_c([tree]),
        "D": engine.run_ablation_d([tree]),
        "E": engine.run_ablation_e([tree]),
        "F": engine.run_ablation_f([tree]),
    }

    assert len(results) == 6
    for key, res in results.items():
        assert isinstance(res, AblationStudyResult)
        assert len(res.rows) >= 2
        md_table = res.to_markdown_table()
        assert "| Condition / Baseline |" in md_table
        assert res.ablation_name in md_table
        print(f"\n{res.ablation_name} PASSED:\n{md_table[:200]}...")
