from __future__ import annotations

import pytest
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.fixed import FixedPolicy, RandomPolicy
from dreamcode_rsi.policies.evolving import DynamicPolicy
from dreamcode_rsi.improvement.developer import PolicyDeveloper
from dreamcode_rsi.improvement.validator import PromotionGate


def make_world(problem_id: str, round_id: int) -> DiscoveryTree:
    tree = DiscoveryTree(problem_id=problem_id, round_id=round_id)
    root = DiscoveryNode(
        node_id="root",
        parent_id=None,
        problem_id=problem_id,
        agent_id="baseline",
        proposal="Naive baseline",
        artifact_uri="uri://root",
        score=0.40,
        correct=True,
    )
    tree.add_node(root)

    # 3 branches with different scores
    for i, (strat, score, is_corr) in enumerate([
        ("slow", 0.3, True),
        ("medium", 0.6, True),
        ("high_optimal", 0.95, True),
        ("broken", 0.0, False),
    ]):
        tree.add_node(DiscoveryNode(
            node_id=f"branch_{i}",
            parent_id="root",
            problem_id=problem_id,
            agent_id="algorithm",
            proposal=f"Strategy {strat}",
            artifact_uri=f"uri://{strat}",
            score=score,
            correct=is_corr,
            depth=1,
        ))
    return tree


def test_dynamic_policy_compilation():
    code = """
class MyCustomPolicy:
    def __init__(self):
        self.name = "custom_test_policy"

    def select_batch(self, available_nodes, workers):
        return available_nodes[:workers]

    def describe(self):
        return {"name": self.name}
"""
    policy = DynamicPolicy(code=code)
    assert policy.name == "custom_test_policy"
    desc = policy.describe()
    assert desc["name"] == "custom_test_policy"


def test_promotion_gate_rejects_malicious_code():
    train_worlds = [make_world("matrix_multiply", 1)]
    val_worlds = [make_world("matrix_multiply", 2)]
    gate = PromotionGate(train_worlds, val_worlds)

    malicious_code = """
import os
class MaliciousPolicy:
    def select_batch(self, nodes, workers):
        os.system("echo bad")
        return []
    def describe(self):
        return {}
"""
    incumbent = FixedPolicy()
    report = gate.validate_and_decide(malicious_code, incumbent, candidate_name="malicious")
    assert not report.promoted
    assert "prohibited module" in report.reason.lower()


def test_promotion_gate_rejects_broken_interface():
    train_worlds = [make_world("matrix_multiply", 1)]
    val_worlds = [make_world("matrix_multiply", 2)]
    gate = PromotionGate(train_worlds, val_worlds)

    broken_code = """
class BrokenPolicy:
    def not_select_batch(self):
        pass
"""
    incumbent = FixedPolicy()
    report = gate.validate_and_decide(broken_code, incumbent, candidate_name="broken")
    assert not report.promoted
    assert "interface compliance" in report.reason.lower()


def test_recursive_policy_improvement_and_promotion_loop():
    """
    CRITICAL RESEARCH MILESTONE (WEEK 3):
    pi_0 (naive random baseline) -> policy developer -> pi_1 (improved policy) -> PromotionGate validation -> Promoted!
    """
    train_world = make_world("matrix_multiply", 1)
    val_world = make_world("matrix_multiply", 2)
    gate = PromotionGate([train_world], [val_world], min_improvement_margin=0.0)

    # 1. Incumbent policy pi_0 is random baseline
    incumbent_pi_0 = RandomPolicy(seed=42, name="pi_0_random")

    # 2. Developer agent generates candidate policy pi_1
    developer = PolicyDeveloper()
    candidate_pi_1 = developer.improve_policy(
        current_policy_code=FixedPolicy().describe()["description"],
        replay_report="Incumbent pi_0 selected suboptimal branches. Needs greedy score prioritization.",
        round_index=1,
    )

    assert candidate_pi_1.version.startswith("pi_1")
    assert len(candidate_pi_1.source_code) > 0

    # 3. Independent Promotion Gate validates candidate
    report = gate.validate_and_decide(
        candidate_code=candidate_pi_1.source_code,
        incumbent_policy=incumbent_pi_0,
        candidate_name=candidate_pi_1.version,
    )

    print("\n--- Promotion Gate Decision Report ---")
    print(f"Candidate: {report.candidate_version}")
    print(f"Promoted: {report.promoted}")
    print(f"Train Score: {report.candidate_train_score:.4f} vs Incumbent: {report.incumbent_score:.4f}")
    print(f"Reason: {report.reason}")

    assert report.promoted is True
    assert report.candidate_val_score >= report.incumbent_score

