from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import structlog

from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.base import ExplorationPolicy
from dreamcode_rsi.replay.scoring import compute_replay_score, ReplayScoreBreakdown

logger = structlog.get_logger(__name__)


@dataclass
class ObservedTree:
    """Represents the currently revealed sub-graph during off-policy simulation."""
    revealed_node_ids: set[str] = field(default_factory=set)
    revealed_nodes: list[DiscoveryNode] = field(default_factory=list)

    @classmethod
    def root(cls, root_node: DiscoveryNode) -> ObservedTree:
        return cls(
            revealed_node_ids={root_node.node_id},
            revealed_nodes=[root_node],
        )

    def reveal(self, node: DiscoveryNode) -> None:
        self.revealed_node_ids.add(node.node_id)
        self.revealed_nodes.append(node)


@dataclass
class ReplayResult:
    policy_name: str
    problem_id: str
    best_score: float
    attempts: int
    rounds: int
    score_breakdown: ReplayScoreBreakdown
    trajectory: list[str]  # Ordered sequence of revealed node_ids
    round_history: list[dict[str, Any]]
    best_node_id: str | None = None

    @property
    def final_score(self) -> float:
        return self.score_breakdown.total_score


class ReplaySimulator:
    """
    Deterministic Off-Policy Replay Simulator.
    Evaluates alternative exploration policies over frozen historical trees without generating new code.
    """

    def __init__(self, frozen_tree: DiscoveryTree):
        self.tree = frozen_tree
        if not self.tree.root_id or self.tree.root_id not in self.tree.nodes:
            raise ValueError(f"Tree '{frozen_tree.problem_id}' lacks a valid root node.")
        self.root_node = self.tree.nodes[self.tree.root_id]

    def run(
        self,
        policy: ExplorationPolicy,
        max_rounds: int = 10,
        workers: int = 2,
        beta_cost: float = 0.01,
        beta_parallel: float = 0.05,
    ) -> ReplayResult:
        observed = ObservedTree.root(self.root_node)
        best_score = self.root_node.score
        best_node_id = self.root_node.node_id
        attempts = 0
        rounds = 0
        trajectory: list[str] = [self.root_node.node_id]
        round_history: list[dict[str, Any]] = []

        while rounds < max_rounds:
            available = self.tree.unrevealed_children(observed.revealed_node_ids)
            if not available:
                break

            batch = policy.select_batch(available, workers)
            if not batch:
                break

            round_reveals: list[str] = []
            for node in batch:
                observed.reveal(node)
                trajectory.append(node.node_id)
                round_reveals.append(node.node_id)
                attempts += 1
                if node.score > best_score:
                    best_score = node.score
                    best_node_id = node.node_id

            rounds += 1
            round_history.append({
                "round": rounds,
                "batch_size": len(batch),
                "nodes_revealed": round_reveals,
                "current_best_score": best_score,
            })

        breakdown = compute_replay_score(
            best_quality=best_score,
            attempts=attempts,
            rounds=rounds,
            beta_cost=beta_cost,
            beta_parallel=beta_parallel,
        )

        return ReplayResult(
            policy_name=getattr(policy, "name", "unknown_policy"),
            problem_id=self.tree.problem_id,
            best_score=best_score,
            best_node_id=best_node_id,
            attempts=attempts,
            rounds=rounds,
            score_breakdown=breakdown,
            trajectory=trajectory,
            round_history=round_history,
        )

