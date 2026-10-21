from __future__ import annotations

from typing import Any
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.policies.base import ExplorationPolicy


class GreedyPolicy(ExplorationPolicy):
    """Greedy Best-First exploration policy that prioritizes promising branches."""

    def __init__(self, name: str = "greedy_best_score_baseline"):
        self.name = name

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        if not available_nodes:
            return []

        # Sort available nodes:
        # 1. Prioritize nodes with higher direct score or higher parent score
        # 2. Prioritize correct nodes
        # 3. Deterministic tie-breaker by node_id
        sorted_nodes = sorted(
            available_nodes,
            key=lambda n: (n.correct, n.score, -n.depth),
            reverse=True,
        )

        k = min(len(sorted_nodes), max(1, workers))
        return sorted_nodes[:k]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "greedy",
            "description": "Greedy best-first selection sorted by score, correctness, and shallow depth.",
        }

