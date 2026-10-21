from __future__ import annotations

import math
from typing import Any
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.policies.base import ExplorationPolicy


class UncertaintyAwarePolicy(ExplorationPolicy):
    """
    Research Extension: Uncertainty-Aware Exploration Controller.
    
    Formula:
      priority(node) = (w1 * expected_gain(node)
                      + w2 * uncertainty(node)
                      + w3 * diversity(node)
                      + w4 * historical_success(node)
                      - w5 * estimated_cost(node))
    """

    def __init__(
        self,
        name: str = "uncertainty_aware_layadream",
        w1: float = 1.0,  # expected_gain
        w2: float = 0.4,  # uncertainty
        w3: float = 0.3,  # diversity
        w4: float = 0.5,  # historical_success
        w5: float = 0.2,  # estimated_cost
    ):
        self.name = name
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.agent_visit_counts: dict[str, int] = {}
        self.strategy_visit_counts: dict[str, int] = {}

    def compute_node_priority(self, node: DiscoveryNode, total_available: int) -> float:
        # 1. Expected Gain: score adjusted for depth
        score = getattr(node, "score", 0.5)
        depth = getattr(node, "depth", 1)
        expected_gain = score * (0.95 ** depth)

        # 2. Uncertainty Score: derived from sparse observations & visit novelty
        agent_id = getattr(node, "agent_id", "algorithm")
        strat = node.metadata.get("strategy_tag", "unknown") if hasattr(node, "metadata") else "unknown"
        agent_visits = self.agent_visit_counts.get(agent_id, 0)
        strat_visits = self.strategy_visit_counts.get(strat, 0)

        # Sparse observation heuristic: 1 / sqrt(visits + 1)
        sparsity = 1.0 / math.sqrt(agent_visits + strat_visits + 1.0)
        uncertainty = sparsity

        # 3. Diversity Score: bonus for unrepresented specialist agents
        diversity = 1.0 if agent_visits == 0 else (1.0 / (agent_visits + 1.0))

        # 4. Historical Success Score: correctness probability
        is_correct = getattr(node, "correct", True)
        hist_success = 1.0 if is_correct else 0.0

        # 5. Estimated Cost: normalized runtime proxy
        runtime_ms = getattr(node, "runtime_ms", 10.0) or 10.0
        est_cost = min(2.0, runtime_ms / 15.0)

        # Priority calculation
        priority = (
            (self.w1 * expected_gain)
            + (self.w2 * uncertainty)
            + (self.w3 * diversity)
            + (self.w4 * hist_success)
            - (self.w5 * est_cost)
        )
        return priority

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        if not available_nodes:
            return []

        # Score every available candidate node
        prioritized = [
            (node, self.compute_node_priority(node, len(available_nodes)))
            for node in available_nodes
        ]
        # Sort descending by priority
        prioritized.sort(key=lambda item: item[1], reverse=True)

        selected: list[DiscoveryNode] = []
        k = min(len(available_nodes), max(1, workers))

        for node, prio in prioritized[:k]:
            selected.append(node)
            agent_id = getattr(node, "agent_id", "algorithm")
            strat = node.metadata.get("strategy_tag", "unknown") if hasattr(node, "metadata") else "unknown"
            self.agent_visit_counts[agent_id] = self.agent_visit_counts.get(agent_id, 0) + 1
            self.strategy_visit_counts[strat] = self.strategy_visit_counts.get(strat, 0) + 1

        return selected

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "uncertainty_aware",
            "weights": {
                "w1_gain": self.w1,
                "w2_uncertainty": self.w2,
                "w3_diversity": self.w3,
                "w4_success": self.w4,
                "w5_cost": self.w5,
            },
            "formula": "w1*gain + w2*uncertainty + w3*diversity + w4*success - w5*cost",
        }

