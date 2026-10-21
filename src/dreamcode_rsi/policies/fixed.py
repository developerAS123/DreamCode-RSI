from __future__ import annotations

import random
from typing import Any
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.policies.base import ExplorationPolicy


class RandomPolicy(ExplorationPolicy):
    """Uniform random exploration baseline."""

    def __init__(self, seed: int | None = 42, name: str = "random_baseline"):
        self.name = name
        self.seed = seed
        self._rng = random.Random(seed)

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        if not available_nodes:
            return []
        k = min(len(available_nodes), max(1, workers))
        # Uniform sample without replacement
        return self._rng.sample(available_nodes, k)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "random",
            "seed": self.seed,
            "description": "Uniform random selection over all available frontier children.",
        }


class FixedPolicy(ExplorationPolicy):
    """Deterministic FIFO / Breadth-First fixed exploration baseline."""

    def __init__(self, name: str = "fixed_fifo_baseline"):
        self.name = name

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        if not available_nodes:
            return []
        k = min(len(available_nodes), max(1, workers))
        # Deterministic FIFO order (by creation/index)
        return available_nodes[:k]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "fixed",
            "description": "Deterministic FIFO breadth-first queue order over candidate nodes.",
        }

