from __future__ import annotations

from typing import Protocol, runtime_checkable
from dreamcode_rsi.discovery.node import DiscoveryNode


@runtime_checkable
class ExplorationPolicy(Protocol):
    """Protocol defining exploration policy interface for online search and offline replay."""

    name: str

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        """Given unrevealed candidate children whose parents are observed, select up to `workers` nodes."""
        ...

    def describe(self) -> dict:
        """Return structured human-readable and machine-readable metadata of the policy."""
        ...

