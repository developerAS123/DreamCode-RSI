from __future__ import annotations

import json
from typing import Iterator
import networkx as nx
from pydantic import BaseModel, Field

from dreamcode_rsi.discovery.node import DiscoveryNode


class DiscoveryTree(BaseModel):
    problem_id: str
    round_id: int = 0
    nodes: dict[str, DiscoveryNode] = Field(default_factory=dict)
    root_id: str | None = None

    def add_node(self, node: DiscoveryNode) -> None:
        """Add node enforcing tree invariants."""
        if not self.nodes:
            # First node must be root
            node.parent_id = None
            node.depth = 0
            self.root_id = node.node_id
            self.nodes[node.node_id] = node
            return

        if node.node_id in self.nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in tree.")

        if node.parent_id is None:
            raise ValueError(f"Non-root node '{node.node_id}' must have a valid parent_id.")

        if node.parent_id not in self.nodes:
            raise KeyError(f"Parent '{node.parent_id}' not found in tree.")

        parent = self.nodes[node.parent_id]
        node.depth = parent.depth + 1
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> DiscoveryNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in tree.")
        return self.nodes[node_id]

    def get_children(self, node_id: str) -> list[DiscoveryNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def get_leaves(self) -> list[DiscoveryNode]:
        parent_ids = {n.parent_id for n in self.nodes.values() if n.parent_id is not None}
        return [n for n in self.nodes.values() if n.node_id not in parent_ids]

    def best_node(self) -> DiscoveryNode:
        if not self.nodes:
            raise ValueError("Tree is empty.")
        return max(self.nodes.values(), key=lambda n: n.score)

    def size(self) -> int:
        return len(self.nodes)

    def unrevealed_children(self, revealed_node_ids: set[str]) -> list[DiscoveryNode]:
        """
        Find candidate recorded children that can be revealed next.
        A child can be revealed if:
          1. It has NOT yet been revealed.
          2. Its parent HAS been revealed.
        """
        candidates: list[DiscoveryNode] = []
        for n in self.nodes.values():
            if n.node_id in revealed_node_ids:
                continue
            if n.parent_id is not None and n.parent_id in revealed_node_ids:
                candidates.append(n)
        return candidates

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for node in self.nodes.values():
            g.add_node(
                node.node_id,
                proposal=node.proposal,
                score=node.score,
                correct=node.correct,
                runtime_ms=node.runtime_ms,
                depth=node.depth,
                agent_id=node.agent_id,
            )
            if node.parent_id:
                g.add_edge(node.parent_id, node.node_id)
        return g

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "round_id": self.round_id,
            "root_id": self.root_id,
            "nodes": {nid: n.model_dump(mode="json") for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveryTree:
        tree = cls(
            problem_id=data["problem_id"],
            round_id=data.get("round_id", 0),
            root_id=data.get("root_id"),
        )
        for nid, ndict in data.get("nodes", {}).items():
            tree.nodes[nid] = DiscoveryNode.model_validate(ndict)
        return tree

    def render_ascii(self) -> str:
        """Render a readable ASCII representation of the tree."""
        if not self.root_id or self.root_id not in self.nodes:
            return "<Empty Tree>"

        lines: list[str] = []

        def _traverse(node_id: str, prefix: str = "", is_last: bool = True):
            node = self.nodes[node_id]
            connector = "└── " if is_last else "├── "
            status = "✓" if node.correct else "✗"
            line = f"{prefix}{connector}[{status}] {node.node_id} (score={node.score:.3f}, ms={node.runtime_ms or 0:.1f}) - {node.proposal[:45]}"
            lines.append(line)

            children = self.get_children(node_id)
            child_prefix = prefix + ("    " if is_last else "│   ")
            for idx, child in enumerate(children):
                _traverse(child.node_id, child_prefix, idx == len(children) - 1)

        root = self.nodes[self.root_id]
        lines.append(f"Root: {root.node_id} (Baseline) score={root.score:.3f}")
        for idx, child in enumerate(self.get_children(self.root_id)):
            _traverse(child.node_id, "", idx == len(self.get_children(self.root_id)) - 1)

        return "\n".join(lines)

