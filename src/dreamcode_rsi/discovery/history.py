from __future__ import annotations

import json
from pathlib import Path
from dreamcode_rsi.config import EXPERIMENTS_DIR
from dreamcode_rsi.discovery.tree import DiscoveryTree


class TreeHistoryManager:
    """Manages disk persistence and retrieval of versioned discovery trees for the simulator pool."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or EXPERIMENTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_tree(self, tree: DiscoveryTree, experiment_id: str | None = None) -> Path:
        exp_id = experiment_id or f"{tree.problem_id}_round_{tree.round_id}"
        file_path = self.base_dir / f"{exp_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tree.to_dict(), f, indent=2)
        return file_path

    def load_tree(self, file_path: Path | str) -> DiscoveryTree:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DiscoveryTree.from_dict(data)

    def list_trees(self, problem_id: str | None = None) -> list[Path]:
        pattern = f"{problem_id}_round_*.json" if problem_id else "*_round_*.json"
        return sorted(self.base_dir.glob(pattern))

    def get_simulator_pool(self, problem_id: str | None = None) -> list[DiscoveryTree]:
        return [self.load_tree(p) for p in self.list_trees(problem_id)]

