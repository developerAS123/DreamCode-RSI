from __future__ import annotations

import ast
import types
from typing import Any
import structlog

from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.policies.base import ExplorationPolicy

logger = structlog.get_logger(__name__)


class DynamicPolicy(ExplorationPolicy):
    """
    Wraps dynamically evolved Python policy code created by the policy improvement agent.
    """

    def __init__(self, code: str, class_name: str | None = None, name: str = "evolving_policy"):
        self.code = code
        self.name = name
        self.class_name = class_name
        self._instance: Any = None
        self._compile_and_instantiate()

    def _compile_and_instantiate(self) -> None:
        # 1. Parse AST to locate ExplorationPolicy subclass
        tree = ast.parse(self.code)
        target_class: str | None = self.class_name

        if target_class is None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    target_class = node.name
                    break

        if not target_class:
            raise ValueError("No class definition found in policy source code.")

        # 2. Compile module in isolated namespace
        module_ns: dict[str, Any] = {
            "DiscoveryNode": DiscoveryNode,
            "ExplorationPolicy": ExplorationPolicy,
        }
        bytecode = compile(tree, filename="<dynamic_policy>", mode="exec")
        exec(bytecode, module_ns)

        if target_class not in module_ns:
            raise AttributeError(f"Class '{target_class}' was not initialized in module.")

        cls_obj = module_ns[target_class]
        self._instance = cls_obj()
        if hasattr(self._instance, "name"):
            self.name = getattr(self._instance, "name")

    def select_batch(
        self,
        available_nodes: list[DiscoveryNode],
        workers: int,
    ) -> list[DiscoveryNode]:
        if not hasattr(self._instance, "select_batch"):
            raise AttributeError("Policy instance does not implement select_batch().")
        return self._instance.select_batch(available_nodes, workers)

    def describe(self) -> dict:
        if hasattr(self._instance, "describe"):
            return self._instance.describe()
        return {
            "name": self.name,
            "type": "dynamic",
            "class_name": self.class_name,
            "code_snippet": self.code[:120],
        }

