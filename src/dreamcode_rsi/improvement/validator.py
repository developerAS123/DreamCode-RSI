from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Sequence
import structlog

from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.base import ExplorationPolicy
from dreamcode_rsi.policies.evolving import DynamicPolicy
from dreamcode_rsi.replay.runner import ReplayRunner, PolicyEvaluationSummary

logger = structlog.get_logger(__name__)


@dataclass
class ValidationStageResult:
    stage_name: str
    passed: bool
    details: str


@dataclass
class PromotionGateReport:
    candidate_version: str
    promoted: bool
    reason: str
    incumbent_score: float
    candidate_train_score: float
    candidate_val_score: float
    stages: list[ValidationStageResult] = field(default_factory=list)


class PromotionGate:
    """
    Independent 5-stage validation gate ensuring candidate policies are safe, compliant, and superior.
    The controller cannot declare itself successful.
    """

    def __init__(
        self,
        training_worlds: Sequence[DiscoveryTree],
        reserved_validation_worlds: Sequence[DiscoveryTree],
        min_improvement_margin: float = 0.0,
    ):
        if not training_worlds:
            raise ValueError("PromotionGate requires at least one training world.")
        if not reserved_validation_worlds:
            raise ValueError("PromotionGate requires at least one reserved validation world.")

        self.training_worlds = list(training_worlds)
        self.reserved_val_worlds = list(reserved_validation_worlds)
        self.min_improvement_margin = min_improvement_margin
        self.train_runner = ReplayRunner(self.training_worlds)
        self.val_runner = ReplayRunner(self.reserved_val_worlds)

    def validate_and_decide(
        self,
        candidate_code: str,
        incumbent_policy: ExplorationPolicy,
        candidate_name: str = "candidate_policy",
    ) -> PromotionGateReport:
        stages: list[ValidationStageResult] = []

        # ---------------------------------------------------------
        # Stage 1: Static AST & Security Check
        # ---------------------------------------------------------
        try:
            tree = ast.parse(candidate_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check for prohibited modules
                    names = [a.name.split(".")[0] for a in getattr(node, "names", [])]
                    if getattr(node, "module", None):
                        names.append(node.module.split(".")[0])
                    for name in names:
                        if name in {"os", "sys", "subprocess", "socket", "requests", "shutil"}:
                            stages.append(ValidationStageResult(
                                stage_name="Stage 1: Static Security",
                                passed=False,
                                details=f"Prohibited import detected: '{name}'"
                            ))
                            return PromotionGateReport(
                                candidate_version=candidate_name,
                                promoted=False,
                                reason=f"Failed static security check: prohibited module '{name}'.",
                                incumbent_score=0.0,
                                candidate_train_score=0.0,
                                candidate_val_score=0.0,
                                stages=stages,
                            )
            stages.append(ValidationStageResult("Stage 1: Static Security", True, "AST parsed and verified clean."))
        except SyntaxError as se:
            stages.append(ValidationStageResult("Stage 1: Static Security", False, f"SyntaxError: {str(se)}"))
            return PromotionGateReport(
                candidate_version=candidate_name,
                promoted=False,
                reason=f"Failed syntax check: {str(se)}",
                incumbent_score=0.0,
                candidate_train_score=0.0,
                candidate_val_score=0.0,
                stages=stages,
            )

        # ---------------------------------------------------------
        # Stage 2: Policy Interface Unit Test
        # ---------------------------------------------------------
        try:
            dynamic_policy = DynamicPolicy(code=candidate_code, name=candidate_name)
            dummy_node = DiscoveryNode(
                node_id="dummy",
                parent_id="root",
                problem_id="test_mock",
                agent_id="algorithm",
                proposal="test proposal",
                artifact_uri="uri://dummy",
                score=0.5,
                correct=True,
            )
            batch = dynamic_policy.select_batch([dummy_node], workers=1)
            desc = dynamic_policy.describe()
            if not isinstance(batch, list) or not isinstance(desc, dict):
                raise TypeError("select_batch must return a list and describe must return a dict.")
            stages.append(ValidationStageResult("Stage 2: Interface Compliance", True, "Passed select_batch() and describe() unit checks."))
        except Exception as exc:
            stages.append(ValidationStageResult("Stage 2: Interface Compliance", False, str(exc)))
            return PromotionGateReport(
                candidate_version=candidate_name,
                promoted=False,
                reason=f"Failed interface compliance: {str(exc)}",
                incumbent_score=0.0,
                candidate_train_score=0.0,
                candidate_val_score=0.0,
                stages=stages,
            )

        # ---------------------------------------------------------
        # Stage 3: Replay Across Historical Training Worlds
        # ---------------------------------------------------------
        incumbent_train_summary = self.train_runner.evaluate_policy(incumbent_policy)
        candidate_train_summary = self.train_runner.evaluate_policy(dynamic_policy)

        stages.append(ValidationStageResult(
            "Stage 3: Training Replay",
            True,
            f"Train Score: {candidate_train_summary.mean_replay_score:.4f} vs Incumbent: {incumbent_train_summary.mean_replay_score:.4f}"
        ))

        # ---------------------------------------------------------
        # Stage 4: Replay Across Reserved Validation Worlds
        # ---------------------------------------------------------
        incumbent_val_summary = self.val_runner.evaluate_policy(incumbent_policy)
        candidate_val_summary = self.val_runner.evaluate_policy(dynamic_policy)

        stages.append(ValidationStageResult(
            "Stage 4: Reserved Validation Set Replay",
            True,
            f"Val Score: {candidate_val_summary.mean_replay_score:.4f} vs Incumbent: {incumbent_val_summary.mean_replay_score:.4f}"
        ))

        # ---------------------------------------------------------
        # Stage 5: Acceptance Rule Check
        # Candidate must beat incumbent on validation set by >= min_improvement_margin
        # ---------------------------------------------------------
        val_diff = candidate_val_summary.mean_replay_score - incumbent_val_summary.mean_replay_score
        promoted = val_diff >= self.min_improvement_margin

        if promoted:
            reason = (
                f"Accepted for promotion! Validation score ({candidate_val_summary.mean_replay_score:.4f}) "
                f"exceeds incumbent ({incumbent_val_summary.mean_replay_score:.4f}) by {val_diff:+.4f}."
            )
        else:
            reason = (
                f"Rejected. Validation score ({candidate_val_summary.mean_replay_score:.4f}) did not exceed "
                f"incumbent ({incumbent_val_summary.mean_replay_score:.4f}) by required margin {self.min_improvement_margin}."
            )

        stages.append(ValidationStageResult("Stage 5: Acceptance Rule", promoted, reason))

        return PromotionGateReport(
            candidate_version=candidate_name,
            promoted=promoted,
            reason=reason,
            incumbent_score=incumbent_val_summary.mean_replay_score,
            candidate_train_score=candidate_train_summary.mean_replay_score,
            candidate_val_score=candidate_val_summary.mean_replay_score,
            stages=stages,
        )

