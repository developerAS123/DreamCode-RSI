from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import structlog

from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.policies.base import ExplorationPolicy
from dreamcode_rsi.replay.simulator import ReplaySimulator, ReplayResult

logger = structlog.get_logger(__name__)


@dataclass
class PolicyEvaluationSummary:
    policy_name: str
    mean_replay_score: float
    mean_best_quality: float
    mean_attempts: float
    mean_rounds: float
    results: list[ReplayResult] = field(default_factory=list)


@dataclass
class ReplayComparisonReport:
    summaries: dict[str, PolicyEvaluationSummary]
    best_policy_name: str
    n_trees_evaluated: int

    def to_dict(self) -> dict:
        return {
            "best_policy": self.best_policy_name,
            "trees_evaluated": self.n_trees_evaluated,
            "policies": {
                name: {
                    "mean_score": s.mean_replay_score,
                    "mean_quality": s.mean_best_quality,
                    "mean_attempts": s.mean_attempts,
                    "mean_rounds": s.mean_rounds,
                }
                for name, s in self.summaries.items()
            },
        }

    def format_text_report(self) -> str:
        lines = [
            f"=== Dream-RSI Replay Evaluation Report ({self.n_trees_evaluated} Simulator Worlds) ===",
            f"Top Performing Policy: {self.best_policy_name}",
            "",
            f"{'Policy Name':<28} | {'Score':<8} | {'Quality':<8} | {'Attempts':<9} | {'Rounds':<7}",
            "-" * 70,
        ]
        for name, summary in sorted(self.summaries.items(), key=lambda x: x[1].mean_replay_score, reverse=True):
            lines.append(
                f"{name:<28} | {summary.mean_replay_score:<8.4f} | {summary.mean_best_quality:<8.4f} | {summary.mean_attempts:<9.1f} | {summary.mean_rounds:<7.1f}"
            )
        return "\n".join(lines)


class ReplayRunner:
    """Orchestrates multi-policy evaluation across a pool of frozen discovery trees."""

    def __init__(self, simulator_pool: Sequence[DiscoveryTree]):
        self.simulator_pool = list(simulator_pool)
        if not self.simulator_pool:
            raise ValueError("ReplayRunner requires at least one tree in simulator pool.")

    def evaluate_policy(
        self,
        policy: ExplorationPolicy,
        max_rounds: int = 10,
        workers: int = 2,
    ) -> PolicyEvaluationSummary:
        results: list[ReplayResult] = []
        for tree in self.simulator_pool:
            sim = ReplaySimulator(tree)
            res = sim.run(policy, max_rounds=max_rounds, workers=workers)
            results.append(res)

        mean_score = sum(r.final_score for r in results) / len(results)
        mean_quality = sum(r.best_score for r in results) / len(results)
        mean_attempts = sum(r.attempts for r in results) / len(results)
        mean_rounds = sum(r.rounds for r in results) / len(results)

        return PolicyEvaluationSummary(
            policy_name=getattr(policy, "name", "unnamed_policy"),
            mean_replay_score=round(mean_score, 4),
            mean_best_quality=round(mean_quality, 4),
            mean_attempts=round(mean_attempts, 2),
            mean_rounds=round(mean_rounds, 2),
            results=results,
        )

    def compare_policies(
        self,
        policies: Sequence[ExplorationPolicy],
        max_rounds: int = 10,
        workers: int = 2,
    ) -> ReplayComparisonReport:
        summaries: dict[str, PolicyEvaluationSummary] = {}
        for p in policies:
            summary = self.evaluate_policy(p, max_rounds=max_rounds, workers=workers)
            summaries[p.name] = summary

        best_policy = max(summaries.items(), key=lambda x: x[1].mean_replay_score)[0]
        return ReplayComparisonReport(
            summaries=summaries,
            best_policy_name=best_policy,
            n_trees_evaluated=len(self.simulator_pool),
        )

