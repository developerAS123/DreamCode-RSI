from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplayScoreBreakdown:
    total_score: float
    best_quality: float
    attempts: int
    rounds: int
    cost_penalty: float
    parallel_bonus: float
    parallel_efficiency: float


def compute_replay_score(
    best_quality: float,
    attempts: int,
    rounds: int,
    beta_cost: float = 0.01,
    beta_parallel: float = 0.05,
) -> ReplayScoreBreakdown:
    """
    Direct implementation of Dream-RSI replay scoring objective:
      score = best_quality - (beta_cost * attempts) + (beta_parallel * parallel_efficiency)
    where parallel_efficiency = attempts / max(1, rounds).
    """
    safe_rounds = max(1, rounds)
    parallel_efficiency = attempts / safe_rounds
    cost_penalty = beta_cost * attempts
    parallel_bonus = beta_parallel * parallel_efficiency

    total = best_quality - cost_penalty + parallel_bonus
    return ReplayScoreBreakdown(
        total_score=round(total, 4),
        best_quality=round(best_quality, 4),
        attempts=attempts,
        rounds=rounds,
        cost_penalty=round(cost_penalty, 4),
        parallel_bonus=round(parallel_bonus, 4),
        parallel_efficiency=round(parallel_efficiency, 3),
    )

