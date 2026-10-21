from __future__ import annotations

from typing import TypedDict, Any, List, Dict


class DreamState(TypedDict):
    experiment_id: str
    task_id: str
    round_index: int
    max_rounds: int
    policy_code: str
    policy_version: str
    selected_agent: str
    routing: Dict[str, Any]
    candidate_code: str | None
    candidate_version: str | None
    candidate_promoted: bool
    retry_count: int
    latest_results: List[Dict[str, Any]]
    history_ids: List[str]
    best_score: float
    status: str
    current_tree_id: str | None

