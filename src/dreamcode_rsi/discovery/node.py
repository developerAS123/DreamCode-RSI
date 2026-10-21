from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class DiscoveryNode(BaseModel):
    node_id: str
    parent_id: str | None
    problem_id: str
    agent_id: str
    proposal: str  # Natural language description of approach
    artifact_uri: str  # Path to generated code file
    score: float  # Normalized quality score (0-1)
    correct: bool
    runtime_ms: float | None = None
    memory_mb: float | None = None
    error: str | None = None
    depth: int = 0
    round_id: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_root(self) -> bool:
        return self.parent_id is None

