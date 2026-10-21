from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AgentRoute(BaseModel):
    selected_agent: str
    probabilities: dict[str, float]
    confidence: float
    fallback_used: bool = False
    needs_parallelism: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

