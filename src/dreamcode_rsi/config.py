from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
CACHE_DIR = DATA_DIR / "cache"

for p in (DATA_DIR, EXPERIMENTS_DIR, ARTIFACTS_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)


class SandboxConfig(BaseModel):
    mode: str = Field(
        default="subprocess",
        description="'docker' or 'subprocess'. Automatically falls back if Docker daemon is unreachable."
    )
    docker_image: str = "python:3.12-slim"
    timeout_seconds: float = 5.0
    cpu_limit: float = 2.0
    memory_limit_mb: int = 512
    max_output_bytes: int = 65536
    network_enabled: bool = False
    pids_limit: int = 32


class RouterConfig(BaseModel):
    confidence_threshold: float = 0.70
    default_specialist: str = "algorithm"
    available_specialists: list[str] = ["algorithm", "math", "gpu", "debug"]


class ReplayConfig(BaseModel):
    beta_cost: float = 0.01
    beta_parallel: float = 0.05
    max_replay_rounds: int = 10
    default_workers: int = 2


class AppConfig(BaseModel):
    environment: str = os.getenv("ENVIRONMENT", "development")
    db_url: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/dreamcode.db")
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    hf_token: str | None = os.getenv("HF_TOKEN")


config = AppConfig()

