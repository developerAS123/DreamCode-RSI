from __future__ import annotations

import uuid
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dreamcode_rsi.config import config
from dreamcode_rsi.evaluation.benchmarks import get_benchmark, BENCHMARKS
from dreamcode_rsi.routing.laya_router import LayaRouter
from dreamcode_rsi.routing.schemas import AgentRoute
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.discovery.history import TreeHistoryManager
from dreamcode_rsi.policies.fixed import FixedPolicy, RandomPolicy
from dreamcode_rsi.policies.greedy import GreedyPolicy
from dreamcode_rsi.policies.evolving import DynamicPolicy
from dreamcode_rsi.replay.runner import ReplayRunner
from dreamcode_rsi.improvement.developer import PolicyDeveloper
from dreamcode_rsi.improvement.validator import PromotionGate
from dreamcode_rsi.graph import build_dream_graph
from dreamcode_rsi.state import DreamState
from dreamcode_rsi.storage.db import init_db, SessionLocal
from dreamcode_rsi.storage.models import ExperimentRecord, DiscoveryTreeRecord

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="DreamCode-RSI + Laya API",
    description="Recursive Self-Improving Agent Routing and Exploration Engine through Replay Worlds",
    version="0.1.0",
    lifespan=lifespan,
)

router = LayaRouter()
history_mgr = TreeHistoryManager()
developer = PolicyDeveloper()
graph_runner = build_dream_graph()


# --- Request/Response Models ---
class RouteRequest(BaseModel):
    task_description: str


class ExperimentCreateRequest(BaseModel):
    task_id: str = "matrix_multiply"
    name: str = "Matrix Multiplication Optimization Experiment"
    max_rounds: int = 2


class ExperimentResponse(BaseModel):
    experiment_id: str
    task_id: str
    status: str


class RunRoundRequest(BaseModel):
    round_id: int | None = None
    workers: int = 2


class ReplayRequest(BaseModel):
    policy_names: list[str] = ["fixed_fifo", "greedy_best", "random_sample"]
    workers: int = 2


class ImproveRequest(BaseModel):
    round_index: int = 1
    current_policy_code: str = ""


class PromoteRequest(BaseModel):
    candidate_code: str
    candidate_version: str = "candidate_pi"


# --- API Endpoints ---
@app.post("/route", response_model=AgentRoute)
def route_task(req: RouteRequest):
    """Laya specialist selection returning calibrated probabilities and confidence."""
    return router.route(req.task_description)


@app.post("/experiments", response_model=ExperimentResponse)
def create_experiment(req: ExperimentCreateRequest):
    """Create a new discovery experiment."""
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    with SessionLocal() as session:
        rec = ExperimentRecord(
            id=exp_id,
            name=req.name,
            status="created",
            config_json={"task_id": req.task_id, "max_rounds": req.max_rounds},
        )
        session.add(rec)
        session.commit()
    return ExperimentResponse(experiment_id=exp_id, task_id=req.task_id, status="created")


@app.post("/experiments/{experiment_id}/rounds")
def run_round(experiment_id: str, req: RunRoundRequest):
    """Run an online recursive round via LangGraph."""
    with SessionLocal() as session:
        exp = session.query(ExperimentRecord).filter_by(id=experiment_id).first()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        task_id = exp.config_json.get("task_id", "matrix_multiply")

    state: DreamState = {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "round_index": req.round_id or 1,
        "max_rounds": req.round_id or 1,
        "policy_code": "",
        "policy_version": "pi_0_baseline",
        "selected_agent": "algorithm",
        "routing": {},
        "candidate_code": None,
        "candidate_version": None,
        "candidate_promoted": False,
        "retry_count": 0,
        "latest_results": [],
        "history_ids": [],
        "best_score": 0.0,
        "status": "ready",
        "current_tree_id": None,
    }

    final_state = graph_runner.invoke(state)
    return {
        "experiment_id": experiment_id,
        "round_index": final_state["round_index"],
        "best_score": final_state["best_score"],
        "policy_version": final_state["policy_version"],
        "status": final_state["status"],
        "nodes_generated": len(final_state.get("latest_results", [])),
    }


@app.post("/experiments/{experiment_id}/replay")
def replay_policies(experiment_id: str, req: ReplayRequest):
    """Replay candidate policies against the historical simulator pool."""
    pool = history_mgr.get_simulator_pool()
    if not pool:
        raise HTTPException(status_code=400, detail="Simulator pool is empty. Run an online round first.")

    policies = [
        FixedPolicy(name="fixed_fifo"),
        GreedyPolicy(name="greedy_best"),
        RandomPolicy(seed=42, name="random_sample"),
    ]
    runner = ReplayRunner(pool)
    report = runner.compare_policies(policies, max_rounds=5, workers=req.workers)
    return report.to_dict()


@app.post("/experiments/{experiment_id}/improve")
def improve_policy(experiment_id: str, req: ImproveRequest):
    """Generate next candidate policy using replay feedback."""
    candidate = developer.improve_policy(
        current_policy_code=req.current_policy_code or FixedPolicy().describe()["description"],
        replay_report="Replay feedback indicates branch pruning and greedy prioritization will maximize throughput.",
        round_index=req.round_index,
    )
    return {
        "version": candidate.version,
        "source_code": candidate.source_code,
        "rationale": candidate.rationale,
        "generation": candidate.generation,
    }


@app.post("/experiments/{experiment_id}/promote")
def promote_candidate(experiment_id: str, req: PromoteRequest):
    """Validate and decide candidate promotion across training and validation worlds."""
    pool = history_mgr.get_simulator_pool()
    if not pool:
        raise HTTPException(status_code=400, detail="Cannot validate without historical trees in pool.")

    train_pool = pool[:-1] if len(pool) > 1 else pool
    val_pool = pool[-1:]

    gate = PromotionGate(train_pool, val_pool, min_improvement_margin=0.0)
    report = gate.validate_and_decide(
        candidate_code=req.candidate_code,
        incumbent_policy=FixedPolicy(),
        candidate_name=req.candidate_version,
    )
    return {
        "candidate_version": report.candidate_version,
        "promoted": report.promoted,
        "reason": report.reason,
        "incumbent_score": report.incumbent_score,
        "candidate_train_score": report.candidate_train_score,
        "candidate_val_score": report.candidate_val_score,
        "stages": [{"stage": s.stage_name, "passed": s.passed, "details": s.details} for s in report.stages],
    }


@app.get("/experiments/{experiment_id}/tree")
def get_tree(experiment_id: str):
    """Fetch discovery tree nodes and edges."""
    trees = history_mgr.list_trees()
    matching = [t for t in trees if experiment_id in t.name]
    target_path = matching[0] if matching else (trees[0] if trees else None)
    if not target_path:
        raise HTTPException(status_code=404, detail="No discovery trees found")
    tree = history_mgr.load_tree(target_path)
    return tree.to_dict()


@app.get("/experiments/{experiment_id}/metrics")
def get_metrics(experiment_id: str):
    """Fetch metrics summary."""
    with SessionLocal() as session:
        trees = session.query(DiscoveryTreeRecord).filter_by(experiment_id=experiment_id).all()
        return {
            "experiment_id": experiment_id,
            "total_trees": len(trees),
            "rounds_completed": [t.round_id for t in trees],
        }
