from __future__ import annotations

import pytest
from dreamcode_rsi.routing.laya_router import LayaRouter
from dreamcode_rsi.storage.db import init_db, SessionLocal
from dreamcode_rsi.storage.models import ExperimentRecord, RoutingTraceRecord
from dreamcode_rsi.graph import build_dream_graph
from dreamcode_rsi.state import DreamState


def test_laya_router_decisions_and_confidence():
    router = LayaRouter(confidence_threshold=0.70)

    # 1. Algorithm problem
    route_algo = router.route("Optimize matrix multiplication cache locality and loop traversal.")
    assert route_algo.selected_agent in ["algorithm", "gpu"]
    assert "algorithm" in route_algo.probabilities
    assert route_algo.confidence >= 0.50
    assert route_algo.needs_parallelism is True

    # 2. Debug problem
    route_debug = router.route("Fix failing test assertion error and repair index out of bounds.")
    assert route_debug.selected_agent == "debug"
    assert route_debug.probabilities["debug"] > 0.30

    # 3. GPU problem
    route_gpu = router.route("Optimize CUDA kernel shared memory tiling and SIMD warp occupancy.")
    assert route_gpu.selected_agent == "gpu"


def test_db_persistence_initialization():
    init_db()
    with SessionLocal() as session:
        exp = ExperimentRecord(
            id="exp_test_001",
            name="Matrix Multiplication RSI Test",
            status="running",
            best_score=0.88,
        )
        session.merge(exp)
        session.commit()

        loaded = session.query(ExperimentRecord).filter_by(id="exp_test_001").first()
        assert loaded is not None
        assert loaded.name == "Matrix Multiplication RSI Test"
        assert loaded.best_score == 0.88


def test_full_langgraph_two_round_execution():
    """
    CRITICAL RESEARCH MILESTONE (WEEK 4):
    Execute the dual-level LangGraph state machine across 2 rounds:
    Laya Routing -> Online Round 1 -> Freeze -> Replay -> Develop pi_1 -> Validate -> Promote -> Online Round 2 -> END.
    """
    init_db()
    app = build_dream_graph()

    initial_state: DreamState = {
        "experiment_id": "test_graph_exp_01",
        "task_id": "matrix_multiply",
        "round_index": 1,
        "max_rounds": 2,  # 2 recursive rounds
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
        "status": "initialized",
        "current_tree_id": None,
    }

    final_state = app.invoke(initial_state)

    print("\n--- LangGraph Execution Result ---")
    print("Final Status:", final_state["status"])
    print("Final Round Index:", final_state["round_index"])
    print("Active Policy Version:", final_state["policy_version"])
    print("Best Evaluator Score:", final_state["best_score"])
    print("History Tree IDs:", final_state["history_ids"])

    assert final_state["round_index"] > 2  # Completed 2 rounds and incremented past max_rounds
    assert final_state["best_score"] > 0.50
    assert len(final_state["history_ids"]) >= 2
    assert final_state["policy_version"] != "pi_0_baseline"  # Policy evolved!

