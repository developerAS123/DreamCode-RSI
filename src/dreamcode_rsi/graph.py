from __future__ import annotations

import uuid
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
import structlog

from dreamcode_rsi.config import config
from dreamcode_rsi.state import DreamState
from dreamcode_rsi.evaluation.benchmarks import get_benchmark
from dreamcode_rsi.evaluation.evaluator import CodeEvaluator
from dreamcode_rsi.discovery.node import DiscoveryNode
from dreamcode_rsi.discovery.tree import DiscoveryTree
from dreamcode_rsi.discovery.agent import DiscoveryAgent, MATMUL_CANDIDATE_STRATEGIES
from dreamcode_rsi.discovery.history import TreeHistoryManager
from dreamcode_rsi.routing.laya_router import LayaRouter
from dreamcode_rsi.policies.fixed import FixedPolicy
from dreamcode_rsi.policies.evolving import DynamicPolicy
from dreamcode_rsi.replay.runner import ReplayRunner
from dreamcode_rsi.improvement.developer import PolicyDeveloper
from dreamcode_rsi.improvement.validator import PromotionGate
from dreamcode_rsi.storage.db import SessionLocal, init_db
from dreamcode_rsi.storage.models import (
    ExperimentRecord,
    DiscoveryTreeRecord,
    DiscoveryNodeRecord,
    RoutingTraceRecord,
    PolicyVersionRecord,
)

logger = structlog.get_logger(__name__)


# Global singletons for graph execution
_evaluator = CodeEvaluator()
_agent = DiscoveryAgent(evaluator=_evaluator)
_history_mgr = TreeHistoryManager()
_router = LayaRouter()
_developer = PolicyDeveloper()


def route_with_laya(state: DreamState) -> dict[str, Any]:
    task = get_benchmark(state["task_id"])
    route = _router.route(task.description)

    # Persist routing trace
    try:
        with SessionLocal() as session:
            trace_rec = RoutingTraceRecord(
                id=uuid.uuid4().hex[:16],
                experiment_id=state["experiment_id"],
                task_id=task.task_id,
                selected_agent=route.selected_agent,
                confidence=route.confidence,
                probabilities_json=route.probabilities,
                fallback_used=route.fallback_used,
            )
            session.add(trace_rec)
            session.commit()
    except Exception as exc:
        logger.warning("Failed to persist routing trace", error=str(exc))

    return {
        "selected_agent": route.selected_agent,
        "routing": route.model_dump(),
        "status": f"routed_to_{route.selected_agent}",
    }


def run_online_exploration(state: DreamState) -> dict[str, Any]:
    task = get_benchmark(state["task_id"])
    round_id = state["round_index"]
    current_best = state.get("best_score", 0.0)

    # Instantiate current exploration policy
    policy_code = state.get("policy_code")
    if policy_code and "class " in policy_code:
        try:
            policy = DynamicPolicy(code=policy_code, name=state.get("policy_version", "active_policy"))
        except Exception:
            policy = FixedPolicy()
    else:
        policy = FixedPolicy()

    # Initialize new tree for this round
    tree = DiscoveryTree(problem_id=task.task_id, round_id=round_id)

    # 1. Add baseline root
    base_eval = _evaluator.evaluate(task.baseline_code, task)
    root_node = DiscoveryNode(
        node_id=f"root_r{round_id}",
        parent_id=None,
        problem_id=task.task_id,
        agent_id="baseline",
        proposal="Naive baseline implementation",
        artifact_uri="uri://baseline",
        score=base_eval.score,
        correct=base_eval.correct,
        runtime_ms=base_eval.runtime_ms,
        memory_mb=base_eval.memory_mb,
    )
    tree.add_node(root_node)
    current_best = max(current_best, root_node.score)

    # 2. Explore branches in bounded rounds guided by policy
    active_parents = [root_node.node_id]
    latest_nodes: list[dict[str, Any]] = [root_node.model_dump(mode="json")]

    for step in range(len(MATMUL_CANDIDATE_STRATEGIES)):
        parent_id = active_parents[step % len(active_parents)]
        parent_node = tree.get_node(parent_id)

        child = _agent.generate_candidate(
            problem=task,
            parent_node=parent_node,
            strategy_index=step,
            round_id=round_id,
            agent_id=state.get("selected_agent", "algorithm"),
        )
        tree.add_node(child)
        latest_nodes.append(child.model_dump(mode="json"))

        if child.correct and child.score > 0.5:
            active_parents.append(child.node_id)
        if child.score > current_best:
            current_best = child.score

    # Save tree to disk
    tree_file = _history_mgr.save_tree(tree, experiment_id=f"{state['experiment_id']}_r{round_id}")

    return {
        "best_score": round(current_best, 4),
        "latest_results": latest_nodes,
        "current_tree_id": f"{state['experiment_id']}_r{round_id}",
        "status": f"online_exploration_round_{round_id}_complete",
    }


def freeze_tree(state: DreamState) -> dict[str, Any]:
    tree_id = state["current_tree_id"]
    if not tree_id:
        return {"status": "freeze_tree_skipped"}

    # Persist tree metadata to SQLite / Supabase
    try:
        tree_path = _history_mgr.base_dir / f"{tree_id}.json"
        tree = _history_mgr.load_tree(tree_path)

        with SessionLocal() as session:
            tree_rec = DiscoveryTreeRecord(
                id=tree_id,
                experiment_id=state["experiment_id"],
                round_id=state["round_index"],
                problem_id=state["task_id"],
                root_id=tree.root_id,
                node_count=tree.size(),
                tree_json=tree.to_dict(),
            )
            session.merge(tree_rec)

            for n in tree.nodes.values():
                node_rec = DiscoveryNodeRecord(
                    id=n.node_id,
                    tree_id=tree_id,
                    parent_id=n.parent_id,
                    agent_id=n.agent_id,
                    proposal=n.proposal,
                    artifact_uri=n.artifact_uri,
                    score=n.score,
                    correct=n.correct,
                    runtime_ms=n.runtime_ms,
                    depth=n.depth,
                    round_id=state["round_index"],
                )
                session.merge(node_rec)

            session.commit()
    except Exception as exc:
        logger.warning("Failed to persist frozen tree to DB", error=str(exc))

    return {"status": "tree_frozen"}


def add_to_simulator_pool(state: DreamState) -> dict[str, Any]:
    tree_id = state["current_tree_id"]
    history_ids = list(state.get("history_ids", []))
    if tree_id and tree_id not in history_ids:
        history_ids.append(tree_id)
    return {"history_ids": history_ids, "status": "simulator_pool_updated"}


def replay_candidate_policies(state: DreamState) -> dict[str, Any]:
    pool = _history_mgr.get_simulator_pool(state["task_id"])
    if not pool:
        return {"status": "no_trees_in_pool"}

    runner = ReplayRunner(pool)
    current_policy = FixedPolicy(name=state.get("policy_version", "incumbent"))
    summary = runner.evaluate_policy(current_policy)

    return {
        "status": "replay_complete",
        "latest_results": [
            {
                "type": "replay_summary",
                "policy": current_policy.name,
                "score": summary.mean_replay_score,
                "quality": summary.mean_best_quality,
            }
        ],
    }


def develop_next_policy(state: DreamState) -> dict[str, Any]:
    current_code = state.get("policy_code") or FixedPolicy().describe()["description"]
    round_idx = state["round_index"]

    candidate = _developer.improve_policy(
        current_policy_code=current_code,
        replay_report=f"Round {round_idx} replay summary. Best score: {state.get('best_score')}",
        round_index=round_idx,
    )

    return {
        "candidate_code": candidate.source_code,
        "candidate_version": candidate.version,
        "candidate_promoted": False,
        "status": f"developed_candidate_{candidate.version}",
    }


def validate_candidate(state: DreamState) -> dict[str, Any]:
    pool = _history_mgr.get_simulator_pool(state["task_id"])
    if not pool:
        # Self-contained fallback tree
        train_pool = [_history_mgr.load_tree(_history_mgr.base_dir / f"{state['current_tree_id']}.json")]
        val_pool = train_pool
    elif len(pool) == 1:
        train_pool = pool
        val_pool = pool
    else:
        # Split into training set and reserved validation set
        train_pool = pool[:-1]
        val_pool = pool[-1:]

    gate = PromotionGate(train_pool, val_pool, min_improvement_margin=0.0)

    # Incumbent
    curr_code = state.get("policy_code")
    incumbent = DynamicPolicy(code=curr_code) if curr_code else FixedPolicy()

    cand_code = state.get("candidate_code", "")
    cand_ver = state.get("candidate_version", "candidate")

    report = gate.validate_and_decide(cand_code, incumbent, candidate_name=cand_ver)

    # Persist policy version record
    try:
        with SessionLocal() as session:
            p_rec = PolicyVersionRecord(
                id=uuid.uuid4().hex[:16],
                version=cand_ver,
                source_code=cand_code,
                rationale=report.reason,
                generation=state["round_index"],
                train_score=report.candidate_train_score,
                val_score=report.candidate_val_score,
                promoted=report.promoted,
            )
            session.add(p_rec)
            session.commit()
    except Exception as exc:
        logger.warning("Failed to persist policy record", error=str(exc))

    return {
        "candidate_promoted": report.promoted,
        "retry_count": state.get("retry_count", 0) + 1,
        "status": f"validation_decision_{'promoted' if report.promoted else 'rejected'}",
    }


def should_promote(state: DreamState) -> Literal["promote_policy", "retry_develop"]:
    if state.get("candidate_promoted", False):
        return "promote_policy"
    if state.get("retry_count", 0) < 1:
        return "retry_develop"
    return "promote_policy"


def promote_policy(state: DreamState) -> dict[str, Any]:
    if state.get("candidate_promoted", False):
        new_code = state.get("candidate_code", state.get("policy_code", ""))
        new_ver = state.get("candidate_version", state.get("policy_version", "incumbent"))
    else:
        # Keep incumbent
        new_code = state.get("policy_code", "")
        new_ver = state.get("policy_version", "incumbent")

    next_round = state["round_index"] + 1

    return {
        "policy_code": new_code,
        "policy_version": new_ver,
        "round_index": next_round,
        "retry_count": 0,
        "status": f"promoted_policy_for_round_{next_round}",
    }


def should_continue(state: DreamState) -> Literal["run_online_exploration", "end"]:
    if state["round_index"] <= state["max_rounds"]:
        return "run_online_exploration"
    return "end"


def build_dream_graph() -> Any:
    """Build the end-to-end LangGraph StateMachine."""
    builder = StateGraph(DreamState)

    builder.add_node("route_with_laya", route_with_laya)
    builder.add_node("run_online_exploration", run_online_exploration)
    builder.add_node("freeze_tree", freeze_tree)
    builder.add_node("add_to_simulator_pool", add_to_simulator_pool)
    builder.add_node("replay_candidate_policies", replay_candidate_policies)
    builder.add_node("develop_next_policy", develop_next_policy)
    builder.add_node("validate_candidate", validate_candidate)
    builder.add_node("promote_policy", promote_policy)

    # Connections
    builder.add_edge(START, "route_with_laya")
    builder.add_edge("route_with_laya", "run_online_exploration")
    builder.add_edge("run_online_exploration", "freeze_tree")
    builder.add_edge("freeze_tree", "add_to_simulator_pool")
    builder.add_edge("add_to_simulator_pool", "replay_candidate_policies")
    builder.add_edge("replay_candidate_policies", "develop_next_policy")
    builder.add_edge("develop_next_policy", "validate_candidate")

    builder.add_conditional_edges(
        "validate_candidate",
        should_promote,
        {
            "promote_policy": "promote_policy",
            "retry_develop": "develop_next_policy",
        },
    )

    builder.add_conditional_edges(
        "promote_policy",
        should_continue,
        {
            "run_online_exploration": "run_online_exploration",
            "end": END,
        },
    )

    return builder.compile()

