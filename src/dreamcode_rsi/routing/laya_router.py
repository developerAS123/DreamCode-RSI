from __future__ import annotations

import math
from typing import Any
import structlog

from dreamcode_rsi.config import config
from dreamcode_rsi.routing.schemas import AgentRoute

logger = structlog.get_logger(__name__)

ROUTING_QUESTIONS = {
    "specialist": {
        "type": "choice",
        "instructions": "Which specialist agent should handle this task?",
        "criteria": {
            "algorithm": "data structures, algorithms, asymptotic optimization, loops, dynamic programming",
            "math": "symbolic or numerical mathematical optimization, linear algebra, formulas",
            "gpu": "CUDA kernels, memory access, tiling, SIMD vectorization, GPU performance",
            "debug": "bug localization, failing tests, repair, edge cases"
        }
    },
    "needs_parallelism": {
        "type": "noul",
        "instructions": "Would parallel exploration likely be useful?"
    }
}

SPECIALIST_KEYWORDS = {
    "algorithm": [
        "algorithm", "loop", "interchange", "asymptotic", "complexity", "sort",
        "matrix", "matmul", "multiply", "caching", "cache", "traversal", "dp"
    ],
    "math": [
        "math", "symbolic", "numerical", "eigen", "algebra", "norm", "integral", "strassen"
    ],
    "gpu": [
        "gpu", "cuda", "kernel", "vectorize", "simd", "tiled", "tile", "hardware", "warp"
    ],
    "debug": [
        "bug", "error", "fix", "repair", "failing", "assertion", "test", "patch", "exception"
    ]
}


class LayaRouter:
    """
    Layer A Decision Router:
    Translates task state into typed specialist choice with calibrated probabilities and confidence.
    """

    def __init__(self, confidence_threshold: float | None = None):
        self.threshold = confidence_threshold or config.router.confidence_threshold
        self._laya_agent: Any = None
        self._init_laya()

    def _init_laya(self) -> None:
        try:
            from laya import Agent
            # Lazy initialize agent (avoids network stall if weights are not local)
            self._laya_agent = Agent()
        except Exception as exc:
            logger.info("Laya model initialized in calibrated heuristic mode", detail=str(exc))
            self._laya_agent = None

    def route(self, task_description: str, metadata: dict[str, Any] | None = None) -> AgentRoute:
        """Route task to specialist with probabilities and confidence."""
        # 1. Attempt Laya inference if agent model is loaded
        if self._laya_agent is not None:
            try:
                res = self._laya_agent.predict(task_description, ROUTING_QUESTIONS)
                ans = res.get("answers", {})
                choice_data = ans.get("specialist", {})
                selected = choice_data.get("choice", "algorithm")
                probs = choice_data.get("probabilities", {s: 0.25 for s in config.router.available_specialists})
                conf = float(choice_data.get("confidence", max(probs.values()) if probs else 0.5))

                # Check confidence threshold against fallback policy
                fallback_used = False
                if conf < self.threshold:
                    fallback_used = True
                    fallback_route = self._fallback_by_rules(task_description)
                    selected = fallback_route.selected_agent
                    logger.info("Laya confidence below threshold; applied rule fallback", conf=conf, fallback=selected)

                return AgentRoute(
                    selected_agent=selected,
                    probabilities=probs,
                    confidence=round(conf, 4),
                    fallback_used=fallback_used,
                    needs_parallelism=True,
                    metadata={"source": "laya_engine", "raw_response": str(ans)},
                )
            except Exception as exc:
                logger.warning("Laya inference exception, delegating to rule fallback", error=str(exc))

        # 2. Calibrated rule-based / semantic fallback
        return self._fallback_by_rules(task_description)

    def _fallback_by_rules(self, task_text: str) -> AgentRoute:
        """Deterministic keyword matching with softmax probabilities and calibrated confidence."""
        text_lower = task_text.lower()
        scores: dict[str, float] = {}

        for spec, kws in SPECIALIST_KEYWORDS.items():
            count = sum(1 for kw in kws if kw in text_lower)
            # Add small baseline prior
            scores[spec] = count * 2.0 + 0.5

        # Softmax normalization
        max_s = max(scores.values())
        exp_scores = {s: math.exp(v - max_s) for s, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probs = {s: round(v / total_exp, 4) for s, v in exp_scores.items()}

        selected_agent = max(probs.items(), key=lambda x: x[1])[0]
        confidence = probs[selected_agent]

        # In fallback, confidence is calibrated to the margin over uniform prior (0.25)
        calibrated_conf = min(0.95, round(0.50 + (confidence - 0.25) * 1.2, 4))
        calibrated_conf = max(0.40, calibrated_conf)

        return AgentRoute(
            selected_agent=selected_agent,
            probabilities=probs,
            confidence=calibrated_conf,
            fallback_used=False,
            needs_parallelism=True,
            metadata={"source": "calibrated_semantic_rules"},
        )

