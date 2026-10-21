from __future__ import annotations

import os
from dataclasses import dataclass
import httpx
import structlog

from dreamcode_rsi.config import config
from dreamcode_rsi.replay.runner import ReplayComparisonReport

logger = structlog.get_logger(__name__)

POLICY_PROMPT_TEMPLATE = """You are improving an exploration controller for a recursive self-improving code search engine.
You may modify ONLY the policy implementation.
The controller chooses which recorded discovery branches to open, how many branches to batch, and when to stop.

Replay evidence:
{replay_report}

Current policy:
```python
{policy_source}
```

Constraints:
1. Must define a class implementing select_batch(self, available_nodes: list[DiscoveryNode], workers: int) -> list[DiscoveryNode]
2. Must have a describe(self) -> dict method.
3. No external OS or network calls.
4. Output ONLY valid Python code block inside ```python ... ``` followed by a short rationale.
"""

EVOLUTION_CATALOGUE = [
    # Iteration 1: Score-prioritized greedy batching with depth tiebreaker
    {
        "version": "pi_1_greedy_priority",
        "rationale": "Prioritize nodes with higher direct score and depth discount to favor shallow high-impact branches.",
        "code": '''
class EvolvedExplorationPolicy:
    def __init__(self, name: str = "pi_1_greedy_priority"):
        self.name = name

    def select_batch(self, available_nodes: list[DiscoveryNode], workers: int) -> list[DiscoveryNode]:
        if not available_nodes:
            return []
        # Score priority with slight depth penalty to prefer shallow high-payoff nodes
        def priority(n):
            depth_penalty = 0.05 * getattr(n, "depth", 0)
            score = getattr(n, "score", 0.0)
            correct_bonus = 0.2 if getattr(n, "correct", False) else 0.0
            return score + correct_bonus - depth_penalty

        sorted_nodes = sorted(available_nodes, key=priority, reverse=True)
        return sorted_nodes[:max(1, workers)]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rule": "score + correct_bonus - depth_penalty",
            "generation": 1
        }
'''.strip()
    },
    # Iteration 2: Parallel batch diversity with category spread
    {
        "version": "pi_2_diverse_parallel",
        "rationale": "Maximizes batch diversity across different agent specialists and strategies to prevent redundant worker rollouts.",
        "code": '''
class EvolvedExplorationPolicy:
    def __init__(self, name: str = "pi_2_diverse_parallel"):
        self.name = name

    def select_batch(self, available_nodes: list[DiscoveryNode], workers: int) -> list[DiscoveryNode]:
        if not available_nodes:
            return []
        
        # Sort by score
        candidates = sorted(available_nodes, key=lambda n: (n.correct, n.score), reverse=True)
        
        # Select diverse agents across the worker batch
        selected = []
        seen_agents = set()
        
        for n in candidates:
            if len(selected) >= workers:
                break
            agent = getattr(n, "agent_id", "default")
            if agent not in seen_agents or len(selected) + (len(candidates) - len(selected)) <= workers:
                selected.append(n)
                seen_agents.add(agent)
                
        # Fill remaining slots if any
        for n in candidates:
            if len(selected) >= workers:
                break
            if n not in selected:
                selected.append(n)
                
        return selected

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rule": "diverse_agent_selection + score_sorting",
            "generation": 2
        }
'''.strip()
    },
    # Iteration 3: Cost-aware early stopping & threshold filtering
    {
        "version": "pi_3_cost_aware_threshold",
        "rationale": "Filters out historically failing branches (score==0) and stops spending worker budget on stagnant frontiers.",
        "code": '''
class EvolvedExplorationPolicy:
    def __init__(self, name: str = "pi_3_cost_aware_threshold"):
        self.name = name

    def select_batch(self, available_nodes: list[DiscoveryNode], workers: int) -> list[DiscoveryNode]:
        if not available_nodes:
            return []
            
        # Filter out nodes known to fail or invalid syntax
        valid_nodes = [n for n in available_nodes if n.correct or n.score > 0.1]
        pool = valid_nodes if valid_nodes else available_nodes
        
        # Rank by score-to-cost ratio
        def rank_fn(n):
            runtime = getattr(n, "runtime_ms", 10.0) or 10.0
            speed_factor = 10.0 / max(1.0, runtime)
            return (n.score * 1.5) + (speed_factor * 0.1)

        ranked = sorted(pool, key=rank_fn, reverse=True)
        return ranked[:max(1, workers)]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rule": "cost_aware_filtering + speed_factor",
            "generation": 3
        }
'''.strip()
    }
]


@dataclass
class PolicyCandidate:
    version: str
    source_code: str
    rationale: str
    generation: int


class PolicyDeveloper:
    """Agent that writes and improves exploration policy code based on replay evidence."""

    def __init__(self):
        self.gemini_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")

    def improve_policy(
        self,
        current_policy_code: str,
        replay_report: str | ReplayComparisonReport,
        round_index: int = 1,
    ) -> PolicyCandidate:
        report_text = (
            replay_report.format_text_report()
            if isinstance(replay_report, ReplayComparisonReport)
            else str(replay_report)
        )

        prompt = POLICY_PROMPT_TEMPLATE.format(
            replay_report=report_text,
            policy_source=current_policy_code,
        )

        # 1. Try Gemini API if key is configured
        if self.gemini_key:
            try:
                candidate = self._call_gemini(prompt, round_index)
                if candidate:
                    return candidate
            except Exception as exc:
                logger.warning("Gemini policy improvement call failed, using heuristic evolution", error=str(exc))

        # 2. Try OpenAI API if key is configured
        if self.openai_key:
            try:
                candidate = self._call_openai(prompt, round_index)
                if candidate:
                    return candidate
            except Exception as exc:
                logger.warning("OpenAI policy improvement call failed, using heuristic evolution", error=str(exc))

        # 3. Fallback to research-calibrated heuristic meta-mutations
        meta = EVOLUTION_CATALOGUE[(round_index - 1) % len(EVOLUTION_CATALOGUE)]
        return PolicyCandidate(
            version=meta["version"],
            source_code=meta["code"],
            rationale=meta["rationale"],
            generation=round_index,
        )

    def _call_gemini(self, prompt: str, round_index: int) -> PolicyCandidate | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4}
        }
        resp = httpx.post(url, json=payload, timeout=20.0)
        if resp.status_code == 200:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_llm_response(text, round_index)
        return None

    def _call_openai(self, prompt: str, round_index: int) -> PolicyCandidate | None:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=20.0)
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return self._parse_llm_response(text, round_index)
        return None

    def _parse_llm_response(self, text: str, round_index: int) -> PolicyCandidate:
        code = ""
        rationale = "Generated by LLM policy improvement agent."
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
            rationale_parts = text.split("```")
            if len(rationale_parts) > 2:
                rationale = rationale_parts[2].strip()
        elif "```" in text:
            code = text.split("```")[1].split("```")[0].strip()
        else:
            code = text.strip()

        return PolicyCandidate(
            version=f"pi_{round_index}_llm_evolved",
            source_code=code,
            rationale=rationale[:300],
            generation=round_index,
        )

