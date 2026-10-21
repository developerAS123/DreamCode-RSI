from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any
import structlog

from dreamcode_rsi.evaluation.benchmarks import BenchmarkTask, get_benchmark
from dreamcode_rsi.evaluation.sandbox import SandboxExecutor, ExecutionResult

logger = structlog.get_logger(__name__)

DISALLOWED_MODULES = {
    "os", "subprocess", "socket", "urllib", "requests",
    "http", "posix", "shutil", "importlib", "ctypes", "sys"
}


@dataclass
class EvaluationReport:
    score: float
    correct: bool
    runtime_ms: float | None
    memory_mb: float | None
    speedup: float | None
    error: str | None
    static_check_passed: bool
    raw_stdout: str = ""
    raw_stderr: str = ""


class CodeEvaluator:
    """Multi-stage evaluator: Static AST Checks -> Sandbox Execution -> Metric Normalization."""

    def __init__(self, sandbox: SandboxExecutor | None = None):
        self.sandbox = sandbox or SandboxExecutor()

    def check_ast_security(self, code: str) -> tuple[bool, str | None]:
        """Perform AST linting to disallow dangerous module imports or dynamic exec."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in DISALLOWED_MODULES:
                        return False, f"SecurityViolation: Import of '{root_pkg}' is prohibited."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in DISALLOWED_MODULES:
                        return False, f"SecurityViolation: Import from '{root_pkg}' is prohibited."
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
                    return False, f"SecurityViolation: Dynamic '{node.func.id}' call is prohibited."

        return True, None

    def evaluate(self, code: str, task: BenchmarkTask) -> EvaluationReport:
        # 1. Static Security & Syntax Check
        passed_static, static_err = self.check_ast_security(code)
        if not passed_static:
            logger.warning("Code failed static check", error=static_err)
            return EvaluationReport(
                score=0.0,
                correct=False,
                runtime_ms=None,
                memory_mb=None,
                speedup=None,
                error=static_err,
                static_check_passed=False,
            )

        # 2. Build full test script from benchmark harness
        script = task.test_harness_template.replace("{candidate_code}", code)

        # 3. Execute in Sandbox
        res: ExecutionResult = self.sandbox.execute(script)

        if res.timed_out:
            return EvaluationReport(
                score=0.0,
                correct=False,
                runtime_ms=res.wall_time_ms,
                memory_mb=0.0,
                speedup=0.0,
                error="Execution timed out",
                static_check_passed=True,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
            )

        # Parse JSON output from stdout
        parsed_out: dict[str, Any] = {}
        for line in reversed(res.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed_out = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        correct = parsed_out.get("correct", False)
        runtime_ms = parsed_out.get("runtime_ms")
        err = parsed_out.get("error") or (res.stderr.strip() if res.exit_code != 0 else None)

        if not correct or runtime_ms is None:
            return EvaluationReport(
                score=0.0,
                correct=False,
                runtime_ms=runtime_ms,
                memory_mb=res.memory_mb,
                speedup=None,
                error=err or "Incorrect output or assertion failed",
                static_check_passed=True,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
            )

        # 4. Score Normalization
        baseline_ms = task.baseline_runtime_ms
        speedup = baseline_ms / max(0.1, runtime_ms)

        if speedup <= 1.0:
            # Slower than or equal to baseline: scaled from 0.20 to 0.50
            score = max(0.20, 0.50 * speedup)
        else:
            # Faster than baseline: scaled from 0.50 to 1.00 (reaches 1.0 at 4x speedup)
            score = min(1.0, 0.50 + 0.50 * ((speedup - 1.0) / 3.0))

        score = round(score, 4)
        speedup = round(speedup, 2)

        return EvaluationReport(
            score=score,
            correct=True,
            runtime_ms=round(runtime_ms, 2),
            memory_mb=round(res.memory_mb, 2),
            speedup=speedup,
            error=None,
            static_check_passed=True,
            raw_stdout=res.stdout,
            raw_stderr=res.stderr,
        )

