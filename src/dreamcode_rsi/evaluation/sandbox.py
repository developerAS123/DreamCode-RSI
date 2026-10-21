from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import structlog

from dreamcode_rsi.config import config

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    wall_time_ms: float
    memory_mb: float
    timed_out: bool = False
    security_violation: bool = False
    error: str | None = None


def _set_resource_limits(cpu_limit_sec: int, memory_limit_bytes: int, pids_limit: int):
    """Linux preexec_fn to enforce resource boundaries on untrusted child processes."""
    try:
        # Set CPU time limit (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_sec, cpu_limit_sec + 1))
    except Exception:
        pass

    try:
        # Set address space / virtual memory limit
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    except Exception:
        pass

    try:
        # Limit number of child processes
        resource.setrlimit(resource.RLIMIT_NPROC, (pids_limit, pids_limit))
    except Exception:
        pass

    try:
        # Limit max output file size (5MB max)
        max_fsize = 5 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize, max_fsize))
    except Exception:
        pass


class SubprocessSandbox:
    """In-process isolated executor utilizing Linux rlimits and strict subprocess control."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        memory_limit_mb: int = 512,
        cpu_limit_seconds: int = 4,
        pids_limit: int = 32,
    ):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_seconds = cpu_limit_seconds
        self.pids_limit = pids_limit

    def execute(self, script_code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="dreamcode_sandbox_") as tmpdir:
            run_file = Path(tmpdir) / "run.py"
            run_file.write_text(script_code, encoding="utf-8")

            # Clean environment without host credentials or secrets
            clean_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": "",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            memory_bytes = self.memory_limit_mb * 1024 * 1024
            cpu_secs = int(self.cpu_limit_seconds)
            pids = self.pids_limit

            start_time = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    ["python3", str(run_file)],
                    cwd=tmpdir,
                    env=clean_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=lambda: _set_resource_limits(cpu_secs, memory_bytes, pids),
                )
                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return ExecutionResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=proc.returncode,
                        wall_time_ms=elapsed_ms,
                        memory_mb=0.0,
                        timed_out=False,
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    return ExecutionResult(
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout_seconds}s",
                        exit_code=-1,
                        wall_time_ms=self.timeout_seconds * 1000.0,
                        memory_mb=0.0,
                        timed_out=True,
                        error="TimeoutExpired",
                    )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=1,
                    wall_time_ms=elapsed_ms,
                    memory_mb=0.0,
                    error=str(exc),
                )


class DockerSandbox:
    """Docker-based isolated container executor matching blueprint specification."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout_seconds: float = 6.0,
        memory_limit_mb: int = 512,
        cpus: str = "2.0",
        pids_limit: int = 64,
    ):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.cpus = cpus
        self.pids_limit = pids_limit

    def is_available(self) -> bool:
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(self, script_code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="dreamcode_docker_") as tmpdir:
            script_path = Path(tmpdir) / "run.py"
            script_path.write_text(script_code, encoding="utf-8")
            os.chmod(script_path, 0o755)

            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                f"--cpus={self.cpus}",
                f"--memory={self.memory_limit_mb}m",
                f"--pids-limit={self.pids_limit}",
                "-v", f"{tmpdir}:/workspace:ro",
                "-w", "/tmp",
                self.image,
                "python", "/workspace/run.py",
            ]

            start_time = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ExecutionResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    wall_time_ms=elapsed_ms,
                    memory_mb=0.0,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Docker container timed out after {self.timeout_seconds}s",
                    exit_code=-1,
                    wall_time_ms=self.timeout_seconds * 1000.0,
                    memory_mb=0.0,
                    timed_out=True,
                    error="TimeoutExpired",
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=1,
                    wall_time_ms=elapsed_ms,
                    memory_mb=0.0,
                    error=str(exc),
                )


class SandboxExecutor:
    """Unified Sandbox with automatic selection and fallback."""

    def __init__(self):
        cfg = config.sandbox
        self.docker = DockerSandbox(
            image=cfg.docker_image,
            timeout_seconds=cfg.timeout_seconds,
            memory_limit_mb=cfg.memory_limit_mb,
            cpus=str(cfg.cpu_limit),
            pids_limit=cfg.pids_limit,
        )
        self.subprocess = SubprocessSandbox(
            timeout_seconds=cfg.timeout_seconds,
            memory_limit_mb=cfg.memory_limit_mb,
            cpu_limit_seconds=int(cfg.cpu_limit),
            pids_limit=cfg.pids_limit,
        )
        self.preferred_mode = cfg.mode

    def execute(self, script_code: str) -> ExecutionResult:
        if self.preferred_mode == "docker" and self.docker.is_available():
            try:
                return self.docker.execute(script_code)
            except Exception as exc:
                logger.warning("Docker execution failed, falling back to subprocess sandbox", error=str(exc))
                return self.subprocess.execute(script_code)
        return self.subprocess.execute(script_code)

