import sys
from pathlib import Path

# Required for Hugging Face ZeroGPU startup scanner
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

if has_spaces:
    @spaces.GPU(duration=60)
    def gpu_task_marker(task_choice: str, num_rounds: int, worker_count: int):
        """Marker function for ZeroGPU dynamic GPU allocation."""
        from app.gradio_app import run_discovery_flow
        return run_discovery_flow(task_choice, num_rounds, worker_count)

# Ensure src/ is on sys.path for direct imports
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
