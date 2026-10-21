import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct imports
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

