from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = ROOT / "tools/runtime"
if not RUNTIME_DIRECTORY.is_dir():
    RUNTIME_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIRECTORY))
