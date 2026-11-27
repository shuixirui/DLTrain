from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIRECTORY = ROOT / "train/runtime"
if not RUNTIME_DIRECTORY.is_dir():
    RUNTIME_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIRECTORY))
