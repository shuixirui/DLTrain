from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("train/runtime", "data", "benchmark", "tools/legacy"):
    sys.path.insert(0, str(ROOT / relative))
