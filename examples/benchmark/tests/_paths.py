from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ("tools/runtime", "tools/preprocess", "examples/benchmark"):
    sys.path.insert(0, str(ROOT / relative))
