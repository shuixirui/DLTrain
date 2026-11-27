#!/usr/bin/env bash
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export WIKI_DATA_DIR="$WORKDIR/datasets/wikipedia-zh-20231101"
export WIKI_TOKEN_DIR="$WORKDIR/datasets/interns2-500"
mkdir -p "$WORKDIR/logs" "$WIKI_DATA_DIR"
source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh

python "$WORKDIR/fetch_wikipedia_zh.py" --output-dir "$WIKI_DATA_DIR" \
    2>&1 | tee "$WORKDIR/logs/prepare-wikipedia-zh.log"
python "$WORKDIR/prepare_interns2_corpus.py" \
    --input "$WIKI_DATA_DIR/train.jsonl" --text-field text \
    --output-dir "$WIKI_TOKEN_DIR" \
    --model-path /mnt/afs-a3-weight-share/Intern-S2-Preview \
    --max-tokens 2000000 \
    2>&1 | tee "$WORKDIR/logs/tokenize-wikipedia-zh.log"

python - <<'PY'
import json
import os
from pathlib import Path

source = json.loads((Path(os.environ["WIKI_DATA_DIR"]) / "source.json").read_text(encoding="utf-8"))
path = Path(os.environ["WIKI_TOKEN_DIR"]) / "tokens.meta.json"
meta = json.loads(path.read_text(encoding="utf-8"))
assert meta["signature"]["input_sha256"] == source["jsonl_sha256"], "Source and tokenized corpus differ"
assert meta["num_tokens"] >= 256001, f"Only {meta['num_tokens']} tokens; at least 256001 required for this 500-step configuration"
meta["upstream_dataset"] = source
temporary = path.with_name(f"tokens.meta.json.tmp-{os.getpid()}")
temporary.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
temporary.replace(path)
print("WIKIPEDIA_ZH_CORPUS_READY", "tokens=", meta["num_tokens"], "path=", source["jsonl_path"], flush=True)
PY
