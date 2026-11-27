#!/usr/bin/env bash
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export RP_DATA_DIR="$WORKDIR/datasets/redpajama-v2-en-head"
export CORPUS_DIR="$WORKDIR/datasets/interns2-redpajama-500"
mkdir -p "$RP_DATA_DIR" "$CORPUS_DIR" "$WORKDIR/logs"
source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh

python - <<'PY'
import gzip
import json
import os
import sys
import time
from pathlib import Path

# Wait for this workspace's existing full-device training jobs to exit.
names = {"train_interns2_500.py", "train_interns2_smoke.py", "load_interns2_weights.py"}
while True:
    busy = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, ProcessLookupError):
            continue
        if any(Path(arg).name in names for arg in args if arg):
            busy.append(proc.name)
    if not busy:
        break
    print("WAITING_FOR_EXISTING_TRAINING:", ",".join(busy), "retry in 30s", flush=True)
    time.sleep(30)

sys.path.insert(0, os.environ["WORKDIR"])
from fetch_wikipedia_zh import download_attempt, sha256_file

root = Path(os.environ["RP_DATA_DIR"])
repo = "togethercomputer/RedPajama-Data-V2"
revision = "949be9cdd1d82fbc379de4909dc4e89ca9b87680"
relative = "sample/documents/2023-06/0000/en_head.json.gz"
digest = "bc742de614bc508778a42a3d1d2f3cb71e558af88114a0c3fb9a67c6356e5ca7"
source = root / "en_head.json.gz"
if source.exists():
    assert source.stat().st_size == 56818623 and sha256_file(source) == digest, "Existing download checksum mismatch"
else:
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        print("DOWNLOAD_SOURCE:", host, flush=True)
        try:
            download_attempt(f"{host}/datasets/{repo}/resolve/{revision}/{relative}?download=true", source, 56818623, digest)
            break
        except Exception as error:
            print("DOWNLOAD_RETRY:", str(error), flush=True)
    else:
        raise RuntimeError("RedPajama download failed; no training started")
print("REDPAJAMA_SHA256_OK:", digest, flush=True)

destination = root / "train.jsonl"
temporary = root / f"train.jsonl.tmp-{os.getpid()}"
count = characters = 0
with gzip.open(source, "rt", encoding="utf-8") as input_file, temporary.open("w", encoding="utf-8", newline="\n") as output:
    for line in input_file:
        row = json.loads(line)
        text = row.get("raw_content")
        if not isinstance(text, str) or len(text.strip()) < 256:
            continue
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        characters += len(text)
        if count >= 5000 or characters >= 12000000:
            break
    output.flush()
    os.fsync(output.fileno())
assert count > 0, "No raw_content text found"
temporary.replace(destination)
metadata = {
    "dataset": repo, "revision": revision, "shard": relative,
    "shard_sha256": digest, "shard_sha256_verified": True,
    "dataset_card": f"https://huggingface.co/datasets/{repo}",
    "documents": count, "characters": characters,
    "jsonl_sha256": sha256_file(destination), "text_field": "raw_content",
    "selection": "first up to 5000 documents of at least 256 characters; stop at complete document after 12000000 characters; no additional quality-signal filtering or deduplication",
}
(root / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("REDPAJAMA_TEXT_READY:", json.dumps(metadata), flush=True)
PY

python "$WORKDIR/prepare_interns2_corpus.py" \
    --input "$RP_DATA_DIR/train.jsonl" --text-field raw_content \
    --model-path /mnt/afs-a3-weight-share/Intern-S2-Preview \
    --output-dir "$CORPUS_DIR" --max-tokens 2000000 \
    2>&1 | tee "$WORKDIR/logs/tokenize-redpajama.log"

python - <<'PY'
import json
import os
from pathlib import Path

source = json.loads((Path(os.environ["RP_DATA_DIR"]) / "source.json").read_text())
path = Path(os.environ["CORPUS_DIR"]) / "tokens.meta.json"
meta = json.loads(path.read_text())
assert meta["signature"]["input_sha256"] == source["jsonl_sha256"]
assert meta["num_tokens"] >= 256001, f"Insufficient tokens: {meta['num_tokens']}"
meta["upstream_dataset"] = source
temporary = path.with_name(f"tokens.meta.json.tmp-{os.getpid()}")
temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print("REDPAJAMA_CORPUS_READY tokens=", meta["num_tokens"], flush=True)


PY

export RUN_ID="interns2-redpajama-500-$(date -u +%Y%m%dT%H%M%SZ)-$$"
DATA_MODE=text \
TRAIN_TEXT_PATH="$RP_DATA_DIR/train.jsonl" \
TEXT_FIELD=raw_content \
TRAIN_STEPS=500 SEQ_LENGTH=64 NUM_MICROBATCHES=1 LEARNING_RATE=0.001 \
bash "$WORKDIR/start_interns2_500.sh"
