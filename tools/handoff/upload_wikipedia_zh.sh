bash <<'PJLAB_WIKIPEDIA_UPLOAD' || printf '%s\n' 'WIKIPEDIA_UPLOAD_FAILED'
set -eu
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/fetch_wikipedia_zh.py' <<'PJLAB_WIKIPEDIA_FILE_0'
"""Fetch one hash-pinned Wikimedia Chinese Wikipedia shard and retain article attribution."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "wikimedia/wikipedia"
REVISION = "5428a5d9ebb8d68732fc1227ebc5e8d5cec28592"
SHARD = "20231101.zh/train-00002-of-00006.parquet"
SIZE = 126796358
SHA256 = "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"
SOURCE_CARD = "https://huggingface.co/datasets/wikimedia/wikipedia"
POINTER_SOURCE = f"{SOURCE_CARD}/commit/{REVISION}"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_attempt(url, target, expected_size=SIZE, expected_sha256=SHA256):
    partial = target.with_name(target.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        raise ValueError("Partial file is larger than expected; inspect it before retrying")
    if start != expected_size:
        headers = {"User-Agent": "InternS2-measured-training/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            status = response.status
            if status == 206:
                expected_range = f"bytes {start}-"
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_range) or not content_range.endswith(f"/{expected_size}"):
                    raise ValueError("Unexpected Content-Range; refusing to append another resource")
                mode = "ab"
            elif status == 200:
                mode, start = "wb", 0  # server did not support resume
            else:
                raise ValueError(f"Unexpected download HTTP status {status}")
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Server returned an HTML page instead of a dataset shard")
            received, last_report = start, time.monotonic()
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if received > expected_size:
                        raise ValueError("Download exceeds the pinned file size")
                    if time.monotonic() - last_report >= 5:
                        print(f"DOWNLOAD_PROGRESS: {received}/{expected_size} bytes", flush=True)
                        last_report = time.monotonic()
                output.flush()
                os.fsync(output.fileno())
    if partial.stat().st_size != expected_size:
        raise ValueError("Incomplete dataset shard; the partial file is retained for resume")
    if sha256_file(partial) != expected_sha256:
        rejected = partial.with_name(partial.name + f".rejected-{time.time_ns()}")
        partial.replace(rejected)
        raise ValueError("Downloaded file SHA256 differs from Wikimedia's published pointer; retained as .rejected")
    partial.replace(target)


def fetch_shard(output_dir):
    target = output_dir / Path(SHARD).name
    if target.exists():
        if target.stat().st_size != SIZE or sha256_file(target) != SHA256:
            raise ValueError("Existing shard checksum mismatch; refusing to silently overwrite it")
        print("WIKIPEDIA_SHARD_CACHE_VERIFIED", flush=True)
        return target, "verified-local-cache"
    # A mirror can only supply the exact bytes from the publisher's pinned SHA256.
    endpoints = ("https://huggingface.co", "https://hf-mirror.com")
    for endpoint in endpoints:
        url = f"{endpoint}/datasets/{REPOSITORY}/resolve/{REVISION}/{SHARD}?download=true"
        for attempt in range(1, 3):
            print(f"DOWNLOAD_SOURCE: {endpoint}, attempt {attempt}/2, pinned revision {REVISION}", flush=True)
            try:
                download_attempt(url, target)
                return target, endpoint
            except Exception as error:
                print(f"DOWNLOAD_RETRY: {type(error).__name__}: {error}", flush=True)
    raise RuntimeError("Official and mirror downloads failed; keep the log and retry. No training was started.")


def select_articles(records, max_documents=5000, max_characters=12_000_000, min_characters=256):
    count, characters = 0, 0
    for row in records:
        text = row.get("text")
        if not isinstance(text, str) or len(text.strip()) < min_characters:
            continue
        result = {key: row[key] for key in ("id", "url", "title", "text")}
        if not isinstance(result["url"], str) or not result["url"].startswith("https://zh.wikipedia.org/"):
            raise ValueError("Unexpected article provenance URL")
        yield result
        count += 1
        characters += len(text)
        if count >= max_documents or characters >= max_characters:
            break


def extract_corpus(parquet_path, output_dir, endpoint):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)

    def records():
        for batch in parquet.iter_batches(batch_size=128, columns=["id", "url", "title", "text"]):
            yield from batch.to_pylist()

    destination = output_dir / "train.jsonl"
    temporary = destination.with_name(f"train.jsonl.tmp-{os.getpid()}")
    count, characters = 0, 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for row in select_articles(records()):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                characters += len(row["text"])
            output.flush()
            os.fsync(output.fileno())
        if not count:
            raise ValueError("No usable Chinese Wikipedia text articles found")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        "schema_version": 1, "dataset": REPOSITORY, "config": "20231101.zh", "split": "train",
        "revision": REVISION, "shard": SHARD, "shard_size_bytes": SIZE,
        "shard_sha256": SHA256, "shard_sha256_verified": True,
        "dataset_card": SOURCE_CARD, "published_pointer_source": POINTER_SOURCE,
        "download_endpoint": endpoint, "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_labels_from_dataset_card": ["cc-by-sa-3.0", "gfdl"],
        "attribution": "Wikipedia contributors; article id, title and source URL retained in each JSONL record",
        "selection": "first up to 5000 articles with at least 256 characters in this pinned shard; stop after a complete article once 12000000 characters reached; no shuffle",
        "documents": count, "text_characters": characters,
        "jsonl_path": str(destination.resolve()), "jsonl_sha256": sha256_file(destination),
        "text_field": "text", "held_out_evaluation": False,
        "scope": "fixed training subset for a 500-step systems test; not a representative quality benchmark",
    }
    (output_dir / "source.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WIKIPEDIA_TEXT_READY:", json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard, endpoint = fetch_shard(args.output_dir)
    extract_corpus(shard, args.output_dir, endpoint)
PJLAB_WIKIPEDIA_FILE_0
printf '%s\n' 'd89703287e835fa43f02837d3c78c11485b07a4d41de3781418c1f57f5c3cf9c  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/fetch_wikipedia_zh.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/prepare_wikipedia_zh.sh' <<'PJLAB_WIKIPEDIA_FILE_1'
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
PJLAB_WIKIPEDIA_FILE_1
printf '%s\n' 'dc453a1da93418288b7246c22713b3ad8469f2d8fcdf621fd77ce421e6bd88d1  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/prepare_wikipedia_zh.sh' | sha256sum -c -
printf '%s\n' 'PJLAB_WIKIPEDIA_SCRIPTS_READY'
PJLAB_WIKIPEDIA_UPLOAD
