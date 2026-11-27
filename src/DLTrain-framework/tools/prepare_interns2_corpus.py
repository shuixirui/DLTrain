"""Tokenize a local UTF-8 text/JSONL corpus for an explicitly recorded training run."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def documents(path, field):
    compressed = path.suffix.lower() == ".gz"
    suffix = path.with_suffix("").suffix.lower() if compressed else path.suffix.lower()
    source_file = gzip.open(path, "rt", encoding="utf-8-sig") if compressed else path.open(encoding="utf-8-sig")
    with source_file as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if suffix in (".json", ".jsonl"):
                record = json.loads(line)
                text = record.get(field) if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Line {line_number}: expected a string field {field!r}")
            else:
                text = line
            if text.strip():
                yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", dest="model", type=Path, default=Path("/mnt/afs-a3-weight-share/Intern-S2-Preview"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    args = parser.parse_args()
    suffix = args.input.with_suffix("").suffix.lower() if args.input.suffix.lower() == ".gz" else args.input.suffix.lower()
    if suffix not in (".txt", ".json", ".jsonl"):
        parser.error("--input must contain UTF-8 text or line-delimited JSON, optionally gzip-compressed")
    if args.max_tokens < 1024:
        parser.error("--max-tokens must be at least 1024")

    import numpy as np
    import transformers
    from transformers import AutoTokenizer

    # Hash the source once. No corpus or model weights are modified.
    source_hash = sha256_file(args.input)
    tokenizer_files = {}
    for pattern in ("config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenization*.py"):
        for path in args.model.glob(pattern):
            if path.is_file():
                tokenizer_files[path.name] = sha256_file(path)
    signature = {
        "input_sha256": source_hash, "text_field": args.text_field,
        "max_tokens": args.max_tokens, "tokenizer_files_sha256": tokenizer_files,
        "transformers_version": transformers.__version__,
        "document_policy": "each non-empty txt line / JSONL record is a document; append EOS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary, manifest = args.output_dir / "tokens.bin", args.output_dir / "tokens.meta.json"
    if binary.is_file() and manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature") == signature and sha256_file(binary) == old["tokens_sha256"]:
            print("CORPUS_CACHE_VERIFIED:", manifest, flush=True)
            return

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError("Tokenizer must expose a single EOS token ID")
    excluded_ids = set()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for key, value in config.items():
        if key.endswith("token_id") and any(tag in key for tag in ("image", "video", "vision", "ts_", "time_series")):
            if isinstance(value, int):
                excluded_ids.add(value)
    token_count, doc_count, skipped = 0, 0, 0
    temporary = binary.with_name(f"tokens.bin.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            for text in documents(args.input, args.text_field):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if excluded_ids.intersection(ids):
                    raise ValueError("Corpus contains multimodal placeholder tokens; this run supports text only")
                if not ids:
                    skipped += 1
                    continue
                ids.append(eos)
                ids = ids[:args.max_tokens - token_count]
                np.asarray(ids, dtype="<u4").tofile(output)
                token_count += len(ids)
                doc_count += 1
                if token_count >= args.max_tokens:
                    break
            output.flush()
            os.fsync(output.fileno())
        if token_count < 1024:
            raise ValueError(f"Only {token_count} tokens found; use at least 1024 tokens of real text")
        temporary.replace(binary)
    finally:
        if temporary.exists():
            temporary.unlink()
    meta = {
        "schema_version": 1, "signature": signature,
        "input_path": str(args.input.resolve()), "model_path": str(args.model.resolve()),
        "dtype": "<u4", "num_tokens": token_count, "num_documents": doc_count,
        "empty_token_documents_skipped": skipped, "eos_token_id": eos,
        "tokens_sha256": sha256_file(binary),
        "scope": "training text only; no held-out evaluation; documents joined with EOS",
    }
    meta_tmp = manifest.with_name(f"tokens.meta.json.tmp-{os.getpid()}")
    meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta_tmp.replace(manifest)
    print("CORPUS_READY:", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
