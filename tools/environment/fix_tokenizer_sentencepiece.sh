#!/usr/bin/env bash
# Run in a child bash; this does not change interactive shell exit options.
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="$WORKDIR/.cache/pip"
source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh

printf '%s\n' 'sentencepiece==0.2.1' > "$WORKDIR/tokenizer-requirements.txt"
python -m pip download \
    --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    --find-links="$WORKDIR/wheelhouse" \
    --timeout 120 --retries 2 --only-binary=:all: --no-deps \
    --dest "$WORKDIR/wheelhouse" \
    -c "$WORKDIR/runtime-requirements.txt" \
    -c "$WORKDIR/adaptor-requirements.txt" \
    -c "$WORKDIR/model-requirements.txt" \
    -c "$WORKDIR/bridge-compat-requirements.txt" \
    -r "$WORKDIR/tokenizer-requirements.txt"
python -m pip install \
    --no-index --no-deps --find-links="$WORKDIR/wheelhouse" \
    -c "$WORKDIR/runtime-requirements.txt" \
    -c "$WORKDIR/adaptor-requirements.txt" \
    -c "$WORKDIR/model-requirements.txt" \
    -c "$WORKDIR/bridge-compat-requirements.txt" \
    -r "$WORKDIR/tokenizer-requirements.txt"
python -I -m pip check

python - <<'PY'
import json
from pathlib import Path
import sentencepiece
from transformers import AutoTokenizer

root = Path("/mnt/afs-a3-weight-share/Intern-S2-Preview")
tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
ids = tokenizer.encode("Intern-S2 training test.", add_special_tokens=False)
config = json.loads((root / "config.json").read_text())
vocab = config["text_config"]["vocab_size"]
assert ids and all(isinstance(token, int) and 0 <= token < vocab for token in ids)
assert isinstance(tokenizer.eos_token_id, int)
print("SENTENCEPIECE_VERSION:", sentencepiece.__version__, flush=True)
print("TOKENIZER_CLASS:", type(tokenizer).__name__, flush=True)
print("TOKENIZER_LOAD_AND_ENCODE_OK", flush=True)
PY

bash "$WORKDIR/prepare_wikipedia_zh.sh"
DATA_MODE=text \
TRAIN_TEXT_PATH="$WORKDIR/datasets/wikipedia-zh-20231101/train.jsonl" \
TEXT_FIELD=text \
CORPUS_DIR="$WORKDIR/datasets/interns2-500" \
TRAIN_STEPS=500 SEQ_LENGTH=64 NUM_MICROBATCHES=1 LEARNING_RATE=0.001 \
bash "$WORKDIR/start_interns2_500.sh"
