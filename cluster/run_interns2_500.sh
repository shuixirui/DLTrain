#!/usr/bin/env bash
set -eo pipefail

export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export RUN_ID="${RUN_ID:-interns2-500-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
    printf '%s\n' 'Invalid RUN_ID' >&2
    exit 2
fi
export RUN_DIR="$WORKDIR/runs/$RUN_ID"
mkdir -p "$WORKDIR/runs" "$WORKDIR/logs"
mkdir "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$WORKDIR/logs/latest-500-run.txt"

source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

finish_run() {
    pjlab_exit_code=$?
    trap - EXIT
    export PJLAB_TRAIN_EXIT_CODE="$pjlab_exit_code"
    python - <<'PY'
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

run = Path(os.environ["RUN_DIR"])
(run / "launcher-exit.json").write_text(json.dumps({"exit_code": int(os.environ["PJLAB_TRAIN_EXIT_CODE"])}) + "\n", encoding="utf-8")
archive = run.with_suffix(".zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
    for path in sorted(run.rglob("*")):
        if path.is_file():
            bundle.write(path, arcname=str(path.relative_to(run.parent)))
print("MEASUREMENT_BUNDLE:", archive, flush=True)
PY
    printf 'TRAIN_EXIT_CODE=%s\n' "$pjlab_exit_code"
    exit "$pjlab_exit_code"
}
trap finish_run EXIT

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/start-utc.txt"
uname -a > "$RUN_DIR/uname.txt"
npu-smi info > "$RUN_DIR/npu-smi-start.txt" 2>&1 || true

python -m py_compile \
    "$WORKDIR/train_interns2_500.py" \
    "$WORKDIR/optimizer_update_probe.py" \
    "$WORKDIR/prepare_interns2_corpus.py" \
    "$WORKDIR/plot_interns2_500.py"

pjlab_data_mode="${DATA_MODE:-text}"
pjlab_corpus_dir="${CORPUS_DIR:-$WORKDIR/datasets/interns2-500}"
if [ "$pjlab_data_mode" = text ]; then
    if [ -z "${TRAIN_TEXT_PATH:-}" ]; then
        printf '%s\n' 'Set TRAIN_TEXT_PATH to a real UTF-8 .txt or .jsonl file. No random text is used implicitly.' >&2
        exit 2
    fi
    python "$WORKDIR/prepare_interns2_corpus.py" \
        --input "$TRAIN_TEXT_PATH" \
        --model-path /mnt/afs-a3-weight-share/Intern-S2-Preview \
        --output-dir "$pjlab_corpus_dir" \
        --text-field "${TEXT_FIELD:-text}" \
        --max-tokens "${MAX_CORPUS_TOKENS:-2000000}" \
        2>&1 | tee "$RUN_DIR/prepare-corpus.log"
elif [ "$pjlab_data_mode" != synthetic ]; then
    printf '%s\n' 'DATA_MODE must be text or synthetic' >&2
    exit 2
fi

python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=16 \
    "$WORKDIR/train_interns2_500.py" \
    --run-dir "$RUN_DIR" \
    --steps "${TRAIN_STEPS:-500}" \
    --seq-length "${SEQ_LENGTH:-64}" \
    --num-microbatches "${NUM_MICROBATCHES:-1}" \
    --lr "${LEARNING_RATE:-0.001}" \
    --data-mode "$pjlab_data_mode" \
    --corpus-dir "$pjlab_corpus_dir" \
    2>&1 | tee "$RUN_DIR/train.log"
