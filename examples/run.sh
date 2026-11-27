#!/usr/bin/env bash
set -eo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORKDIR="$(cd -- "$example_dir/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/mnt/afs-a3-weight-share/Intern-S2-Preview}"
export CORPUS_DIR="${CORPUS_DIR:-$WORKDIR/datasets/interns2-redpajama-raw}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export QWEN35_DEBUG="${QWEN35_DEBUG:-0}"

train_steps="${TRAIN_STEPS:-20}"
schedule_steps="${SCHEDULE_STEPS:-500}"
seq_length="${SEQ_LENGTH:-64}"
num_microbatches="${NUM_MICROBATCHES:-4}"
run_id="${RUN_ID:-interns2-example-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
raw_data="$WORKDIR/data/en_head.json.gz"
run_dir="$WORKDIR/runs/$run_id"
dry_run=0

case "${1:-}" in
    --dry-run) dry_run=1; shift ;;
    --help|-h)
        printf '%s\n' 'Usage: bash examples/run.sh [--dry-run]' \
            'Environment: MODEL_PATH, VENV_PATH, TRAIN_STEPS (20), SCHEDULE_STEPS (500),' \
            'SEQ_LENGTH (64), NUM_MICROBATCHES (4), CORPUS_DIR, RUN_ID.'
        exit 0 ;;
esac
if (( $# )); then
    printf 'Unknown argument: %s\n' "$1" >&2
    exit 2
fi
for example_number in "$train_steps" "$schedule_steps" "$seq_length" "$num_microbatches"; do
    if [[ ! "$example_number" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' 'Training sizes and steps must be positive integers.' >&2
        exit 2
    fi
done
if (( train_steps > schedule_steps || seq_length < 16 || seq_length % 2 )); then
    printf '%s\n' 'Require TRAIN_STEPS <= SCHEDULE_STEPS and an even SEQ_LENGTH >= 16.' >&2
    exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '%s\n' 'RUN_ID may contain only letters, digits, underscores and hyphens.' >&2
    exit 2
fi

prepare=(python "$WORKDIR/tools/environment/prepare_workspace.py")
tokenize=(python "$WORKDIR/src/DLTrain-framework/tools/prepare_interns2_corpus.py"
    --input "$raw_data" --text-field raw_content --model-path "$MODEL_PATH"
    --output-dir "$CORPUS_DIR" --max-tokens 2000000)
train=(python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16
    "$WORKDIR/examples/train_interns2.py" --run-dir "$run_dir" --run-role measurement
    --steps "$train_steps" --schedule-steps "$schedule_steps" --warmup-steps 20
    --lr "${LEARNING_RATE:-0.00001}" --min-lr "${MIN_LEARNING_RATE:-0.000001}"
    --seq-length "$seq_length" --num-microbatches "$num_microbatches"
    --data-mode text --corpus-dir "$CORPUS_DIR")

if (( dry_run )); then
    printf 'WORKDIR=%s\nMODEL_PATH=%s\nRAW_DATA=%s\n' "$WORKDIR" "$MODEL_PATH" "$raw_data"
    printf '%q ' "${prepare[@]}"; printf '\n'
    printf '%q ' "${tokenize[@]}"; printf '\n'
    printf '%q ' "${train[@]}"; printf '\n'
    exit 0
fi

cd "$WORKDIR"
venv_path="${VENV_PATH:-$WORKDIR/.venv-dltrain}"
if [[ -f "$venv_path/bin/activate" ]]; then
    source "$venv_path/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' 'Activate the prepared Python environment or set VENV_PATH.' >&2
    exit 1
fi
if [[ -d /usr/local/python3.12.13/lib ]]; then
    export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
fi
cann_env="${CANN_ENV_FILE:-/usr/local/Ascend/cann/set_env.sh}"
if [[ -f "$cann_env" ]]; then source "$cann_env"; fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
[[ -d "$MODEL_PATH" ]] || { printf 'Model directory missing: %s\n' "$MODEL_PATH" >&2; exit 1; }
printf '%s  %s\n' 'bc742de614bc508778a42a3d1d2f3cb71e558af88114a0c3fb9a67c6356e5ca7' "$raw_data" | sha256sum -c -
"${prepare[@]}"
"${tokenize[@]}"
mkdir -p "$WORKDIR/logs"
printf 'RUN_ID=%s\nLOG=%s/logs/%s.log\n' "$run_id" "$WORKDIR" "$run_id"
"${train[@]}" 2>&1 | tee "$WORKDIR/logs/$run_id.log"
