bash <<'PJLAB_GBS32_START' || printf '%s\n' 'GBS32_START_FAILED: inspect the output above.'
set -eo pipefail
WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
mkdir -p /root/workspace "$WORKDIR/logs"
cat > "$WORKDIR/run_interns2_adamw_gbs32.sh" <<'PJLAB_GBS32_RUNNER'
#!/usr/bin/env bash
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export ADAMW_JOB_ID="${ADAMW_JOB_ID:-interns2-redpajama-adamw-gbs32-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export CORPUS_DIR="$WORKDIR/datasets/interns2-redpajama-500"
if [[ ! "$ADAMW_JOB_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '%s\n' 'Invalid ADAMW_JOB_ID' >&2
    exit 2
fi
mkdir -p "$WORKDIR/runs" "$WORKDIR/logs"
source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

write_bundle() {
    export ADAMW_RUN_EXIT_CODE="$1"
    python - <<'PY'
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
run = Path(os.environ["CURRENT_RUN_DIR"])
(run / "launcher-exit.json").write_text(json.dumps({"exit_code": int(os.environ["ADAMW_RUN_EXIT_CODE"])}) + "\n", encoding="utf-8")
with ZipFile(run.with_suffix(".zip"), "w", ZIP_DEFLATED) as bundle:
    for path in sorted(run.rglob("*")):
        if path.is_file():
            bundle.write(path, arcname=str(path.relative_to(run.parent)))
print("ADAMW_RESULT_BUNDLE:", run.with_suffix(".zip"), flush=True)
PY
}

finish_job() {
    pjlab_status=$?
    trap - EXIT
    if [ -n "${CURRENT_RUN_DIR:-}" ]; then
        npu-smi info > "$CURRENT_RUN_DIR/npu-smi-end.txt" 2>&1 || true
        write_bundle "$pjlab_status" || true
    fi
    printf 'ADAMW_PIPELINE_EXIT_CODE=%s\n' "$pjlab_status"
    exit "$pjlab_status"
}
trap finish_job EXIT

python -m py_compile "$WORKDIR/train_interns2_adamw_dist.py" \
    "$WORKDIR/adamw_dist_update_probe.py" "$WORKDIR/adamw_shard_layout.py" "$WORKDIR/adamw_math.py" \
    "$WORKDIR/plot_interns2_adamw.py"

python - <<'PY'
import json
import os
import time
from pathlib import Path
root = Path(os.environ["CORPUS_DIR"])
assert (root / "tokens.bin").is_file() and (root / "tokens.meta.json").is_file(), "Prepare RedPajama first"
meta = json.loads((root / "tokens.meta.json").read_text())
assert meta["num_tokens"] >= 1024001, "Need 1,024,001 tokens for 500 distinct GBS32 / SEQ64 steps"
names = {"train_interns2_500.py", "train_interns2_smoke.py", "load_interns2_weights.py", "train_interns2_adamw.py", "train_interns2_adamw_dist.py"}
while True:
    busy = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            args = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(Path(arg).name in names for arg in args if arg):
            busy.append(path.name)
    if not busy:
        break
    print("WAITING_FOR_EXISTING_TRAINING_PIDS:", ",".join(busy), flush=True)
    time.sleep(30)
PY

run_one() {
    pjlab_role="$1"
    pjlab_steps="$2"
    export RUN_ID="$ADAMW_JOB_ID-$3"
    export CURRENT_RUN_DIR="$WORKDIR/runs/$RUN_ID"
    mkdir "$CURRENT_RUN_DIR"
    cp "$WORKDIR/run_interns2_adamw_gbs32.sh" "$CURRENT_RUN_DIR/"
    printf '%s\n' "$CURRENT_RUN_DIR" > "$WORKDIR/logs/latest-adamw-run.txt"
    npu-smi info > "$CURRENT_RUN_DIR/npu-smi-start.txt" 2>&1 || true
    date -u +%Y-%m-%dT%H:%M:%SZ > "$CURRENT_RUN_DIR/start-utc.txt"
    printf 'ADAMW_RUN_START role=%s steps=%s directory=%s\n' "$pjlab_role" "$pjlab_steps" "$CURRENT_RUN_DIR"
    python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=16 \
        "$WORKDIR/train_interns2_adamw_dist.py" \
        --run-dir "$CURRENT_RUN_DIR" --run-role "$pjlab_role" \
        --steps "$pjlab_steps" --schedule-steps 500 --warmup-steps 20 \
        --lr 0.00001 --min-lr 0.000001 \
        --seq-length 64 --num-microbatches 4 \
        --data-mode text --corpus-dir "$CORPUS_DIR" \
        2>&1 | tee "$CURRENT_RUN_DIR/train.log"
    npu-smi info > "$CURRENT_RUN_DIR/npu-smi-end.txt" 2>&1 || true
    write_bundle 0
    unset CURRENT_RUN_DIR
}

run_one validation 3 check3
export ADAMW_CHECK_DIR="$WORKDIR/runs/$ADAMW_JOB_ID-check3"
python - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["ADAMW_CHECK_DIR"])
summary = json.loads((run / "summary.json").read_text())
manifest = json.loads((run / "manifest.json").read_text())
assert manifest["global_batch_size"] == 32 and manifest["seq_length"] == 64
assert manifest["num_microbatches"] == 4 and manifest["micro_batch_size"] == 1
assert manifest["use_distributed_optimizer"] is True
assert summary["status"] == "passed" and summary["completed_steps"] == 3
assert summary["first_step_updates_verified_on_all_ranks"]
assert summary["min_peak_allocated_headroom_gib_all_ranks"] >= 2.0
assert json.loads((run / "launcher-exit.json").read_text())["exit_code"] == 0
print("ADAMW_SHORT_CHECK_PASSED:", json.dumps(summary), flush=True)
PY

# Start a new measurement from the original HF weights, not from pilot updates.
run_one measurement 500 steps500
printf '%s\n' 'ADAMW_500_PIPELINE_OK'
PJLAB_GBS32_RUNNER
sha256sum -c - <<'PJLAB_GBS32_SHA256'
4637b96d7464a36d9cd9506df5eb824003e18c0fb847a4b003b3c22953e391d3  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_adamw_gbs32.sh
PJLAB_GBS32_SHA256
printf '%s\n' 'GBS32_SCRIPTS_READY SEQ=64 GBS=32 MBS=1 DP=8 ACCUMULATION=4'
export ADAMW_JOB_ID="interns2-redpajama-adamw-gbs32-$(date -u +%Y%m%dT%H%M%SZ)-$$"
nohup bash "$WORKDIR/run_interns2_adamw_gbs32.sh" > "/root/workspace/$ADAMW_JOB_ID.log" 2>&1 < /dev/null &
printf 'STARTED_PID=%s\nRUN_PREFIX=%s\n' "$!" "$ADAMW_JOB_ID"
printf 'Watch: tail -F /root/workspace/%s.log\n' "$ADAMW_JOB_ID"
printf '%s\n' 'Run 3 validation steps first; start 500 steps only after validation passes.'
PJLAB_GBS32_START
