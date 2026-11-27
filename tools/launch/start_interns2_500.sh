#!/usr/bin/env bash
# Run this script with bash. It launches a detached job and returns to the terminal.
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export DATA_MODE="${DATA_MODE:-text}"
export RUN_ID="${RUN_ID:-interns2-500-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
    printf '%s\n' 'Invalid RUN_ID' >&2
    exit 2
fi
if [ "$DATA_MODE" = text ] && [ ! -f "${TRAIN_TEXT_PATH:-}" ]; then
    printf '%s\n' 'Set TRAIN_TEXT_PATH to the complete path of a readable .txt or .jsonl corpus.' >&2
    exit 2
fi
if [ -d "$WORKDIR/runs/$RUN_ID" ]; then
    printf '%s\n' 'RUN_ID already exists; choose a fresh run ID.' >&2
    exit 2
fi
mkdir -p "$WORKDIR/logs"
nohup bash "$WORKDIR/run_interns2_500.sh" \
    > "$WORKDIR/logs/$RUN_ID-launcher.log" 2>&1 < /dev/null &
pjlab_pid=$!
printf '%s\n' "$pjlab_pid" > "$WORKDIR/logs/$RUN_ID.pid"
printf 'STARTED_PID=%s\nRUN_ID=%s\n' "$pjlab_pid" "$RUN_ID"
printf 'Watch: tail -F %s/logs/%s-launcher.log\n' "$WORKDIR" "$RUN_ID"
printf 'Result directory: %s/runs/%s\n' "$WORKDIR" "$RUN_ID"
printf '%s\n' 'Started in background. Check INTERNS2_MEASURED_TRAINING_OK and TRAIN_EXIT_CODE=0 at completion.'
