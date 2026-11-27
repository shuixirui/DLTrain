#!/usr/bin/env bash
set -eo pipefail

export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export RUN_ID="interns2-step-$(date -u +%Y%m%dT%H%M%SZ)"

source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

python -m py_compile "$WORKDIR/train_interns2_smoke.py" "$WORKDIR/optimizer_update_probe.py"
python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=16 \
    "$WORKDIR/train_interns2_smoke.py" \
    2>&1 | tee "$WORKDIR/logs/${RUN_ID}.log"
