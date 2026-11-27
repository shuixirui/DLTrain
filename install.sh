#!/usr/bin/env bash

# Run in a child shell so installer failures do not close an interactive shell.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run this installer with: bash install.sh' >&2
    return 2
fi
set -eo pipefail

export WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VENV_PATH="${VENV_PATH:-$WORKDIR/.venv-dltrain}"
python_bin="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3.12}"
cann_env="${CANN_ENV_FILE:-/usr/local/Ascend/cann/set_env.sh}"
atb_env="${ATB_ENV_FILE:-/usr/local/Ascend/nnal/atb/set_env.sh}"
requirements="$WORKDIR/tools/environment/requirements"
dry_run=0
check_provider=0

while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --check-provider) check_provider=1 ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash install.sh [--dry-run] [--check-provider]' \
                'Installs the Ascend training environment; does not start training.' \
                'Environment: VENV_PATH (.venv-dltrain), PYTHON_BIN (Python 3.12),' \
                'CANN_ENV_FILE, ATB_ENV_FILE, PIP_INDEX_URL, PIP_EXTRA_INDEX_URL,' \
                'MODEL_PATH (required only for the optional Provider check).'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

if [[ "$VENV_PATH" != /* ]]; then
    export VENV_PATH="$WORKDIR/$VENV_PATH"
fi
export MODEL_PATH="${MODEL_PATH:-/mnt/afs-a3-weight-share/Intern-S2-Preview}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKDIR/.cache/pip}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
venv_python="$VENV_PATH/bin/python"

build_dependencies=("pip==25.0.1" "setuptools==80.9.0" "wheel==0.48.0"
    "pybind11==3.1.0" "packaging==26.4")
runtime_install=("$venv_python" -m pip install
    -r "$requirements/runtime-requirements.txt")
model_install=("$venv_python" -m pip install --no-build-isolation
    -c "$requirements/runtime-requirements.txt"
    -c "$requirements/adaptor-requirements.txt"
    -c "$requirements/bridge-compat-requirements.txt"
    -r "$requirements/adaptor-requirements.txt"
    -r "$requirements/model-requirements.txt"
    -r "$requirements/bridge-compat-requirements.txt"
    -r "$requirements/tokenizer-requirements.txt" "nvidia-modelopt==0.44.0")

print_command() { printf '%q ' "$@"; printf '\n'; }
print_plan() {
    printf 'WORKDIR=%s\nVENV_PATH=%s\n' "$WORKDIR" "$VENV_PATH"
    print_command "$python_bin" -I -m venv "$VENV_PATH"
    print_command "$venv_python" -m pip install "${build_dependencies[@]}"
    print_command "${runtime_install[@]}"
    print_command "${model_install[@]}"
    for component in DLTrain-framework DLTrain-NPUEngine DLTrain-Adaptor; do
        print_command "$venv_python" -m pip install \
            --use-pep517 --no-build-isolation --no-deps --no-index \
            -e "$WORKDIR/src/$component"
    done
    print_command "$venv_python" -I -m pip check
    print_command "$venv_python" "$WORKDIR/tools/environment/prepare_workspace.py"
    printf '%s\n' 'Verify NPU availability and imports from the repository source directories.'
    if (( check_provider )); then
        printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
        print_command "$venv_python" "$WORKDIR/tools/validation/check_interns2_provider.py"
    fi
}

if (( dry_run )); then
    print_plan
    exit 0
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
    printf '%s\n' 'This installation recipe requires Linux aarch64 with Ascend CANN.' >&2
    exit 1
}
[[ -f "$cann_env" ]] || { printf 'CANN environment file missing: %s\n' "$cann_env" >&2; exit 1; }
if (( check_provider )) && [[ ! -d "$MODEL_PATH" ]]; then
    printf 'Model directory missing: %s\n' "$MODEL_PATH" >&2
    exit 1
fi
if [[ -d /usr/local/python3.12.13/lib ]]; then
    export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV

mkdir -p "$WORKDIR/logs"
install_id="install-$(date -u +%Y%m%dT%H%M%SZ)-$$"
install_log="$WORKDIR/logs/$install_id.log"

install_environment() {
    cd "$WORKDIR"
    print_plan
    if [[ -d "$WORKDIR/.git" ]] && command -v /usr/bin/git >/dev/null; then
        /usr/bin/git -C "$WORKDIR" rev-parse HEAD
    elif [[ -f "$WORKDIR/source-commit.txt" ]]; then
        cat "$WORKDIR/source-commit.txt"
    else
        printf '%s\n' 'Source archive: commit not recorded.'
    fi
    "$python_bin" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    "$python_bin" -I -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    source "$cann_env"
    if [[ -f "$atb_env" ]]; then source "$atb_env"; fi

    "$venv_python" - <<'PY'
import os
import sys
from pathlib import Path

assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.prefix).resolve() == Path(os.environ['VENV_PATH']).resolve()
print('PYTHON:', sys.executable, sys.version, flush=True)
PY
    "$venv_python" -m pip install "${build_dependencies[@]}"
    "${runtime_install[@]}"
    "${model_install[@]}"
    for component in DLTrain-framework DLTrain-NPUEngine DLTrain-Adaptor; do
        "$venv_python" -m pip install \
            --use-pep517 --no-build-isolation --no-deps --no-index \
            -e "$WORKDIR/src/$component"
    done
    "$venv_python" -I -m pip check
    "$venv_python" "$WORKDIR/tools/environment/prepare_workspace.py"

    "$venv_python" - <<'PY'
import os
import sys
from importlib.metadata import version
from pathlib import Path

root = Path(os.environ['WORKDIR']).resolve()
for package in ('torch', 'torch-npu', 'numpy', 'pyarrow', 'transformers',
                'triton-ascend', 'megatron-core', 'transformer-engine',
                'megatron-adaptor', 'nvidia-modelopt', 'sentencepiece'):
    print('PACKAGE:', package, version(package), flush=True)

import numpy
import pyarrow
import datasets
import torch
import torch_npu

assert torch.npu.is_available(), 'No available NPU'
print('NPU_COUNT:', torch.npu.device_count(), flush=True)

# Populate the real HF optional-dependency probe before the adaptor creates FLA shims.
from transformers.utils.import_utils import is_flash_linear_attention_available
print('HF_FLA_BEFORE_ADAPTOR:', is_flash_linear_attention_available(), flush=True)
from transformers import Qwen3NextForCausalLM
import megatron_adaptor
import megatron.core
import transformer_engine.pytorch
import modelopt.torch.distill
import modelopt.torch.distill.plugins.megatron

sys.path.insert(0, str(root / 'src/DLTrain-Bridge/src'))
import megatron.bridge
from megatron.bridge import AutoBridge

for module, directory in (
    (megatron.core, 'DLTrain-framework'),
    (megatron_adaptor, 'DLTrain-Adaptor'),
    (transformer_engine.pytorch, 'DLTrain-NPUEngine'),
    (megatron.bridge, 'DLTrain-Bridge'),
):
    actual = Path(module.__file__).resolve()
    assert actual.is_relative_to(root / 'src' / directory), (module.__name__, actual)
    print('SOURCE_IMPORT:', module.__name__, actual, flush=True)
print('DLTRAIN_IMPORT_CHECK_OK', flush=True)
PY

    if (( check_provider )); then
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export QWEN35_DEBUG="${QWEN35_DEBUG:-0}"
        "$venv_python" "$WORKDIR/tools/validation/check_interns2_provider.py"
    fi
    "$venv_python" -m pip freeze > "$WORKDIR/logs/$install_id-packages.txt"
    printf '\nDLTRAIN_CLEAN_ENV_READY\n'
    printf 'VENV_PATH=%s\nINSTALL_LOG=%s\n' "$VENV_PATH" "$install_log"
}

install_environment 2>&1 | tee "$install_log"
