# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""NPU device-probe compatibility helpers.

NOTE: this module must NOT import megatron at module level.  It is called
from pre_register_patches before megatron is allowed to load; any megatron
import here would (a) run MLM's import-time device probes (e.g. the GDP
kernels' ``megatron/core/ssm/ops/gdp/common.py``) and (b) capture the real
``torch.compile`` into ``megatron.core.jit.jit_fuser`` before the
``torch.compile -> dummy_compile`` stub is applied, breaking JIT warmup.
"""

from logging import getLogger

LOG = getLogger(__name__)


def get_device_capability(device=None):
    """NPU stub for ``torch.cuda.get_device_capability``.

    transfer_to_npu rewrites ``torch.cuda.is_available()`` -> True but leaves
    ``get_device_capability()`` returning None, which crashes MLM import-time
    probes that index the tuple (e.g. ``common.py``'s
    ``get_device_capability()[0] >= 9``).  Report Hopper-class 9.0 so those
    gates evaluate consistently with the regular patch path.
    """
    return 9, 0


def patch_device_capability():
    """Eagerly backfill ``torch.cuda.get_device_capability`` before megatron import.

    The regular registration (``register_non_mcore_basic_patches``) only
    applies after ``megatron.training`` has been imported, which is too late:
    MLM main #6538 probes device capability at import time.  Mirror the
    ``nvrx_compat`` backfill pattern and install the same stub the normal
    path would install.
    """
    import torch

    torch.cuda.get_device_capability = get_device_capability


_fallback_warned = False


def device_memory_used(device=None):
    """Return the torch/CANN-visible device memory in use, in bytes.

    ``torch.cuda.device_memory_used`` is NVML-backed on recent torch versions
    (``nvidia-ml-py does not seem to be installed`` on NPU), so Megatron-LM's
    ``--log-device-memory-used`` column has no working implementation here.

    torch_npu has no physical-HBM equivalent.  ``torch.npu.mem_get_info()``
    reports the runtime-visible (free, total) pool; its ``total`` excludes HBM
    reserved by the driver, so ``total - free`` is the exact usage of that
    pool, not ``npu-smi``'s physical HBM value.  It is a precisely defined
    torch_npu metric; its deltas track npu-smi while the reserved amount is
    stable. Each call reads the current value, so the logged number is not stale.
    """
    import torch

    global _fallback_warned
    try:
        free, total = torch.npu.mem_get_info(device)
        value = total - free
    except (AttributeError, OSError, RuntimeError, AssertionError) as exc:
        # Keep logging best-effort: a diagnostic column must not abort training.
        if not _fallback_warned:
            LOG.warning(
                'torch.npu.mem_get_info failed (%s); reporting nan for the device memory column.',
                exc,
            )
            _fallback_warned = True
        value = float('nan')
    return value


def patch_device_memory_used(args=None):
    """Backfill ``torch.cuda.device_memory_used``; warn once when the flag is on."""
    import torch

    if args is None or getattr(args, 'log_device_memory_used', False):
        LOG.warning(
            '--log-device-memory-used on NPU reports torch/CANN-visible device memory '
            '(mem_get_info total - free); it excludes driver-reserved HBM and is not '
            'the npu-smi HBM-Usage value.'
        )
    torch.cuda.device_memory_used = device_memory_used
