# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVRx compatibility helpers.

NOTE: this module must NOT import megatron at module level. It is called from
pre_register_patches before megatron is allowed to load; any megatron import
here would trigger strategies/torch.py's import-time HAVE_NVRX probe, which is
exactly what the backfill below is meant to precede.
"""


def patch_nvrx_version_attr():
    """Backfill ``nvidia_resiliency_ext.__version__`` before megatron import.

    The adapted NVRx package on NPU does not define ``__version__`` while
    ``strategies/torch.py`` evaluates ``HAVE_NVRX = has_nvrx_async_support()``
    at import time, which reads the attribute via ``is_nvrx_min_version``.
    Setting a large version here keeps that import-time probe from raising
    ``AttributeError`` and makes the minimum-version comparison pass; the
    adapted package's version string does not reflect its actual feature set,
    so the ``async_ckpt`` symbol probes inside ``has_nvrx_async_support``
    remain the real gate for async-checkpointing support.
    """
    try:
        import nvidia_resiliency_ext as nvrx
    except (ImportError, ModuleNotFoundError):
        return

    nvrx.__version__ = '999.999.999'
