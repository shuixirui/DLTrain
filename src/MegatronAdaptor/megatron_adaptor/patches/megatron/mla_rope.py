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
"""Ascend adaptations for MLA (multi-latent attention) RoPE fusion.

Megatron 0.18 accepts ``apply_rope_fusion`` for MLA + YARN, implemented with Triton
kernels that are efficient on GPU but roughly two orders of magnitude slower than the
CANN rotary ops on Ascend. Two things are needed, both kept minimal here:

* P0: disable that fusion when the TransformerEngineNPU backend is unavailable
  (``mla_rope_fusion_post_init_wrapper``);
* P1: ``npu_mla_rope_backend`` returns replacements for Megatron's
  ``fused_apply_mla_rope_for_q/kv`` that call the NPU backend first and fall back to
  Megatron's implementation when the backend declines the input. Every capability check
  lives in the backend (``MLARopeUnsupportedError``). The backend output is
  element-wise identical to Megatron's kernels, so the per-call fallback is
  layout-transparent; the two replacements are registered as a pair for symmetry.
"""

import warnings
from functools import wraps

_WARNED = False


def reset_warning_state():
    """Reset the one-shot warning flag (test helper)."""
    global _WARNED
    _WARNED = False


def mla_rope_npu_backend_wrapper(orig, backend, unsupported_error):
    """Wrap ``orig`` so it prefers the NPU backend and falls back when it declines.

    Register the *returned* function (it keeps ``orig``'s name, so ``register_patch``
    installs it as a replacement); the factory itself is not a ``Patch`` wrapper.
    ``unsupported_error`` is passed in by :func:`npu_mla_rope_backend` so this factory
    never imports from ``transformer_engine`` itself.
    """

    @wraps(orig)
    def replacement(*args, **kwargs):
        try:
            return backend(*args, **kwargs)
        except unsupported_error:
            return orig(*args, **kwargs)

    return replacement


def npu_mla_rope_backend():
    """Replacements for Megatron's fused MLA rope, or ``(None, None)`` when unusable.

    Every symbol the backend depends on -- the Megatron fusion module, the two kernels
    and ``MLARopeUnsupportedError`` -- is imported inside one guard, so an incomplete
    backend (e.g. a version mismatch) degrades to ``(None, None)`` instead of raising
    at feature registration or config construction.

    Each returned callable already contains the fallback, so callers only need to
    register it against the Megatron symbol.
    """
    try:
        import megatron.core.fusions.fused_mla_yarn_rope_apply as fused_mla_rope
        from transformer_engine.pytorch.attention.rope import (
            MLARopeUnsupportedError,
            fused_apply_mla_rope_for_kv,
            fused_apply_mla_rope_for_q,
        )
    except ImportError:
        return None, None

    return (
        mla_rope_npu_backend_wrapper(
            fused_mla_rope.fused_apply_mla_rope_for_q, fused_apply_mla_rope_for_q, MLARopeUnsupportedError
        ),
        mla_rope_npu_backend_wrapper(
            fused_mla_rope.fused_apply_mla_rope_for_kv, fused_apply_mla_rope_for_kv, MLARopeUnsupportedError
        ),
    )


def npu_mla_rope_available():
    """Whether TransformerEngineNPU provides the MLA rope backend."""
    return npu_mla_rope_backend()[0] is not None


def _warn_disabled_once():
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(
        "MLA + YARN rope fusion is disabled on Ascend: Megatron's Triton kernel is much "
        "slower than the CANN rotary op here. Pass --mla-rope-fusion to force-enable it.",
        UserWarning,
        stacklevel=3,
    )


def mla_rope_fusion_post_init_wrapper(fn):
    """Disable MLA + YARN rope fusion when no NPU backend can serve it."""

    @wraps(fn)
    def wrapper(self):
        disable = (
            getattr(self, "multi_latent_attention", False)
            and getattr(self, "rope_type", None) == "yarn"
            and getattr(self, "apply_rope_fusion", False)
            and not getattr(self, "mla_rope_fusion", False)
            and not npu_mla_rope_available()
        )
        if disable:
            # Set before calling upstream so Megatron's own MLA+fusion warning is skipped.
            self.apply_rope_fusion = False
        fn(self)
        if disable:
            _warn_disabled_once()

    return wrapper
