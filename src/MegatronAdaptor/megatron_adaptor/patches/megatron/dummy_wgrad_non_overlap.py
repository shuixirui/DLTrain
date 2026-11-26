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
"""Skip the persistent dummy wgrad when grad reduction does not overlap backward.

With ``gradient_accumulation_fusion`` the fused kernel accumulates wgrad directly
into ``weight.main_grad``, so the returned ``grad_weight`` is a throw-away
placeholder. Upstream still materializes it because the DDP post-accumulate hook
needs a non-None ``param.grad`` while reduction overlaps backward
(``overlap_grad_reduce=True``); without overlap the placeholder only costs memory
(TE's shape-keyed ``_dummy_wgrads`` cache is never freed), so returning ``None``
is numerically equivalent.

Two decorators, both on Megatron-side bindings (no TE internals touched):
  * ``DistributedDataParallel.__init__``: tags params with
    ``requires_dummy_wgrad = ddp_config.overlap_grad_reduce`` (the marker the
    TransformerEngineNPU gate consumes) and arms the process-wide skip gate.
  * ``megatron.core.tensor_parallel.layers.get_dummy_wgrad``: the TE helper as
    bound by MCore, whose only caller is the fused backward; serves ``None``
    while the gate is armed. Upstream sets ``grad_added_to_main_grad = True``
    itself after the branch, so the DDP / Megatron-FSDP contract holds.
"""

from functools import wraps

# Armed when the DDP config says grad reduction does not overlap backward. Kept
# module-level because the gate value is uniform for the whole run.
_SKIP_DUMMY_WGRAD = False

_LOGGED = False


def _log_gate_status(overlap_grad_reduce, num_params):
    """Report the gate status once, so an ineffective switch stays visible."""
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    state = 'dummy wgrad still allocated (overlap path)' if overlap_grad_reduce else 'no dummy wgrad tensor allocated'
    message = '[fix-non-overlap-dummy-wgrad] gated {} params (overlap_grad_reduce={}): {}'.format(
        num_params, overlap_grad_reduce, state
    )
    try:
        from megatron.training.utils import print_rank_0

        print_rank_0(message)
    except ImportError:
        # mcore-only environment: per-rank lines are acceptable there.
        print(message, flush=True)


def ddp_init_dummy_wgrad_wrapper(fn):
    """Tag DDP-managed trainable params and arm the dummy wgrad gate.

    ``requires_dummy_wgrad`` mirrors the upstream gate (the DDP grad-reduction
    overlap flag) and is uniform within one DDP instance. Params that never get
    tagged keep the ``True`` default, so unwrapped/unpatched paths (e.g. FSDP2)
    behave exactly as before.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)

        params_with_grad = getattr(self, 'params_with_grad', None)
        ddp_config = getattr(self, 'ddp_config', None)
        if params_with_grad is None or ddp_config is None:
            raise RuntimeError(
                'DistributedDataParallel.__init__ no longer exposes params_with_grad/ddp_config; '
                'the --fix-non-overlap-dummy-wgrad patch needs to be updated for this revision.'
            )

        overlap_grad_reduce = ddp_config.overlap_grad_reduce
        global _SKIP_DUMMY_WGRAD
        _SKIP_DUMMY_WGRAD = not overlap_grad_reduce
        for param in params_with_grad:
            param.requires_dummy_wgrad = overlap_grad_reduce
        _log_gate_status(overlap_grad_reduce, len(params_with_grad))
        return result

    return wrapper


def get_dummy_wgrad_skip_gate_wrapper(fn):
    """Serve no dummy wgrad while the non-overlap gate is armed."""

    @wraps(fn)
    def wrapper(shape, dtype, zero=False, device=None):
        if _SKIP_DUMMY_WGRAD:
            return None
        return fn(shape, dtype, zero, device)

    return wrapper
