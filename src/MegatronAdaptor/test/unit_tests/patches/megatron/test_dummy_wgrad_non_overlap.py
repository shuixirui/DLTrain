# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression tests for the ``--fix-non-overlap-dummy-wgrad`` compatibility switch.

Covers the path delta around the switch: off (default) registers no patch and
leaves upstream paths untouched; on, DDP init tags params with
``requires_dummy_wgrad`` and arms the dummy wgrad gate, and the Megatron-side
binding of the TE helper stops serving placeholders while the gate is armed
(whereas the upstream placeholder is preserved for the overlap path). Device and
distributed collaborators are replaced with tiny fakes via monkeypatch; no NPU
device is required.

The TransformerEngine-side gate on ``setup_dummy_wgrad`` is not covered here: it
belongs to TransformerEngineNPU, not to MA.

Run:
    bash test/unit_tests/run_unit_tests.sh <Megatron-LM path>
"""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

import megatron_adaptor  # noqa: F401  ensure the MA patch entry is loaded
from megatron_adaptor.features_manager.megatron_fix.megatron_fix import MegatronFixFeature
from megatron_adaptor.patches.megatron import dummy_wgrad_non_overlap as megatron_fix

_DUMMY_WGRAD_TARGETS = [
    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
    'megatron.core.tensor_parallel.layers.get_dummy_wgrad',
]


class _PatchManagerRecorder:
    """Records register_patch targets instead of applying them."""

    def __init__(self):
        self.registered = []

    def register_patch(self, target, func):
        self.registered.append(target)


class _DummyWgradFactory:
    """``get_dummy_wgrad`` double recording the requested shape/dtype/zero flag."""

    def __init__(self):
        self.calls = []
        self.sentinel = object()

    def __call__(self, shape, dtype, zero=False, device=None):
        self.calls.append(SimpleNamespace(shape=list(shape), dtype=dtype, zero=zero, device=device))
        return self.sentinel


class TestDummyWgradRegistration:
    """Switch parsing and patch-target registration for the dummy wgrad fix
    (the sibling file owns the ddp-one-shot-stream registration cases).
    """

    @staticmethod
    def _args(*flags):
        """Build args from the feature's own parser, so new ``--fix-*`` switches
        default off here instead of forcing every case to list all attributes.
        """
        parser = ArgumentParser()
        MegatronFixFeature().register_args(parser)
        return parser.parse_args(list(flags))

    def test_patches_registered_only_when_switch_on(self, monkeypatch):
        # The only probe-able symbol is the TE-provided helper; make it observable
        # regardless of whether TransformerEngine is installed in the test environment.
        monkeypatch.setattr(
            'megatron.core.tensor_parallel.layers.get_dummy_wgrad',
            lambda *unused, **kwargs: None,
            raising=False,
        )
        feature = MegatronFixFeature()
        patch_manager = _PatchManagerRecorder()

        # Switch off (default): no patch is registered, upstream paths stay untouched.
        feature.register_patches(patch_manager, self._args())
        assert not patch_manager.registered

        # Switch on: patch the DDP tagging point and the MCore-side dummy source.
        feature.register_patches(patch_manager, self._args('--fix-non-overlap-dummy-wgrad'))
        assert patch_manager.registered == _DUMMY_WGRAD_TARGETS

    def test_missing_dummy_wgrad_source_fails_loudly(self, monkeypatch):
        # MCore binds get_dummy_wgrad from TransformerEngine; when it is absent the
        # switch could only fail silently, so registration must refuse instead.
        monkeypatch.delattr('megatron.core.tensor_parallel.layers.get_dummy_wgrad', raising=False)
        with pytest.raises(RuntimeError, match='fix-non-overlap-dummy-wgrad'):
            MegatronFixFeature().register_patches(
                _PatchManagerRecorder(),
                self._args('--fix-non-overlap-dummy-wgrad'),
            )


class TestDdpInitDummyWgradWrapper:
    """DDP init tagging: the gate value is the DDP grad-reduction overlap flag."""

    @staticmethod
    def _make_init(overlap_grad_reduce, num_params=3):
        def init(self):
            self.params_with_grad = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(num_params)]
            self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap_grad_reduce)

        return init

    @pytest.mark.parametrize('overlap_grad_reduce', [True, False])
    def test_tags_params_and_arms_gate(self, monkeypatch, overlap_grad_reduce):
        logged = []
        monkeypatch.setattr(megatron_fix, '_SKIP_DUMMY_WGRAD', False)
        monkeypatch.setattr(
            megatron_fix,
            '_log_gate_status',
            lambda requires_dummy_wgrad, num_params: logged.append((requires_dummy_wgrad, num_params)),
        )
        wrapped = megatron_fix.ddp_init_dummy_wgrad_wrapper(self._make_init(overlap_grad_reduce, num_params=3))
        ddp = SimpleNamespace()

        wrapped(ddp)

        assert [param.requires_dummy_wgrad for param in ddp.params_with_grad] == [overlap_grad_reduce] * 3
        # The gate is armed exactly when the overlap flag is off.
        assert megatron_fix._SKIP_DUMMY_WGRAD is (not overlap_grad_reduce)
        # The self-check contract: one report carrying the gate value and param count.
        assert logged == [(overlap_grad_reduce, 3)]

    def test_rewrapping_stays_idempotent(self, monkeypatch):
        monkeypatch.setattr(megatron_fix, '_log_gate_status', lambda *unused: None)
        inner = megatron_fix.ddp_init_dummy_wgrad_wrapper(self._make_init(False))
        wrapped = megatron_fix.ddp_init_dummy_wgrad_wrapper(inner)
        ddp = SimpleNamespace()

        wrapped(ddp)

        assert [param.requires_dummy_wgrad for param in ddp.params_with_grad] == [False] * 3
        assert megatron_fix._SKIP_DUMMY_WGRAD is True

    def test_missing_ddp_attributes_raise(self, monkeypatch):
        monkeypatch.setattr(megatron_fix, '_log_gate_status', lambda *unused: None)
        wrapped = megatron_fix.ddp_init_dummy_wgrad_wrapper(lambda self: None)

        with pytest.raises(RuntimeError, match='params_with_grad'):
            wrapped(SimpleNamespace())


class TestGetDummyWgradGate:
    """The Megatron-side dummy source stops serving placeholders when armed."""

    def test_armed_gate_returns_none_without_calling_original(self, monkeypatch):
        factory = _DummyWgradFactory()
        monkeypatch.setattr(megatron_fix, '_SKIP_DUMMY_WGRAD', True)
        wrapped = megatron_fix.get_dummy_wgrad_skip_gate_wrapper(factory)

        assert wrapped([4, 3], torch.float32) is None
        assert wrapped([4, 3], torch.float32, zero=True) is None
        assert not factory.calls

    def test_disarmed_gate_defers_to_original(self, monkeypatch):
        factory = _DummyWgradFactory()
        monkeypatch.setattr(megatron_fix, '_SKIP_DUMMY_WGRAD', False)
        wrapped = megatron_fix.get_dummy_wgrad_skip_gate_wrapper(factory)

        assert wrapped([4, 3], torch.float32) is factory.sentinel
        assert wrapped([4, 3], torch.float32, zero=True, device='npu') is factory.sentinel
        assert factory.calls == [
            SimpleNamespace(shape=[4, 3], dtype=torch.float32, zero=False, device=None),
            SimpleNamespace(shape=[4, 3], dtype=torch.float32, zero=True, device='npu'),
        ]
