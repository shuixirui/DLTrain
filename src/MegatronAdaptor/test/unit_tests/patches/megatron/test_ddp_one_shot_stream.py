# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression tests for the ``--fix-ddp-one-shot-stream`` compatibility switch.

Covers the path delta around the switch: off (default) registers no patch and
leaves upstream paths untouched; on, eager initialization shares the current
stream (upstream NVIDIA/Megatron-LM#6885) while non-eager modes keep upstream
behavior (full_iteration -> shared capture stream, local/transformer_engine ->
dedicated stream). Stream collaborators are replaced with tiny fakes via
monkeypatch; no NPU device is required.

Run:
    bash test/unit_tests/run_unit_tests.sh <Megatron-LM path>
"""

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

import megatron_adaptor  # noqa: F401  ensure the MA patch entry is loaded
from megatron.training.training import ModelType
from megatron_adaptor.features_manager.megatron_fix.megatron_fix import MegatronFixFeature
from megatron_adaptor.patches.megatron import ddp_one_shot_stream as ddp_fix


class _FakeStream:
    """Minimal stream double recording wait_stream calls."""

    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


class _StreamCtx:
    """``torch.cuda.stream`` double tracking the active target."""

    def __init__(self, target, active):
        self._target = target
        self._active = active

    def __enter__(self):
        self._active.append(self._target)
        return self._target

    def __exit__(self, *exc_info):
        self._active.pop()


class _CudaStub:
    """``torch.cuda`` double: current stream, Stream() factory, stream ctx, device id."""

    def __init__(self):
        self.current = _FakeStream('current')
        self.created = []
        self.active = []
        self.shared = None

    def current_stream(self):
        return self.current

    def Stream(self):
        stream = _FakeStream('dedicated')
        self.created.append(stream)
        return stream

    def stream(self, target):
        return _StreamCtx(target, self.active)

    def current_device(self):
        return 0


class _FakeChunk:
    """Model chunk double accepted by the patched code paths."""

    def parameters(self):
        return []

    def cuda(self, device):
        pass


class _PatchManagerRecorder:
    """Records register_patch targets instead of applying them."""

    def __init__(self):
        self.registered = []

    def register_patch(self, target, func):
        self.registered.append(target)


def _make_pg(dp_cp_size=8):
    return SimpleNamespace(
        dp=SimpleNamespace(),
        cp=SimpleNamespace(),
        tp=SimpleNamespace(),
        pp=SimpleNamespace(rank=lambda: 0),
        dp_cp=SimpleNamespace(size=lambda: dp_cp_size),
        expt_dp=SimpleNamespace(size=lambda: 1),
    )


def _make_ddp_config(**overrides):
    config = SimpleNamespace(
        num_buckets=None,
        bucket_size=None,
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_model_config(cuda_graph_impl):
    return SimpleNamespace(cuda_graph_impl=cuda_graph_impl)


def _fake_get_megatron_ddp_config(_args):
    """``get_megatron_ddp_config`` double returning a default dummy config."""
    return _make_ddp_config()


def _make_fake_dp(cuda):
    """Build a DP class double recording construction kwargs and the active stream."""
    created = []

    class _FakeDP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active_at_init = list(cuda.active)
            created.append(self)

    return _FakeDP, created


class TestMegatronFixFeature:
    def test_switch_default_off_and_can_be_enabled(self):
        parser = ArgumentParser()
        MegatronFixFeature().register_args(parser)
        assert parser.parse_args([]).fix_ddp_one_shot_stream is False
        assert parser.parse_args(['--fix-ddp-one-shot-stream']).fix_ddp_one_shot_stream is True

    def test_patches_registered_only_when_switch_on(self):
        feature = MegatronFixFeature()
        patch_manager = _PatchManagerRecorder()

        # Switch off (default): no patch is registered, upstream paths stay untouched.
        feature.register_patches(patch_manager, Namespace(fix_ddp_one_shot_stream=False))
        assert not patch_manager.registered

        # Switch on: patch exactly the two init paths the upstream #6885 fix touches.
        feature.register_patches(patch_manager, Namespace(fix_ddp_one_shot_stream=True))
        assert patch_manager.registered == [
            'megatron.training.training.get_model',
            'megatron.training.models.dist_utils._ddp_wrap',
        ]


class TestDdpWrapUpstreamStreamSelection:
    """Path delta on the _ddp_wrap path after the switch; same three-way selection."""

    @staticmethod
    def _make_model(n_chunks=1):
        return [_FakeChunk() for _ in range(n_chunks)]

    def _setup(self, monkeypatch, cuda_graph_impl):
        cuda = _CudaStub()
        monkeypatch.setattr(ddp_fix, 'torch', SimpleNamespace(cuda=cuda))
        monkeypatch.setattr(ddp_fix, 'get_model_config', lambda chunk: _make_model_config(cuda_graph_impl))
        dp_cls, created = _make_fake_dp(cuda)
        monkeypatch.setattr(ddp_fix, 'DistributedDataParallel', dp_cls)
        return cuda, created

    def test_eager_uses_current_stream_without_one_shot_stream(self, monkeypatch):
        cuda, created = self._setup(monkeypatch, 'none')
        model = self._make_model()

        wrapped = ddp_fix.ddp_wrap_with_upstream_stream_selection(
            model,
            False,
            _make_ddp_config(),
            False,
            pg_collection=_make_pg(),
        )

        # Eager branch (the fix): no one-shot stream, init on the current stream,
        # no sync events (same-stream guards).
        assert not cuda.created
        assert created[0].active_at_init == [cuda.current]
        assert not cuda.current.waits
        assert not cuda.active
        assert wrapped == created

    def test_full_iteration_uses_shared_capture_stream(self, monkeypatch):
        cuda, created = self._setup(monkeypatch, 'full_iteration')
        cuda.shared = _FakeStream('shared')
        monkeypatch.setattr(ddp_fix, 'get_shared_capture_stream', lambda: cuda.shared)
        model = self._make_model()

        ddp_fix.ddp_wrap_with_upstream_stream_selection(
            model,
            False,
            _make_ddp_config(),
            False,
            pg_collection=_make_pg(),
        )

        # Non-eager branch: upstream behavior, shared capture stream with sync pair.
        assert not cuda.created
        assert created[0].active_at_init == [cuda.shared]
        assert cuda.shared.waits == [cuda.current]
        assert cuda.current.waits == [cuda.shared]
        assert not cuda.active

    @pytest.mark.parametrize('cuda_graph_impl', ['local', 'transformer_engine'])
    def test_dedicated_stream_for_other_implementations(self, monkeypatch, cuda_graph_impl):
        cuda, created = self._setup(monkeypatch, cuda_graph_impl)
        model = self._make_model()

        ddp_fix.ddp_wrap_with_upstream_stream_selection(
            model,
            False,
            _make_ddp_config(),
            False,
            pg_collection=_make_pg(),
        )

        # Non-eager branch: upstream behavior, dedicated one-shot stream with sync pair.
        assert len(cuda.created) == 1
        assert created[0].active_at_init == [cuda.created[0]]
        assert cuda.created[0].waits == [cuda.current]
        assert cuda.current.waits == [cuda.created[0]]
        assert not cuda.active


class TestGetModelUpstreamStreamSelection:
    """Path delta on the get_model path after the switch; same three-way selection."""

    @staticmethod
    def _make_args():
        return SimpleNamespace(
            create_all_gather_group=False,
            init_model_with_meta_device=False,
            use_torch_fsdp2=False,
            use_megatron_fsdp=False,
            use_cpu_initialization=False,
            fp16=False,
            bf16=False,
            virtual_pipeline_model_parallel_size=None,
            overlap_param_gather_with_optimizer_step=False,
            data_parallel_random_init=False,
        )

    def _setup(self, monkeypatch, cuda_graph_impl):
        cuda = _CudaStub()
        monkeypatch.setattr(ddp_fix, 'torch', SimpleNamespace(cuda=cuda))
        monkeypatch.setattr(ddp_fix, 'get_args', self._make_args)
        monkeypatch.setattr(ddp_fix, 'has_nvidia_modelopt', False)
        monkeypatch.setattr(ddp_fix, 'get_pg_size', lambda pg: 1)
        monkeypatch.setattr(ddp_fix, 'get_pg_rank', lambda pg: 1)
        monkeypatch.setattr(ddp_fix, 'is_pp_first_stage', lambda pg: False)
        monkeypatch.setattr(ddp_fix, 'is_pp_last_stage', lambda pg: False)
        monkeypatch.setattr(ddp_fix, 'correct_amax_history_if_needed', lambda model: None)
        monkeypatch.setattr(
            ddp_fix,
            'tensor_parallel',
            SimpleNamespace(set_defaults_if_not_set_tensor_model_parallel_attributes=lambda p: None),
        )
        monkeypatch.setattr(ddp_fix, 'get_model_config', lambda chunk: _make_model_config(cuda_graph_impl))
        monkeypatch.setattr(ddp_fix, 'get_megatron_ddp_config', _fake_get_megatron_ddp_config)
        monkeypatch.setattr(
            ddp_fix,
            'mpu',
            SimpleNamespace(
                get_pipeline_model_parallel_rank=lambda: 0,
                get_data_parallel_world_size=lambda **kwargs: 8,
            ),
        )

        recorded = {}

        def wrap_stub(model_chunks, config, ddp_config, **kwargs):
            recorded['stream'] = cuda.active[-1] if cuda.active else None
            recorded['result'] = [_FakeChunk()]
            return recorded['result']

        monkeypatch.setattr(ddp_fix, 'wrap_model_chunks_with_ddp', wrap_stub)
        monkeypatch.setattr(ddp_fix, 'get_shared_capture_stream', lambda: cuda.shared)
        return cuda, recorded

    @staticmethod
    def _call_get_model():
        def provider(**kwargs):
            return _FakeChunk()

        return ddp_fix.get_model_with_upstream_stream_selection(
            provider,
            ModelType.encoder_or_decoder,
            True,
            None,
            _make_pg(),
        )

    def test_eager_initializes_on_current_stream_without_one_shot_stream(self, monkeypatch):
        cuda, recorded = self._setup(monkeypatch, 'none')
        result = self._call_get_model()

        # Eager branch (the fix): no one-shot stream, init on the caller's stream,
        # no sync events.
        assert not cuda.created
        assert recorded['stream'] is cuda.current
        assert not cuda.current.waits
        assert not cuda.active
        assert result == recorded['result']

    def test_full_iteration_uses_shared_capture_stream(self, monkeypatch):
        cuda, recorded = self._setup(monkeypatch, 'full_iteration')
        cuda.shared = _FakeStream('shared')

        self._call_get_model()

        # Non-eager branch: upstream behavior, shared capture stream with sync pair.
        assert not cuda.created
        assert recorded['stream'] is cuda.shared
        assert cuda.shared.waits == [cuda.current]
        assert cuda.current.waits == [cuda.shared]
        assert not cuda.active

    @pytest.mark.parametrize('cuda_graph_impl', ['local', 'transformer_engine'])
    def test_dedicated_stream_for_other_implementations(self, monkeypatch, cuda_graph_impl):
        cuda, recorded = self._setup(monkeypatch, cuda_graph_impl)

        self._call_get_model()

        # Non-eager branch: upstream behavior, dedicated one-shot stream with sync pair.
        assert len(cuda.created) == 1
        assert recorded['stream'] is cuda.created[0]
        assert cuda.created[0].waits == [cuda.current]
        assert cuda.current.waits == [cuda.created[0]]
        assert not cuda.active
