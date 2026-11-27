# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Interface tests for the 0.18 Dense hybrid context parallel patch."""

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron_adaptor.features_manager.megatron.hybrid_context_parallel import HybridContextParallelFeature
from megatron_adaptor.patches.megatron import hybrid_context_parallel as dcp


def test_feature_registers_only_for_dcp():
    manager = Mock()
    feature = HybridContextParallelFeature()
    feature.register_patches(manager, SimpleNamespace(hybrid_context_parallel=False))
    feature.register_patches(manager, SimpleNamespace())
    manager.register_patch.assert_not_called()
    feature.register_patches(manager, SimpleNamespace(hybrid_context_parallel=True, num_experts=2))
    targets = [call.args[0] for call in manager.register_patch.call_args_list]
    assert len(targets) == len(set(targets)) == 6
    assert set(targets) == {
        'megatron.core.parallel_state.create_hybrid_dp_cp_groups',
        'megatron.core.transformer.attention.Attention.forward',
        'megatron.core.datasets.data_schedule.HybridCPDataLoaderWrapper',
        'megatron.core.pipeline_parallel.schedules.hybrid_context_parallel_forward_backward',
        'megatron.core.utils.get_batch_on_this_tp_rank',
        'megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__',
    }


@pytest.mark.parametrize('reset_attention,reset_positions', [(False, False), (True, True)])
def test_gpt_eod_resets_at_document_boundaries(monkeypatch, reset_attention, reset_positions):
    from megatron import training
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

    monkeypatch.setattr(training, 'get_tokenizer', lambda: SimpleNamespace(eod=99))
    args = Namespace(
        hybrid_context_parallel=True,
        reset_attention_mask=reset_attention,
        reset_position_ids=reset_positions,
        create_attention_mask_in_dataloader=True,
    )
    HybridContextParallelFeature().pre_validate_args(args)
    tokens = torch.tensor([1, 99, 3, 4, 99])
    labels = torch.tensor([99, 3, 4, 99, 6])
    mask, loss_mask, positions = _get_ltor_masks_and_position_ids(
        tokens, 99, args.reset_position_ids, args.reset_attention_mask, False, args.create_attention_mask_in_dataloader
    )
    assert mask is None
    assert args.reset_attention_mask == reset_attention
    assert args.reset_position_ids == reset_positions
    sample = dict(tokens=tokens, labels=labels, loss_mask=loss_mask, position_ids=positions)
    result = dcp.prepare_raw_batch([sample], pool_size=2)
    assert len(result) == 1
    packed = result[0]
    assert packed['cu_seqlens'].tolist() == [0, 4, 8]
    assert packed['labels'].tolist() == [99, 3, 0, 0, 4, 99, 6, 0]
    assert packed['loss_mask'].tolist() == [1, 1, 0, 0, 1, 1, 1, 0]
    assert packed['position_ids'].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_explicit_packed_provider_preserves_routing_fields():
    sample = dict(
        loss_mask=torch.linspace(0, 1, 16),
        position_ids=torch.arange(16),
        labels=torch.arange(16),
        tokens=torch.arange(16),
        cu_seqlens=torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    packed = dcp.prepare_raw_batch([sample], 2, allow_eod=False, sequence_parallel_size=4)[0]
    assert list(packed) == ['tokens', 'labels', 'loss_mask', 'position_ids', 'cu_seqlens']
    assert all(value is sample[key] for key, value in packed.items())


def test_custom_dense_mask_is_rejected():
    with pytest.raises(ValueError, match='dense attention_mask'):
        dcp.prepare_raw_batch([{'attention_mask': torch.ones(1, 2, 2)}], pool_size=2)


class Group:
    def __init__(self, members):
        self.members = members

    def size(self):
        return len(self.members)


def test_dp1_loader_selects_only_assigned_native_ids():
    class Native:
        def __next__(self):
            return 'native'

    cls = dcp.hybrid_loader_wrapper(Native)
    obj = object.__new__(cls)
    obj.dp_group = Group([0])
    obj.dp_cp_group = SimpleNamespace(rank=lambda: 1)
    obj.config = object()
    waves = [[[0], [1]], [[2], [2]]]

    def assign(lengths, config):
        assert lengths == [(0, 8), (1, 16), (2, 8)]
        return None, waves

    obj.cp_balancing_scheduler = SimpleNamespace(get_groups_and_subsamples=assign)
    obj.unpack_batch = lambda batch: ['a', 'b', 'c']
    obj.data_iterator = iter([[{'cu_seqlens': torch.tensor([0, 8, 24, 32])}]])
    assert next(obj) == ({1: 'b', 2: 'c'}, waves)
    obj.dp_group = Group([0, 1])
    assert next(obj) == 'native'


def test_rotary_scope_uses_actual_group_and_restores_after_error():
    static, actual = Group([4, 5, 6, 7]), Group([4, 5])
    observed = []

    shared = SimpleNamespace(cp=static)

    def native(self, hidden, packed_seq_params=None):
        observed.append(self.pg_collection.cp)
        assert shared.cp is static
        raise RuntimeError('injected')

    model = SimpleNamespace(config=SimpleNamespace(hybrid_context_parallel=True), pg_collection=shared)
    packed = SimpleNamespace(qkv_format='thd', local_cp_size=2, cp_group=actual)
    with pytest.raises(RuntimeError, match='injected'):
        dcp.hybrid_attention_forward_wrapper(native)(model, 'hidden', packed_seq_params=packed)
    assert model.pg_collection is shared
    assert observed == [actual]


@pytest.mark.parametrize("width", [1, 2, 4])
def test_native_tp_receives_one_document_for_mbs_two(monkeypatch, width):
    from megatron.core import parallel_state

    singleton = Group([3])
    monkeypatch.setattr(parallel_state, 'get_hybrid_data_context_parallel_groups', lambda group_size: singleton)
    cu = torch.tensor([[0, 16]], dtype=torch.int32)
    native = Mock(return_value=dict(cu_seqlens=cu, cu_seqlens_padded=cu if width > 1 else None, hybrid_cp_group=None))
    wrapped = dcp.hybrid_tp_batch_wrapper(native)
    token = dcp._JOB_WIDTH.set(width)
    try:
        result = wrapped({}, micro_batch_size=2, cp_size=4)
    finally:
        dcp._JOB_WIDTH.reset(token)
    native.assert_called_once_with({}, micro_batch_size=1, cp_size=width)
    assert result['cu_seqlens_padded'] is cu
    assert result['hybrid_cp_group'] is (singleton if width == 1 else None)
    wrapped({}, micro_batch_size=2, cp_size=4)
    native.assert_called_with({}, micro_batch_size=2, cp_size=4)


@pytest.mark.parametrize(
    'wave,budget,expected_widths',
    [
        ([[0, 1], [2]], 16, [1, 1]),
        ([[0, 1], [2]], 8, [1, 1]),
        ([[0, 1], [0]], 16, [2, 1]),
        ([[0, 1], [0, 1]], 16, [2, 2]),
    ],
)
@pytest.mark.parametrize('forward_only', [False, True])
def test_schedule_keeps_same_wave_documents_separate(monkeypatch, wave, budget, expected_widths, forward_only):
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules
    from megatron.core import rerun_state_machine

    pool = SimpleNamespace(rank=lambda: 0)
    tp = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(parallel_state, 'get_data_parallel_group', lambda **kwargs: pool)
    monkeypatch.setattr(parallel_state, 'get_tensor_model_parallel_group', lambda: tp)
    monkeypatch.setattr(parallel_state, 'get_tensor_model_parallel_src_rank', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda **kwargs: None)
    monkeypatch.setattr(rerun_state_machine, 'RerunDataIterator', lambda iterator: iterator)
    observed = []

    def forward_step(_, iterator, model, num_microbatches, input_tensor, store, config, width, collect, **kwargs):
        batch = next(iterator)
        observed.append((width, batch['cu_seqlens'][0].tolist(), batch['tokens'].tolist()))
        assert batch['cu_seqlens_padded'] is batch['cu_seqlens']
        assert batch['max_seqlen'].tolist() == [8]
        assert batch['local_cp_size'].tolist() == [width]
        return None, int(batch['loss_mask'].sum().item())

    backward = Mock()
    monkeypatch.setattr(schedules, 'forward_step', forward_step)
    monkeypatch.setattr(schedules, 'backward_step', backward)
    samples = {
        gid: {
            'tokens': torch.arange(gid * 8, (gid + 1) * 8),
            'labels': torch.arange(gid * 8, (gid + 1) * 8),
            'loss_mask': torch.ones(8),
            'position_ids': torch.arange(8),
        }
        for gid in (0, 1, 2)
    }
    result = dcp.hybrid_schedule(
        None,
        iter([(samples, [wave])]),
        None,
        1,
        None,
        None,
        [],
        SimpleNamespace(max_seqlen_per_dp_cp_rank=budget),
        False,
        False,
        forward_only,
        nullcontext,
        0,
        lambda *args: False,
        None,
    )
    assert [row[0] for row in observed] == expected_widths
    assert [row[1] for row in observed] == [[0, 8], [0, 8]]
    assert result[1] == 16
    assert sum((row[2][0] for row in observed), []) == list(range(16))
    assert backward.call_count == (0 if forward_only else 2)
    assert dcp._JOB_WIDTH.get() is None
