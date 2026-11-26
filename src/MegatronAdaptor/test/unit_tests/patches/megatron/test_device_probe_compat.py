# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests for the ``torch.cuda.device_memory_used`` NPU backfill.

Covers the public patch entry: the query reports torch/CANN-visible device
memory from ``torch.npu.mem_get_info``, and a query failure degrades to ``nan``
instead of aborting training. Device collaborators are faked, so no NPU
allocation is required.

Run:
    bash test/unit_tests/run_unit_tests.sh <Megatron-LM path>
"""

import math
from types import SimpleNamespace

import torch

from megatron_adaptor.patches.megatron import device_probe_compat

GIB = 1024**3


def _guard_query(monkeypatch):
    """Register the current query for teardown; the patch overwrites it."""
    monkeypatch.setattr(torch.cuda, 'device_memory_used', torch.cuda.device_memory_used)


def _patch(monkeypatch):
    _guard_query(monkeypatch)
    monkeypatch.setattr(device_probe_compat, '_fallback_warned', False)
    device_probe_compat.patch_device_memory_used(SimpleNamespace(log_device_memory_used=False))


def test_reports_torch_visible_usage(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(torch.npu, 'mem_get_info', lambda device=None: (2 * GIB, 8 * GIB))

    assert torch.cuda.device_memory_used() == 6 * GIB


def test_returns_nan_when_mem_get_info_fails(monkeypatch):
    def _raise(device=None):
        raise AttributeError('mem_get_info')

    _patch(monkeypatch)
    monkeypatch.setattr(torch.npu, 'mem_get_info', _raise)

    assert math.isnan(torch.cuda.device_memory_used())
