"""CPU-only tests of the actual shard probe using small NumPy tensor stand-ins."""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from adamw_math import first_step_expectation
from adamw_shard_layout import choose_probe_shards, owned_parameter_shards


class Tensor(np.ndarray):
    __hash__ = object.__hash__

    def __new__(cls, values, dtype=np.float32):
        return np.ndarray.view(np.asarray(values, dtype=dtype), cls)

    @property
    def device(self):
        return "cpu"

    def numel(self):
        return self.size

    def detach(self):
        return self

    def view(self, *shape):
        return self.reshape(*shape)

    def index_select(self, dim, indices):
        return np.take(self, np.asarray(indices, dtype=np.int64), axis=dim)

    def clone(self):
        return self.copy()

    def float(self):
        return self.astype(np.float32)

    def to(self, dtype):
        return self.astype(dtype)

    def abs(self):
        return np.abs(self)

    def count_nonzero(self):
        return Tensor(np.count_nonzero(self), dtype=np.int64)


def load_probe():
    torch = SimpleNamespace(
        no_grad=lambda: lambda fn: fn,
        arange=lambda count, device=None: Tensor(np.arange(count), dtype=np.int64),
        isclose=lambda left, right, **kw: Tensor(np.isclose(left, right, **kw), dtype=bool),
        equal=lambda left, right: np.array_equal(left, right),
    )
    common = SimpleNamespace(scalar=float, active_indices=lambda *args: (_ for _ in ()).throw(AssertionError("Unexpected large tensor")))
    spec = importlib.util.spec_from_file_location("_cpu_adamw_dist_probe", Path(__file__).with_name("adamw_dist_update_probe.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "optimizer_update_probe": common}):
        spec.loader.exec_module(module)
    return module


def fixture(start=3, end=9):
    param = Tensor(np.linspace(-0.03, 0.04, 12).reshape(3, 4))
    param.requires_grad = True
    param.main_grad = Tensor(np.arange(1, 13).reshape(3, 4))
    master = Tensor(param.reshape(-1)[start:end].copy())
    param.main_param = master
    group = {"params": [master], "lr": 5e-7, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
    base = SimpleNamespace(param_groups=[group], state={}, adam_w_mode=True)
    owned = SimpleNamespace(start=start, end=end)
    leaf = SimpleNamespace(
        optimizer=base, config=SimpleNamespace(use_precision_aware_optimizer=False),
        model_float16_groups=[[param]], shard_fp32_from_float16_groups=[[master]], model_fp32_groups=[[]],
        _get_model_param_range_map=lambda unused: {"param": owned},
    )
    named = {"module.language_model.decoder.layers.9.test.weight": param}
    return named, leaf, param, master, base


class DistributedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_probe()

    def prepare_updated(self, start=3, end=9):
        named, leaf, param, master, base = fixture(start, end)
        snapshots, rates = self.probe.capture_update_points(named, leaf, [])
        before = param.copy()
        norm = 100.0
        expected, moment1, moment2 = first_step_expectation(
            master.copy(), param.main_grad.reshape(-1)[start:end], 5e-7,
            0.9, 0.999, 1e-8, 0.01, 1 / (norm + 1e-6),
        )
        master[:] = expected
        param.reshape(-1)[start:end] = master
        base.state[master] = {"step": 1, "exp_avg": Tensor(moment1), "exp_avg_sq": Tensor(moment2)}
        return snapshots, rates, norm, before, param, master, base

    def verify(self, snapshots, rates, norm):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.probe.verify_update_points(snapshots, norm, 1.0, 0, rates)

    def test_nonzero_and_boundary_offsets_use_owned_gradient_and_model_slice(self):
        for start, end in ((0, 4), (3, 9), (10, 12)):
            with self.subTest(start=start, end=end):
                snapshots, rates, norm, before, param, _, _ = self.prepare_updated(start, end)
                summary = self.verify(snapshots, rates, norm)
                self.assertEqual(summary["failed_checks"], [])
                self.assertEqual(summary["largest_model_changes"][0]["owned_model_range"], [start, end])
                self.assertGreater(summary["master_delta_max"], 0)
                np.testing.assert_array_equal(param.reshape(-1)[:start], before.reshape(-1)[:start])
                np.testing.assert_array_equal(param.reshape(-1)[end:], before.reshape(-1)[end:])
                np.testing.assert_array_equal(snapshots[0]["grad_before"], param.main_grad.reshape(-1)[start:end])

    def test_corrupt_moments_and_model_copy_are_rejected(self):
        for fault in ("exp_avg", "exp_avg_sq", "model"):
            with self.subTest(fault=fault):
                snapshots, rates, norm, _, param, master, base = self.prepare_updated()
                if fault == "model":
                    param.reshape(-1)[3] += 0.1
                else:
                    base.state[master][fault][0] += 0.1
                with self.assertRaisesRegex(AssertionError, "state/update/copy"):
                    self.verify(snapshots, rates, norm)

    def test_bad_shard_size_rejected(self):
        named, leaf, param, _, _ = fixture()
        wrong = Tensor(np.zeros(2))
        leaf.shard_fp32_from_float16_groups = [[wrong]]
        leaf.optimizer.param_groups[0]["params"] = [wrong]
        param.main_param = wrong
        with self.assertRaisesRegex(AssertionError, "Master size"):
            owned_parameter_shards(named, leaf)

    def test_fresh_optimizer_required(self):
        named, leaf, _, master, base = fixture()
        base.state[master] = {"step": 1}
        with self.assertRaisesRegex(AssertionError, "fresh optimizer"):
            self.probe.capture_update_points(named, leaf, [])

    def test_fallback_covers_dense_and_expert_leaves_without_requested_ownership(self):
        first, dense, _, _, _ = fixture()
        second, expert, _, _, _ = fixture()
        second = {name.replace("layers.9", "layers.0.mlp.experts"): param for name, param in second.items()}
        shards = owned_parameter_shards({**first, **second}, SimpleNamespace(chained_optimizers=[dense, expert]))
        chosen = choose_probe_shards(shards, [("unowned", object())])
        self.assertEqual({id(shard["leaf"]) for shard in chosen}, {id(dense), id(expert)})


if __name__ == "__main__":
    unittest.main()
