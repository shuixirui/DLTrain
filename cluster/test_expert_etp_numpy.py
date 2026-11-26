"""CPU regression check of the replacement method bodies, without NPU/PyTorch.

NumPy supplies only tensor layout operations; the downstream scatter is captured.
This does not test real process groups or NPU collectives.
"""

import re
import textwrap
from types import SimpleNamespace

import numpy as np

from patch_expert_etp import METHODS


class Tensor:
    def __init__(self, value):
        self.array = np.asarray(value)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        return Tensor(self.array[key])

    def t(self):
        return Tensor(self.array.T)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))

    def to(self, *, dtype):
        return Tensor(self.array.astype(dtype))


def align(weight, shape, name):
    if tuple(weight.shape) == tuple(shape):
        return weight
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(shape):
        return weight.t().contiguous()
    raise ValueError(f"Unexpected {name} shape {weight.shape}; expected {shape}")


class Capture:
    def hf_to_megatron(self, weights, module):
        return weights


class Parent(Capture):
    def _normalize_expert_param_name(self, name):
        return name


namespace = {
    "torch": SimpleNamespace(chunk=lambda value, count, dim: [Tensor(v) for v in np.split(value.array, count, axis=dim)]),
    "extract_expert_number_from_param": lambda name: int(re.search(r"weight(\d+)$", name).group(1)),
    "get_module_and_param_from_name": lambda module, name: (module, module.weight),
    "_align_weight_to_shape": align,
    "Parent": Parent,
}
for name, method in METHODS.items():
    exec("class " + name + "(Parent):\n" + textwrap.indent(textwrap.dedent(method).strip(), "    "), namespace)

Gate = namespace["ExpertMLPGateUpProjMapping"]
Down = namespace["ExpertMLPDownProjMapping"]
gate = np.arange(40, dtype=np.float64).reshape(8, 5) + 100
up = np.arange(40, dtype=np.float64).reshape(8, 5) + 1000
down = np.arange(40, dtype=np.float64).reshape(5, 8) + 2000
count = 0

for tp in (1, 2, 4):
    module = SimpleNamespace(weight=Tensor(np.zeros((16 // tp, 5), dtype=np.float32)))
    mapper = Gate()
    mapper.tp_size, mapper.tp_rank = tp, 0
    mapper.megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight1"
    mapper._gated_mapping = Capture()
    fused = np.concatenate((gate, up), axis=0)
    fixtures = [
        np.stack((fused - 10, fused, fused + 10)),
        np.stack((fused.T - 10, fused.T, fused.T + 10)),
        np.stack((np.stack((gate, up)) - 10, np.stack((gate, up)), np.stack((gate, up)) + 10)),
        np.stack((np.stack((gate.T, up.T)) - 10, np.stack((gate.T, up.T)), np.stack((gate.T, up.T)) + 10)),
    ]
    for fixture in fixtures:
        converted = mapper.hf_to_megatron(Tensor(fixture), module)
        np.testing.assert_array_equal(converted["gate"].array, gate.astype(np.float32))
        np.testing.assert_array_equal(converted["up"].array, up.astype(np.float32))
        assert converted["gate"].dtype == module.weight.dtype
        count += 1
    mapper.tp_rank = 1
    if tp > 1:
        assert mapper.hf_to_megatron(None, module) is None
        count += 1

    module = SimpleNamespace(weight=Tensor(np.zeros((5, 8 // tp), dtype=np.float32)))
    mapper = Down()
    mapper.tp_size, mapper.tp_rank = tp, 0
    mapper.megatron_param = "decoder.layers.0.mlp.experts.linear_fc2.weight1"
    for fixture in (np.stack((down - 10, down, down + 10)), np.stack((down.T - 10, down.T, down.T + 10))):
        converted = mapper.hf_to_megatron(Tensor(fixture), module)
        np.testing.assert_array_equal(converted.array, down.astype(np.float32))
        assert converted.dtype == module.weight.dtype
        count += 1
    if tp > 1:
        mapper.tp_rank = 1
        assert mapper.hf_to_megatron(None, module) is None
        count += 1

mapper = Gate()
mapper.tp_size, mapper.tp_rank = 2, 0
mapper.megatron_param = "decoder.layers.0.mlp.experts.linear_fc1.weight1"
mapper._gated_mapping = Capture()
try:
    mapper.hf_to_megatron(Tensor(np.zeros((3, 15, 5))), SimpleNamespace(weight=Tensor(np.zeros((8, 5)))))
except ValueError:
    count += 1
else:
    raise AssertionError("Wrong shapes must still be rejected")

print(f"EXPERT_ETP_LAYOUT_TEST_OK: {count} cases; scatter captured, no real NPU/collective execution")
