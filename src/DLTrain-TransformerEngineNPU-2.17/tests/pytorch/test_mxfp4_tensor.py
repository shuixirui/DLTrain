# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.quantization import is_mxfp4_available
from transformer_engine.pytorch.tensor import MXFP4Quantizer, MXFP4Tensor

from utils import npu_available


_mxfp4_available, _reason_for_no_mxfp4 = is_mxfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(
    not npu_available() or not _mxfp4_available,
    reason=_reason_for_no_mxfp4 or "NPU device is required",
)


def _uniform_npu(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=math.prod(shape),
        dtype=torch.float32,
        device="cpu",
    ).reshape(shape)
    return values.to(device=torch.device("npu"), dtype=dtype).contiguous()


def test_mxfp4_quantize_invokes_dynamic_mx_quant() -> None:
    dense = _uniform_npu((64, 64), dtype=torch.bfloat16)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=True, with_rht=True)

    quantized = quantizer(dense)
    torch.npu.synchronize()

    assert isinstance(quantized, MXFP4Tensor)
    assert tuple(quantized.shape) == tuple(dense.shape)
    assert quantized.dtype == dense.dtype
    assert quantized._rowwise_data is not None
    assert quantized._rowwise_scale_inv is not None
    assert quantized._columnwise_data is not None
    assert quantized._columnwise_scale_inv is not None
    assert quantized._rowwise_data.device.type == "npu"
    assert quantized._columnwise_data.device.type == "npu"


def test_mxfp4_quantized_matmul_invokes_quant_matmul() -> None:
    inp = _uniform_npu((64, 64), dtype=torch.bfloat16)
    weight = _uniform_npu((96, 64), dtype=torch.bfloat16)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=True, with_rht=False)

    inp_mxfp4 = quantizer(inp)
    weight_mxfp4 = quantizer(weight)
    out = inp_mxfp4.matmul(
        weight_mxfp4,
        TensorUsage.LHS,
        TensorUsage.RHS_TRANS,
        torch.bfloat16,
    )
    torch.npu.synchronize()

    assert tuple(out.shape) == (inp.shape[0], weight.shape[0])
    assert out.dtype == torch.bfloat16
    assert out.device.type == "npu"
