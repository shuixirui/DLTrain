# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Behavioral coverage for recipe-state factory and quantizer layouts."""

from __future__ import annotations

import pytest
import torch

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    MXFP4BlockScaling,
    MXFP8BlockScaling,
    W4A8BlockScaling,
)
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.quantization.state.base import RecipeState
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.w4a8_tensor import W4A8WeightQuantizer
from transformer_engine.pytorch.utils import check_npu_version
from utils import npu_available

pytestmark = [
    pytest.mark.skipif(not npu_available(), reason="NPU device is required"),
    pytest.mark.skipif(not check_npu_version(NPUVersion.A5), reason="Atlas A5 is required"),
]


@pytest.mark.parametrize(
    ("recipe", "mode", "num_quantizers", "quantizer_type"),
    [
        (DelayedScaling(), "forward", 2, Float8Quantizer),
        (Float8CurrentScaling(), "backward", 2, Float8CurrentScalingQuantizer),
        (Float8BlockScaling(), "forward", 3, Float8BlockQuantizer),
        (Float8BlockScaling(), "backward", 2, Float8BlockQuantizer),
        (MXFP8BlockScaling(), "forward", 2, MXFP8Quantizer),
        (MXFP4BlockScaling(), "forward", 3, MXFP4Quantizer),
        (MXFP4BlockScaling(), "backward", 2, MXFP4Quantizer),
        (W4A8BlockScaling(), "forward", 3, None),
        (W4A8BlockScaling(), "backward", 3, MXFP8Quantizer),
    ],
)
def test_recipe_state_factory_builds_expected_quantizer_layout(
    recipe, mode, num_quantizers, quantizer_type
):
    state = RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
        device=torch.device("npu"),
    )
    quantizers = state.make_quantizers()

    assert state.recipe is recipe
    assert state.mode == mode
    assert len(quantizers) == num_quantizers
    if quantizer_type is not None:
        assert all(isinstance(quantizer, quantizer_type) for quantizer in quantizers)
    else:
        assert isinstance(quantizers[0], MXFP8Quantizer)
        assert isinstance(quantizers[1], W4A8WeightQuantizer)
        assert isinstance(quantizers[2], MXFP8Quantizer)


def test_float8_block_state_uses_role_specific_block_dimensions():
    recipe = Float8BlockScaling(
        x_block_scaling_dim=(1, 2),
        w_block_scaling_dim=2,
        grad_block_scaling_dim=1,
    )

    forward_quantizers = RecipeState.create(
        recipe,
        mode="forward",
        num_quantizers=3,
        device=torch.device("npu"),
    ).make_quantizers()
    backward_quantizers = RecipeState.create(
        recipe,
        mode="backward",
        num_quantizers=2,
        device=torch.device("npu"),
    ).make_quantizers()

    assert forward_quantizers[0].block_scaling_dim == (1, 2)
    assert forward_quantizers[1].block_scaling_dim == 2
    assert forward_quantizers[2].block_scaling_dim == (1, 2)
    assert backward_quantizers[0].block_scaling_dim == 1
    assert backward_quantizers[0].columnwise_group_quant_lhs_trans is True
    assert backward_quantizers[1].columnwise_group_quant_lhs_trans is False


def test_w4a8_forward_layout_preserves_hybrid_weight_quantizer():
    state = RecipeState.create(
        W4A8BlockScaling(),
        mode="forward",
        num_quantizers=3,
        device=torch.device("npu"),
    )
    input_quantizer, weight_quantizer, output_quantizer = state.make_quantizers()

    assert isinstance(input_quantizer, MXFP8Quantizer)
    assert isinstance(weight_quantizer, W4A8WeightQuantizer)
    assert isinstance(output_quantizer, MXFP8Quantizer)
    assert weight_quantizer.block_size == state.recipe.block_size
