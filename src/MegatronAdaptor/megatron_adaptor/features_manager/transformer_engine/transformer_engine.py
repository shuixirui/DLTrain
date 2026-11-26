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

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature
from megatron_adaptor.patches import PatchesManager


class TransformerEngineBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("transformer-engine-basic", optimization_level=0)

    def register_patches(self, patch_manager: PatchesManager, args):
        from megatron_adaptor.patches.transformer_engine.transformer_engine import (
            get_npu_grad_accum_fusion_available,
            column_parallel_linear_init_wrapper,
            transformer_config_post_init_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
            transformer_config_post_init_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers._grad_accum_fusion_available",
            get_npu_grad_accum_fusion_available,
        )
        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__",
            column_parallel_linear_init_wrapper,
        )

        from megatron_adaptor.patches.transformer_engine.transformer_engine import (
            safe_unpickler_find_class_patch,
        )

        patch_manager.register_patch(
            "megatron.core.safe_globals.SafeUnpickler.find_class",
            safe_unpickler_find_class_patch,
        )
