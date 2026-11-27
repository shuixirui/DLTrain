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

_MLA_ROPE_MODULE = "megatron.core.fusions.fused_mla_yarn_rope_apply"


class MLARopeFeature(MindSpeedFeature):
    """Ascend adaptation for MLA + YARN RoPE fusion.

    Megatron 0.18 supports ``apply_rope_fusion`` for MLA + YARN with a Triton kernel
    that is much slower than the CANN rotary ops on Ascend. This feature disables that
    fusion by default (unless the TransformerEngineNPU backend is available or the user
    opts in) and routes Megatron's fused MLA RoPE to the NPU backend when present.

    See ``megatron_adaptor/patches/megatron/mla_rope.py``.
    """

    def __init__(self):
        super().__init__("mla-rope", optimization_level=0)

    def register_args(self, parser):
        if not any("--mla-rope-fusion" in action.option_strings for action in parser._actions):
            parser.add_argument(
                "--mla-rope-fusion",
                action="store_true",
                default=False,
                help="Force-enable MLA + YARN RoPE fusion. It is disabled by default on "
                "Ascend because Megatron's Triton kernel is much slower than the CANN "
                "rotary op.",
            )

    def register_patches(self, patch_manager, args):
        from megatron_adaptor.patches.megatron.mla_rope import mla_rope_fusion_post_init_wrapper
        from megatron_adaptor.patches.megatron.mla_rope import npu_mla_rope_backend

        patch_manager.register_patch(
            "megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
            mla_rope_fusion_post_init_wrapper,
        )

        backend_for_q, backend_for_kv = npu_mla_rope_backend()
        if backend_for_q is None or backend_for_kv is None:
            return

        patch_manager.register_patch(
            "megatron.core.fusions.fused_mla_yarn_rope_apply.fused_apply_mla_rope_for_q",
            backend_for_q,
        )
        patch_manager.register_patch(
            "megatron.core.fusions.fused_mla_yarn_rope_apply.fused_apply_mla_rope_for_kv",
            backend_for_kv,
        )
