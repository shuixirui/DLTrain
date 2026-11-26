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
"""Opt-in adaptation of the upstream 0.18 Hybrid Context Parallel feature."""

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature
from megatron_adaptor.patches.megatron import hybrid_context_parallel as dcp


class HybridContextParallelFeature(MindSpeedFeature):
    """The native flag owns activation; MA adapts runtime CP interfaces."""

    def __init__(self):
        super().__init__('hybrid-context-parallel', optimization_level=2)

    def is_need_apply(self, args):
        return getattr(args, self.feature_name, False) is True

    def pre_validate_args(self, args):
        if self.is_need_apply(args) and not getattr(args, 'sft', False):
            # GPT DCP expresses causal boundaries with cu_seqlens.
            args.create_attention_mask_in_dataloader = False

    def register_patches(self, patch_manager, args):
        if not self.is_need_apply(args):
            return
        targets = {
            # Add the missing singleton process group for jobs that use CP1.
            'megatron.core.parallel_state.create_hybrid_dp_cp_groups': dcp.hybrid_groups_wrapper,
            # Expose the packed job group to RoPE before TENPU binds its attention backend.
            'megatron.core.transformer.attention.Attention.forward': dcp.hybrid_attention_forward_wrapper,
            # Split GPT samples at EOD and reuse native document assignment and routing.
            'megatron.core.datasets.data_schedule.HybridCPDataLoaderWrapper': dcp.hybrid_loader_wrapper,
            # Share the job plan with TP peers and pass the actual CP width to the 0.18 forward step for loss
            # scaling.
            'megatron.core.pipeline_parallel.schedules.hybrid_context_parallel_forward_backward': dcp.hybrid_schedule,
            # Reuse native TP broadcast for one document, adding the missing CP1 metadata.
            'megatron.core.utils.get_batch_on_this_tp_rank': dcp.hybrid_tp_batch_wrapper,
            # Keep the per-layer communication type and stream; native TE forward binds each job's PG.
            'megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__': dcp.hybrid_te_init_wrapper,
        }
        for path, implementation in targets.items():
            patch_manager.register_patch(path, implementation)
