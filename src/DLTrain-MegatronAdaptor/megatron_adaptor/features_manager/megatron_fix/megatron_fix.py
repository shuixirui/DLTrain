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


class MegatronFixFeature(MindSpeedFeature):
    """Fixes for upstream Megatron-LM bugs: enabled via explicit ``--fix-*`` switches,
    removed once the corresponding fix lands in the tracked upstream release.
    """

    def __init__(self):
        super().__init__('megatron-fix', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--fix-ddp-one-shot-stream',
            action='store_true',
            default=False,
            help='Fix FSDP memory retention caused by the one-shot DDP/FSDP init side stream '
            '(stream-local CUDACachingAllocator cache; upstream NVIDIA/Megatron-LM#6885): '
            'applies the upstream three-way stream selection - current stream for eager, '
            'shared capture stream for full_iteration, dedicated stream otherwise.',
        )
        group.add_argument(
            '--fix-non-overlap-dummy-wgrad',
            action='store_true',
            default=False,
            help='Avoid the persistent full-sized dummy wgrad tensors left behind by the fused '
            'gradient-accumulation path when DDP grad reduction does not overlap backward '
            '(overlap_grad_reduce=False). The dummy only exists to drive the DDP '
            'post-accumulate hook on the main autograd thread while reduction overlaps '
            'backward; without overlap finalization synchronizes every bucket explicitly.',
        )

    def register_patches(self, patch_manager, args):
        # ``getattr`` defaults keep partial-args callers (unit tests, non-CLI paths)
        # working when a new ``--fix-*`` switch is added to this feature.
        if getattr(args, 'fix_ddp_one_shot_stream', False):
            from megatron_adaptor.patches.megatron.ddp_one_shot_stream import (
                ddp_wrap_with_upstream_stream_selection,
                get_model_with_upstream_stream_selection,
            )

            patch_manager.register_patch(
                'megatron.training.training.get_model',
                get_model_with_upstream_stream_selection,
            )
            patch_manager.register_patch(
                'megatron.training.models.dist_utils._ddp_wrap',
                ddp_wrap_with_upstream_stream_selection,
            )

        if getattr(args, 'fix_non_overlap_dummy_wgrad', False):
            from megatron.core.tensor_parallel import layers as tp_layers
            from megatron_adaptor.patches.megatron.dummy_wgrad_non_overlap import (
                ddp_init_dummy_wgrad_wrapper,
                get_dummy_wgrad_skip_gate_wrapper,
            )

            # MCore binds this helper from TransformerEngine when its module is imported,
            # so the symbol is missing when TransformerEngineNPU is absent. Without it the
            # gate could not skip anything and the switch would fail silently, so fail
            # loudly here (the revision check MA requires before changing a patch).
            # The DDP-side structure the wrapper relies on (params_with_grad/ddp_config)
            # is checked at runtime instead, where it can actually be observed.
            if not hasattr(tp_layers, 'get_dummy_wgrad'):
                raise RuntimeError(
                    '--fix-non-overlap-dummy-wgrad requires '
                    'megatron.core.tensor_parallel.layers.get_dummy_wgrad, which is provided '
                    'by TransformerEngineNPU. Check the installed revisions (see README).'
                )

            patch_manager.register_patch(
                'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                ddp_init_dummy_wgrad_wrapper,
            )
            patch_manager.register_patch(
                'megatron.core.tensor_parallel.layers.get_dummy_wgrad',
                get_dummy_wgrad_skip_gate_wrapper,
            )
