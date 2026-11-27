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


class NPUDeterministicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("npu-deterministic")

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--npu-deterministic',
            action='store_true',
            default=False,
            help='enable deterministic computing for npu.',
        )

    def register_patches(self, patch_manager, args):
        if args.npu_deterministic:
            from megatron_adaptor.patches.patch_utils import is_megatron_training_available

            if not is_megatron_training_available():
                return

            from megatron_adaptor.patches.megatron.npu_deterministic import npu_deterministic_wrapper

            patch_manager.register_patch('megatron.training.initialize._set_random_seed', npu_deterministic_wrapper)
