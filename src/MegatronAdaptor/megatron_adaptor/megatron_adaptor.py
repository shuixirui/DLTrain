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
from logging import getLogger

from torch_npu.contrib import transfer_to_npu  # noqa: F401

from megatron_adaptor.auto_execute import AutoExecuteFunction
from megatron_adaptor.features_manager.features_manager import FeaturesManager

LOG = getLogger(__name__)
_IS_FEATURES_PATCHED = False


@AutoExecuteFunction
def patch_features():
    global _IS_FEATURES_PATCHED
    if _IS_FEATURES_PATCHED:
        return
    _IS_FEATURES_PATCHED = True

    LOG.info("start to patch features in megatron adaptor.")
    from megatron_adaptor.utils.args_utils import get_mindspeed_args

    ms_args = get_mindspeed_args()

    FeaturesManager.apply_features_pre_patches(ms_args)
    FeaturesManager.apply_features_patches(ms_args)
