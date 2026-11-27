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

from megatron_adaptor.features_manager.features_manager import FeaturesManager
from megatron_adaptor.features_manager.megatron.megatron_basic import MegatronBasicFeature
from megatron_adaptor.features_manager.megatron.mla_rope import MLARopeFeature
from megatron_adaptor.features_manager.requirements.requirements import RequirementsBasicFeature
from megatron_adaptor.features_manager.transformer_engine.transformer_engine import TransformerEngineBasicFeature
from megatron_adaptor.features_manager.functional.npu_deterministic import NPUDeterministicFeature
from megatron_adaptor.features_manager.functional.profiler_default import ProfilerDefaultFeature
from megatron_adaptor.features_manager.functional.profile import ProfileFeature
from megatron_adaptor.features_manager.functional.npu_datadump import NPUDataDumpFeature
from megatron_adaptor.features_manager.megatron_fix.megatron_fix import MegatronFixFeature
from megatron_adaptor.features_manager.megatron.hybrid_context_parallel import HybridContextParallelFeature
from megatron_adaptor.auto_execute import AutoExecuteFunction


def create_features_list() -> list:
    """Create the built-in MegatronAdaptor feature list."""
    return [
        RequirementsBasicFeature(),
        MegatronBasicFeature(),
        TransformerEngineBasicFeature(),
        MLARopeFeature(),
        NPUDeterministicFeature(),
        ProfilerDefaultFeature(),
        ProfileFeature(),
        NPUDataDumpFeature(),
        MegatronFixFeature(),
        HybridContextParallelFeature(),
    ]


FEATURE_LIST = create_features_list()


@AutoExecuteFunction
def set_default_features_list():
    """Install MA's built-in feature list unless an inheritor suppresses it."""
    FeaturesManager.set_features_list(FEATURE_LIST)


set_default_features_list()

__all__ = ["FEATURE_LIST", "create_features_list", "set_default_features_list"]
