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

_MEGATRON_TRAINING_AVAILABLE = None


def is_megatron_training_available():
    """
    Check if megatron.training module is available.

    Returns:
        bool: True if megatron.training is available, False otherwise.
    """
    global _MEGATRON_TRAINING_AVAILABLE
    if _MEGATRON_TRAINING_AVAILABLE is not None:
        return _MEGATRON_TRAINING_AVAILABLE

    try:
        import megatron.training  # noqa: F401

        _MEGATRON_TRAINING_AVAILABLE = True
    except ModuleNotFoundError:
        _MEGATRON_TRAINING_AVAILABLE = False

    return _MEGATRON_TRAINING_AVAILABLE
