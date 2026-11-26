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

"""Control import-time adaptor execution without importing the adaptor package."""

from collections.abc import Callable

# Process-wide switch. Decorated functions run at import time, which is
# serialized by Python's import lock, so a plain dict is race-free.
_STATE = {"auto_execute": True}


class AutoExecuteFunction:
    """Execute a decorated function only while automatic execution is enabled."""

    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kwargs):
        if not _STATE["auto_execute"]:
            return None
        return self._func(*args, **kwargs)


class NoExecuteFunction:
    """Temporarily suppress functions decorated with ``AutoExecuteFunction``.

    The flag is process-wide; ``__exit__`` restores the value saved on
    ``__enter__``, so nested contexts and user presets are preserved.
    """

    def __init__(self):
        self._prev = True

    def __enter__(self):
        self._prev = _STATE["auto_execute"]
        _STATE["auto_execute"] = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _STATE["auto_execute"] = self._prev


__all__ = ["AutoExecuteFunction", "NoExecuteFunction"]
