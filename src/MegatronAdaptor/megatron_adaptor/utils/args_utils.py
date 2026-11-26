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

import argparse
import keyword
from contextvars import ContextVar
from logging import getLogger

LOG = getLogger(__name__)
_full_args: ContextVar = ContextVar('full_args', default=None)
_MINDSPEED_ARGS = None


def add_args(args, key, value):
    """Store a `--long-option` in `args` as a field name (`--foo-bar` -> `foo_bar`).

    Names that are not valid identifiers (empty from bare `--`, `--1abc`, `--foo.bar`)
    or Python keywords (`--class`) are dropped: injecting them into TransformerConfig
    breaks dataclass generation.
    """
    if key is not None:
        name = key[2:].replace('-', '_')
        if not name.isidentifier() or keyword.iskeyword(name):
            LOG.debug('Ignore command line option that is not a valid field name: %s', key)
            return
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        args[name] = value


def parser_unknown_args(args, unknown):
    """Convert unknown CLI tokens into `args` entries.

    Handles only long options, e.g. `['--foo', 'bar', '--baz=1'] -> {'foo': 'bar', 'baz': '1'}`.
    Single-`-` tokens (short options like pytest's `-o`, negative numbers) are dropped
    together with their values, e.g. `['--foo', 'bar', '-o', 'baz', '--qux', '1']`
    -> `{'foo': 'bar', 'qux': '1'}`: `baz` belongs to `-o` and must not leak into `--foo`.
    """
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith('--'):
            # New long option: commit the previous one first.
            add_args(args, key, value)
            splits = unknown[i].split('=', maxsplit=1)
            key, value = (unknown[i], None) if len(splits) == 1 else (splits[0], [splits[1]])
        elif unknown[i].startswith('-'):
            # Short option / negative number names no field: commit the previous
            # option, then drop the pending state so its values are discarded.
            add_args(args, key, value)
            key = value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)
    return args


def get_mindspeed_args(get_defaults=False):
    global _MINDSPEED_ARGS
    if not get_defaults and _MINDSPEED_ARGS is not None:
        return _MINDSPEED_ARGS

    parser = argparse.ArgumentParser(description='MegatronAdaptor Arguments', allow_abbrev=False)
    from megatron_adaptor.features_manager.features_manager import FeaturesManager

    FeaturesManager.register_features_args(parser)
    args, unknown = parser.parse_known_args() if not get_defaults else parser.parse_known_args([])
    # Keep valid long options from `unknown` (arguments of features registered later land here).
    args = parser_unknown_args(vars(args), unknown)
    args = argparse.Namespace(**args)
    if not get_defaults:
        _MINDSPEED_ARGS = args
    return args


def reset_mindspeed_args():
    """Invalidate the cached adaptor args after the feature list changes.

    The cached args are parsed from the feature list at parse time; once the
    list changes (e.g. MindSpeed features are appended), a stale cache would
    miss newly registered feature arguments, so it must be re-parsed.
    """
    global _MINDSPEED_ARGS
    _MINDSPEED_ARGS = None


def get_full_args() -> argparse.Namespace:
    args = _full_args.get()
    if args is not None:
        return args
    try:
        from megatron.training.global_vars import get_args

        full_args = get_args()
        if full_args is None:
            full_args = get_mindspeed_args()
    except ImportError:
        LOG.warning('Failed from megatron.training import get_args, use megatron_adaptor arguments.')
        full_args = get_mindspeed_args()
    except AssertionError as e:
        if 'args is not initialized.' in str(e):
            full_args = get_mindspeed_args()
        else:
            raise e
    return full_args


def set_full_args(args: argparse.Namespace):
    _full_args.set(args)
