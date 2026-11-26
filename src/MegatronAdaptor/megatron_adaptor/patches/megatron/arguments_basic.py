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
from functools import wraps

from megatron_adaptor.features_manager.features_manager import FeaturesManager


def extra_args_provider_decorator(extra_args_provider):
    """Make a extra args parser for megatron."""

    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        FeaturesManager.register_features_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    """Decorate parse_args function of megatron."""

    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def validate_args_wrapper(validate_args):
    """A decorator for megatron arguments validation function."""

    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        # make prev validation and copy some args.
        FeaturesManager.pre_validate_features_args(args)

        # make megatron args validation then restore args that are copied.
        args = validate_args(args, defaults)

        # make post validation after megatron validation.
        FeaturesManager.post_validate_features_args(args=args)

        FeaturesManager.validate_features_args(args=args)

        return args

    return wrapper
