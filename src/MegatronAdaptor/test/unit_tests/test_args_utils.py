# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""parser_unknown_args 的选项过滤回归测试。

未知参数只有是合法长选项（`--xxx`）时才转换为配置字段名：短选项（如 pytest 的 `-o`）、
负数、非法标识符与 Python 关键字不得进入结果，且它们携带的值不得泄漏到前一个合法选项。

不依赖 NPU / megatron，直接对接口输入输出断言。
"""

import pytest

from megatron_adaptor.utils.args_utils import parser_unknown_args


@pytest.mark.parametrize(
    ("unknown", "expected"),
    [
        # 短选项：单字符会被旧实现截成空 key，进而生成非法 dataclass 字段
        (["-o", "tmp_path_retention_policy=none"], {}),
        (["-o=foo"], {}),
        # 短选项与其值一起丢弃，不能挂到前一个长选项上
        (["-x", "1", "--real-opt", "2"], {"real_opt": "2"}),
        (["--foo", "bar", "-o", "baz", "--qux", "1"], {"foo": "bar", "qux": "1"}),
        # 正常长选项
        (["--foo-bar=1"], {"foo_bar": "1"}),
        (["--foo=bar=baz"], {"foo": "bar=baz"}),
        (["--flag"], {"flag": True}),
        (["--foo", "--bar"], {"foo": True, "bar": True}),
        (["--multi", "v1", "v2"], {"multi": ["v1", "v2"]}),
        # 非法字段名
        (["--1abc", "x"], {}),
        (["--foo.bar", "v"], {}),
        (["--class", "x"], {}),
        (["--", "x"], {}),
    ],
)
def test_only_valid_long_options_are_kept(unknown, expected):
    assert parser_unknown_args({}, list(unknown)) == expected


def test_existing_known_args_are_preserved():
    args = {"seed": 1}
    assert parser_unknown_args(args, ["-o", "x", "--extra-feature", "on"]) == {
        "seed": 1,
        "extra_feature": "on",
    }
