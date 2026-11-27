# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests for the ``safe_unpickler_find_class_patch``.

Upstream ``megatron.core.safe_globals.SafeUnpickler`` allowlists only the
NVIDIA TE module layout, so classes pickled by TE-NPU into FP8 extra-state
(NPU float8 dtypes, recipe metadata) are rejected on checkpoint load. The
patch extends the allowlist while keeping the restricted-unpickler semantics.

``pickle.Unpickler`` is the C-implemented ``_pickle.Unpickler`` whose
``find_class`` cannot be mocked, so allowed entries are verified through real
class resolution (torch / in-test stub modules) and the deny path through the
raised ``pickle.UnpicklingError``.

Run:
    pytest test/unit_tests/patches/transformer_engine/test_safe_unpickler.py -v -s
"""

import importlib
import io
import pickle
import sys
import types

import pytest

import megatron_adaptor  # noqa: F401
from megatron_adaptor.patches.transformer_engine.transformer_engine import (
    _EXTRA_SAFE_CLASSES,
    safe_unpickler_find_class_patch,
)


class _FakeSafeUnpickler(pickle.Unpickler):
    """Stand-in exposing only what the patch relies on: ``_SAFE_CLASSES``."""

    _SAFE_CLASSES = frozenset({("collections", "OrderedDict")})


class TestExtraSafeClasses:
    @pytest.mark.parametrize("module,name", sorted(_EXTRA_SAFE_CLASSES))
    def test_extra_safe_class_resolved(self, module, name):
        """The patch must resolve every extended allowlist entry.

        Some dtype class names (e.g. ``torch.Float8_e4m2``) only exist on
        newer torch; when the host env lacks the attribute, inject a
        temporary sentinel so the resolution branch stays verifiable.
        """
        mod = importlib.import_module(module)
        existed = hasattr(mod, name)
        if not existed:
            setattr(mod, name, _FakeSafeUnpickler)
        try:
            unpickler = _FakeSafeUnpickler(io.BytesIO(b""))
            assert safe_unpickler_find_class_patch(unpickler, module, name) is not None
        finally:
            if not existed:
                delattr(mod, name)


class TestRecipeModulePrefix:
    def test_any_class_under_recipe_modules_resolved(self):
        """Allowlist also covers any class under ``transformer_engine.common.recipe``."""
        unpickler = _FakeSafeUnpickler(io.BytesIO(b""))
        leaf_name = "transformer_engine.common.recipe"
        attr = "_ma_test_sentinel_cls"
        created = []
        try:
            parts = leaf_name.split(".")
            for i in range(1, len(parts) + 1):
                name = ".".join(parts[:i])
                if name not in sys.modules:
                    mod = types.ModuleType(name)
                    sys.modules[name] = mod
                    created.append(name)
                    parent = sys.modules.get(".".join(parts[: i - 1]))
                    if parent is not None:
                        setattr(parent, parts[i - 1], mod)
            leaf = sys.modules[leaf_name]
            existed = hasattr(leaf, attr)
            setattr(leaf, attr, _FakeSafeUnpickler)
            assert safe_unpickler_find_class_patch(unpickler, leaf_name, attr) is _FakeSafeUnpickler
            if not existed:
                delattr(leaf, attr)
        finally:
            for name in reversed(created):
                mod = sys.modules.pop(name, None)
                parent_name, _, child = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, child, None) is mod:
                    delattr(parent, child)


class TestDisallowedClass:
    def test_disallowed_class_rejected(self):
        unpickler = _FakeSafeUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError, match="os.system"):
            safe_unpickler_find_class_patch(unpickler, "os", "system")
