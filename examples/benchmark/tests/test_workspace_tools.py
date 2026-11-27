
from _paths import ROOT
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("prepare_workspace", ROOT / "train/environment/prepare_workspace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkspacePreparationTests(unittest.TestCase):
    def fixture(self, root):
        (root / "train/runtime").mkdir(parents=True)
        (root / "train/environment/requirements").mkdir(parents=True)
        (root / "train/runtime/train.py").write_bytes(b"original runtime\n")
        (root / "train/environment/requirements/runtime.txt").write_bytes(b"pinned dependency\n")
        (root / "train/environment/component_versions.py").write_text(
            "COMPONENTS = " + repr([{"directory": "src/component", "commit": "a" * 40}]) + "\n")

    def test_conflicting_runtime_preserved_and_nothing_else_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "train.py").write_bytes(b"existing user runtime\n")
            with self.assertRaises(RuntimeError):
                module.prepare(root)
            self.assertEqual((root / "train.py").read_bytes(), b"existing user runtime\n")
            self.assertFalse((root / "runtime.txt").exists())
            self.assertFalse((root / "src").exists())

    def test_dry_run_then_repeated_prepare_preserves_files_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            module.prepare(root, dry_run=True)
            self.assertFalse((root / "train.py").exists())
            module.prepare(root)
            self.assertEqual((root / "src/component/source-commit.txt").read_text(), "a" * 40 + "\n")
            previous = (root / "train.py").stat().st_mtime_ns
            module.prepare(root)
            self.assertEqual((root / "train.py").stat().st_mtime_ns, previous)
            self.assertEqual((root / "train.py").read_bytes(), b"original runtime\n")

    def test_missing_component_entry_point_stops_before_runtime_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "train/environment/component_versions.py").write_text(
                "COMPONENTS = " + repr([{"directory": "src/missing", "commit": "a" * 40,
                                         "entry_point": "package/__init__.py"}]) + "\n")
            with self.assertRaises(FileNotFoundError):
                module.prepare(root)
            self.assertFalse((root / "train.py").exists())


if __name__ == "__main__":
    unittest.main()
