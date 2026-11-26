import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prepare_workspace", ROOT / "tools/prepare_workspace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkspacePreparationTests(unittest.TestCase):
    def fixture(self, root):
        (root / "cluster").mkdir()
        (root / "requirements").mkdir()
        (root / "cluster/train.py").write_bytes(b"original runtime\n")
        (root / "requirements/runtime.txt").write_bytes(b"pinned dependency\n")
        (root / "sources.lock.json").write_text(json.dumps({"components": [
            {"directory": "src/component", "commit": "a" * 40}]}))

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


if __name__ == "__main__":
    unittest.main()
