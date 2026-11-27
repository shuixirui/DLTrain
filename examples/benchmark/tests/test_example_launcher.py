from _paths import ROOT

import os
from pathlib import Path
import shutil
import subprocess
import unittest


BASH = shutil.which("bash") or ("D:/AppGallery/Git/bin/bash.exe" if Path("D:/AppGallery/Git/bin/bash.exe").is_file() else None)


@unittest.skipUnless(BASH, "Bash is required for the launch preview")
class ExampleLauncherTests(unittest.TestCase):
    def preview(self, **changes):
        environment = dict(os.environ)
        for name in ("TRAIN_STEPS", "SCHEDULE_STEPS", "SEQ_LENGTH", "NUM_MICROBATCHES", "RUN_ID"):
            environment.pop(name, None)
        environment.update(changes)
        return subprocess.run([BASH, str(ROOT / "examples/interns2/run.sh"), "--dry-run"],
                              cwd=ROOT, env=environment, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)

    def test_preview_uses_raw_data_and_twenty_steps_without_loading_model(self):
        result = self.preview(MODEL_PATH="/model-that-does-not-exist")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--steps 20", result.stdout)
        self.assertIn("--text-field raw_content", result.stdout)
        self.assertIn("data/en_head.json.gz", result.stdout)
        self.assertIn("--nproc-per-node=16", result.stdout)
        self.assertIn("--num-microbatches 4", result.stdout)

    def test_invalid_step_budget_stops_during_preview(self):
        result = self.preview(TRAIN_STEPS="501", SCHEDULE_STEPS="500")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("torch.distributed.run", result.stdout)


if __name__ == "__main__":
    unittest.main()
