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
        return subprocess.run([BASH, str(ROOT / "examples/run.sh"), "--dry-run"],
                              cwd=ROOT, env=environment, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)

    def test_preview_uses_raw_data_and_twenty_steps_without_loading_model(self):
        result = self.preview(MODEL_PATH="/model-that-does-not-exist")
        self.assertEqual(result.returncode, 0, result.stderr)
        workdir = next(line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("WORKDIR="))
        self.assertEqual(Path(workdir).name, ROOT.name)
        self.assertIn("--steps 20", result.stdout)
        self.assertIn("--text-field raw_content", result.stdout)
        self.assertIn("data/en_head.json.gz", result.stdout)
        self.assertIn("--nproc-per-node=16", result.stdout)
        self.assertIn("--num-microbatches 4", result.stdout)

    def test_invalid_step_budget_stops_during_preview(self):
        result = self.preview(TRAIN_STEPS="501", SCHEDULE_STEPS="500")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("torch.distributed.run", result.stdout)


@unittest.skipUnless(BASH, "Bash is required for the installer preview")
class InstallerTests(unittest.TestCase):
    def invoke(self, *arguments, **changes):
        environment = dict(os.environ)
        environment.pop("VENV_PATH", None)
        environment.update(changes)
        return subprocess.run([BASH, str(ROOT / "install.sh"), *arguments],
                              cwd=ROOT, env=environment, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)

    def test_preview_needs_no_python_cann_or_model_and_does_not_create_environment(self):
        result = self.invoke("--check-provider", "--dry-run",
                             VENV_PATH=".venv-preview-only-do-not-create",
                             PYTHON_BIN="/missing-python",
                             CANN_ENV_FILE="/missing-cann.sh",
                             MODEL_PATH="/missing-model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/missing-python -I -m venv", result.stdout)
        self.assertIn("nvidia-modelopt==0.44.0", result.stdout)
        self.assertIn("tokenizer-requirements.txt", result.stdout)
        self.assertIn("src/DLTrain-framework", result.stdout)
        self.assertIn("src/DLTrain-NPUEngine", result.stdout)
        self.assertIn("src/DLTrain-Adaptor", result.stdout)
        self.assertIn("check_interns2_provider.py", result.stdout)
        self.assertFalse((ROOT / ".venv-preview-only-do-not-create").exists())

    def test_default_installation_does_not_require_provider_or_start_training(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(".venv-dltrain", result.stdout)
        self.assertNotIn("check_interns2_provider.py", result.stdout)
        self.assertNotIn("torch.distributed.run", result.stdout)

    def test_unknown_argument_stops_before_installation(self):
        result = self.invoke("--not-a-real-option")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("pip install", result.stdout)

    def test_sourcing_does_not_close_the_calling_shell(self):
        result = subprocess.run(
            [BASH, "-c", 'source ./install.sh; code=$?; printf "CALLER_ALIVE:%s\\n" "$code"'],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CALLER_ALIVE:2", result.stdout)


if __name__ == "__main__":
    unittest.main()
