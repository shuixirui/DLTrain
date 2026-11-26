"""Run the actual shell coordinator with fake training commands; never launches hardware jobs."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


BASH = shutil.which("bash") or ("D:/AppGallery/Git/bin/bash.exe" if Path("D:/AppGallery/Git/bin/bash.exe").is_file() else None)


@unittest.skipUnless(BASH, "Bash not available")
class CoordinatorTests(unittest.TestCase):
    script_name = "run_interns2_adamw.sh"

    def run_case(self, mode):
        original = Path(__file__).with_name(self.script_name).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix=".adamw-test-", dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/activate").write_text("")
            (root / self.script_name).write_text(original, encoding="utf-8")
            workdir = root.as_posix()
            candidate = original.replace("export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018", "export WORKDIR=" + shlex.quote(workdir))
            candidate = candidate.replace("source /usr/local/Ascend/cann/set_env.sh", ": # environment mocked for CPU-only coordinator test")
            candidate = candidate.replace("npu-smi info", "true")
            harness = r'''
python() {
    if [[ "$1" = "-m" && "$2" = "torch.distributed.run" ]]; then
        role=""
        while [[ $# -gt 0 ]]; do
            if [[ "$1" = "--run-role" ]]; then role="$2"; fi
            shift
        done
        printf '%s\n' "$role" >> "$WORKDIR/calls.txt"
        if [[ "$role" = validation && "$TEST_MODE" = training_failure ]]; then return 9; fi
    elif [[ "$1" = "-" ]]; then
        body=$(cat)
        if [[ "$body" = *ADAMW_SHORT_CHECK_PASSED* && "$TEST_MODE" = summary_failure ]]; then return 8; fi
    fi
    return 0
}
'''
            script = root / "test.sh"
            script.write_text(harness + "\n" + candidate, encoding="utf-8", newline="\n")
            env = dict(os.environ, TEST_MODE=mode, ADAMW_JOB_ID="TEST_ONLY")
            result = subprocess.run([BASH, script.as_posix()], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=25)
            calls = (root / "calls.txt").read_text().splitlines() if (root / "calls.txt").exists() else []
            return result.returncode, calls, result.stdout + result.stderr

    def test_validation_training_failure_stops_measurement(self):
        code, calls, output = self.run_case("training_failure")
        self.assertEqual(code, 9, output)
        self.assertEqual(calls, ["validation"])
        self.assertNotIn("ADAMW_500_PIPELINE_OK", output)

    def test_validation_summary_failure_stops_measurement(self):
        code, calls, output = self.run_case("summary_failure")
        self.assertEqual(code, 8, output)
        self.assertEqual(calls, ["validation"])

    def test_passing_validation_launches_measurement(self):
        code, calls, output = self.run_case("pass")
        self.assertEqual(code, 0, output)
        self.assertEqual(calls, ["validation", "measurement"])
        self.assertIn("ADAMW_500_PIPELINE_OK", output)


class DistributedCoordinatorTests(CoordinatorTests):
    script_name = "run_interns2_adamw_dist.sh"


class GBS32CoordinatorTests(CoordinatorTests):
    script_name = "run_interns2_adamw_gbs32.sh"


if __name__ == "__main__":
    unittest.main()
