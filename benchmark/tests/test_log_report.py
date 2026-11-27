"""Exercise log integrity checks with transient corruptions of the delivered real log."""

from _paths import ROOT

import tempfile
import unittest
from pathlib import Path

from plot_interns2_log import load_log


SOURCE = ROOT / "interns2-redpajama-adamw-dist-20261126T134623Z-380481.log"
GBS32_SOURCE = SOURCE.with_name("interns2-redpajama-adamw-gbs32-20261126T150752Z-412139.log")


@unittest.skipUnless(SOURCE.exists(), "Real delivered log not present")
class LogIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")

    def reject(self, text):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "altered.log"
            log.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_log(log)

    def test_complete_run_excludes_validation(self):
        rows, report = load_log(SOURCE)
        self.assertEqual([row["step"] for row in rows], list(range(1, 501)))
        self.assertEqual(report["validation_steps_excluded"], 3)
        self.assertEqual(report["status"], "passed")
        self.assertAlmostEqual(rows[-1]["lm_loss"], 4.045388)

    def test_missing_formal_step_rejected(self):
        lines = self.text.splitlines(keepends=True)
        self.reject("".join(line for line in lines if not line.startswith("ITER 250/500 ")))

    def test_duplicate_formal_step_rejected(self):
        self.reject(self.text.replace("ITER 250/500 ", "ITER 249/500 "))

    def test_success_without_zero_exit_rejected(self):
        self.reject(self.text.replace("ADAMW_PIPELINE_EXIT_CODE=0", "ADAMW_PIPELINE_EXIT_CODE=1"))

    def test_invalid_scheduled_learning_rate_rejected(self):
        line = next(line for line in self.text.splitlines() if line.startswith("ITER 250/500 "))
        fields = line.split()
        fields[4] = "lr=0.001"
        self.reject(self.text.replace(line, " ".join(fields)))

    def test_native_summary_disagreement_rejected(self):
        self.reject(self.text.replace('"final_lm_loss": 4.0453877449035645', '"final_lm_loss": 14.0453877449035645'))

    def test_nonfinite_measurement_rejected(self):
        self.reject(self.text.replace("ITER 500/500 loss=4.045388", "ITER 500/500 loss=nan"))

    def test_wrong_provider_topology_rejected(self):
        self.reject(self.text.replace('"tensor_model_parallel_size": 2', '"tensor_model_parallel_size": 4'))

    def test_batch_mislabeled_as_gbs32_rejected_by_throughput(self):
        self.reject(self.text.replace('"global_batch_size": 8', '"global_batch_size": 32'))

    @unittest.skipUnless(GBS32_SOURCE.exists(), "GBS32 delivered log not present")
    def test_actual_gbs32_run_has_four_times_token_budget(self):
        rows, report = load_log(GBS32_SOURCE)
        self.assertEqual(len(rows), 500)
        self.assertEqual(report["validation_steps_excluded"], 3)
        self.assertEqual(report["configuration_from_log"]["global_batch_size"], 32)
        self.assertEqual(report["printed_metrics_statistics"]["tokens_per_step"], 2048)
        self.assertEqual(report["printed_metrics_statistics"]["total_training_tokens"], 1024000)
        self.assertAlmostEqual(rows[-1]["lm_loss"], 5.470520)


if __name__ == "__main__":
    unittest.main()
