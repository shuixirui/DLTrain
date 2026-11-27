"""CPU checks for report integrity and local corpus input parsing, not NPU training."""

from _paths import ROOT

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from plot_interns2_500 import load_run, plot_run
from prepare_interns2_corpus import documents


class MeasurementIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.manifest = {
            "run_id": "TEST_FIXTURE_ONLY", "status": "passed", "requested_steps": 5,
            "tokens_per_step": 512, "lr": 0.001, "timing_warmup_steps_omitted": 4,
            "model_name": "Intern-S2-Preview", "num_layers": 40, "num_experts": 256,
            "world_size": 16, "tp": 2, "pp": 1, "ep": 8, "etp": 2, "cp": 1,
            "dp": 8, "seq_length": 64, "global_batch_size": 8,
            "optimizer": "Megatron SGD", "data_mode": "synthetic",
        }
        # Synthetic fixtures live only in a TemporaryDirectory and are never
        # retained or represented as actual hardware measurements.
        self.rows = [dict(step=i, lm_loss=17.0, grad_norm_before_clip=10.0, step_seconds_max_rank=float(i), lr=0.001, tokens=512, optimizer_update_success=True) for i in range(1, 6)]
        self.summary = {"status": "passed", "completed_steps": 5, "first_step_updates_verified_on_all_ranks": True}
        self.save()

    def save(self):
        (self.run / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.run / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (self.run / "launcher-exit.json").write_text('{"exit_code":0}', encoding="utf-8")
        (self.run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_complete_and_warmup_does_not_remove_loss_points(self):
        _, rows, report = load_run(self.run)
        self.assertEqual(len(rows), 5)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["median_step_seconds"], 5.0)
        self.assertEqual(report["timing_sample_count"], 1)

    def test_missing_steps_cannot_pass_even_with_success_summary(self):
        self.rows.pop()
        self.save()
        self.assertNotEqual(load_run(self.run)[2]["status"], "passed")

    def test_bad_launcher_exit_cannot_pass(self):
        (self.run / "launcher-exit.json").write_text('{"exit_code":1}')
        self.assertNotEqual(load_run(self.run)[2]["status"], "passed")

    def test_duplicate_steps_rejected(self):
        self.rows[1]["step"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "unique"):
            load_run(self.run)

    def test_nan_rejected(self):
        self.rows[0]["grad_norm_before_clip"] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            load_run(self.run)

    def test_truncated_tail_is_partial(self):
        self.rows.pop()
        self.save()
        with (self.run / "metrics.jsonl").open("a") as output:
            output.write('{"step":5')
        with warnings.catch_warnings(record=True):
            report = load_run(self.run)[2]
        self.assertEqual(report["observed_steps"], 4)
        self.assertNotEqual(report["status"], "passed")

    def test_batch_mismatch_rejected(self):
        self.rows[0]["tokens"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_run(self.run)

    def test_exports_only_available_points(self):
        self.rows = self.rows[:1]
        self.save()
        plot_run(self.run, self.run / "plot")
        self.assertTrue((self.run / "plot/report.png").is_file())
        self.assertTrue((self.run / "plot/report.svg").is_file())
        summary = json.loads((self.run / "plot/plot-summary.json").read_text())
        self.assertEqual(summary["observed_steps"], 1)
        self.assertFalse(summary["missing_steps_filled"])
        self.assertNotEqual(summary["status"], "passed")

    def test_jsonl_text_field_and_invalid_record(self):
        source = self.run / "data.jsonl"
        source.write_text('{"body":"actual text"}\n', encoding="utf-8")
        self.assertEqual(list(documents(source, "body")), ["actual text"])
        with self.assertRaises(ValueError):
            list(documents(source, "text"))


if __name__ == "__main__":
    unittest.main()
