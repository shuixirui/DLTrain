"""CPU checks of schedule, AdamW expectations, and scheduled measurement validation."""

from _paths import ROOT

import json
import math
import tempfile
import unittest
from pathlib import Path

from adamw_math import first_step_expectation, scheduled_lr
from plot_interns2_adamw import load_run


class AdamWMathTests(unittest.TestCase):
    def test_schedule_boundaries_and_pilot_prefix(self):
        self.assertAlmostEqual(scheduled_lr(1, 1e-5, 1e-6), 5e-7)
        self.assertAlmostEqual(scheduled_lr(3, 1e-5, 1e-6), 1.5e-6)
        self.assertAlmostEqual(scheduled_lr(20, 1e-5, 1e-6), 1e-5)
        self.assertAlmostEqual(scheduled_lr(500, 1e-5, 1e-6), 1e-6)
        self.assertGreater(scheduled_lr(20, 1e-5, 1e-6), scheduled_lr(21, 1e-5, 1e-6))

    def test_schedule_is_positive_and_decreases_after_warmup(self):
        rates = [scheduled_lr(step, 1e-5, 1e-6) for step in range(1, 501)]
        self.assertTrue(all(math.isfinite(rate) and rate > 0 for rate in rates))
        self.assertTrue(all(left >= right for left, right in zip(rates[19:], rates[20:])))

    def test_invalid_schedule_rejected(self):
        for args in [(0, 1e-5, 1e-6), (501, 1e-5, 1e-6), (1, 1e-6, 1e-5), (1, float("nan"), 1e-6)]:
            with self.assertRaises(ValueError):
                scheduled_lr(*args)

    def test_adamw_bias_correction_clipping_epsilon_and_decay(self):
        for grad in (-2.0, 0.0, 1e-7, 3.0):
            for factor in (1.0, 1e-4):
                for decay in (0.0, 0.01):
                    with self.subTest(grad=grad, clip=factor, decay=decay):
                        weight, lr, eps = 0.125, 5e-7, 1e-8
                        actual, m, v = first_step_expectation(weight, grad, lr, 0.9, 0.999, eps, decay, factor)
                        clipped = grad * factor
                        reference = weight * (1 - lr * decay) - lr * clipped / (abs(clipped) + eps)
                        self.assertAlmostEqual(actual, reference, places=14)
                        self.assertAlmostEqual(m, 0.1 * clipped, places=14)
                        self.assertAlmostEqual(v, 0.001 * clipped**2, places=14)


class ScheduledRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        manifest = {"run_id": "TEST_ONLY", "status": "running", "requested_steps": 500, "tokens_per_step": 512, "lr": 1e-5,
                    "lr_schedule": {"peak_lr": 1e-5, "min_lr": 1e-6, "warmup_steps": 20, "schedule_steps": 500}}
        (self.run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.rows = [dict(step=step, lm_loss=12.0, grad_norm_before_clip=100.0, step_seconds_max_rank=3.0, tokens=512, lr=rate, optimizer_update_success=True)
                     for step, rate in enumerate((5e-7, 1e-6, 1.5e-6), 1)]
        self.save()

    def save(self):
        (self.run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_scheduled_lr_accepted_but_incomplete_run_cannot_pass(self):
        report = load_run(self.run)[2]
        self.assertEqual(report["observed_steps"], 3)
        self.assertNotEqual(report["status"], "passed")

    def test_accidental_old_constant_lr_rejected(self):
        self.rows[0]["lr"] = 1e-3
        self.save()
        with self.assertRaisesRegex(ValueError, "learning rate"):
            load_run(self.run)


if __name__ == "__main__":
    unittest.main()
