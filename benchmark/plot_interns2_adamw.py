"""Plot measured training records; never interpolate missing steps or infer success from loss."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import warnings
from pathlib import Path
from _runtime_path import RUNTIME_DIRECTORY
from adamw_math import scheduled_lr


def load_run(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    requested = int(manifest["requested_steps"])
    if requested <= 0:
        raise ValueError("requested_steps must be positive")
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    fields = ("lm_loss", "grad_norm_before_clip", "step_seconds_max_rank", "lr", "tokens")
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                warnings.warn("Ignoring a truncated final JSONL record; run cannot pass with missing records")
                break
            raise ValueError(f"Invalid metrics JSON at line {index + 1}") from None
        if row.get("step") != len(rows) + 1:
            raise ValueError("Steps must be contiguous, ordered, and unique starting at 1")
        if row["step"] > requested:
            raise ValueError("More measured steps than the requested run length")
        for field in fields:
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]):
                raise ValueError(f"Non-finite/missing {field} at step {row['step']}")
        if row["lm_loss"] < 0 or min(row[field] for field in fields[1:]) <= 0:
            raise ValueError(f"Invalid loss, gradient, duration, learning rate or tokens at step {row['step']}")
        if row.get("optimizer_update_success") is not True:
            raise ValueError(f"Unsuccessful optimizer update at step {row['step']}")
        schedule = manifest["lr_schedule"]
        expected_lr = scheduled_lr(row["step"], schedule["peak_lr"], schedule["min_lr"], schedule["warmup_steps"], schedule["schedule_steps"])
        if row["tokens"] != manifest["tokens_per_step"] or not math.isclose(row["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-16):
            raise ValueError("Per-step batch tokens or scheduled learning rate differs from the recorded manifest")
        rows.append(row)
    if not rows:
        raise ValueError("No complete measured steps found; refusing to fabricate a curve")
    summary_path, exit_path = run / "summary.json", run / "launcher-exit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    launcher = json.loads(exit_path.read_text(encoding="utf-8")) if exit_path.exists() else {}
    passed = (
        len(rows) == requested and manifest.get("status") == "passed"
        and summary.get("status") == "passed" and summary.get("completed_steps") == requested
        and summary.get("first_step_updates_verified_on_all_ranks") is True
        and launcher.get("exit_code") == 0
    )
    status = "passed" if passed else "incomplete / not passed"
    omitted = int(manifest.get("timing_warmup_steps_omitted", 4))
    if omitted < 0:
        raise ValueError("Negative warmup count")
    timings = [row["step_seconds_max_rank"] for row in rows if row["step"] > omitted]
    report = {
        "run_id": manifest["run_id"], "status": status,
        "observed_steps": len(rows), "requested_steps": requested,
        "timing_warmup_steps_omitted": omitted, "timing_sample_count": len(timings),
        "median_step_seconds": statistics.median(timings) if timings else None,
        "initial_lm_loss": rows[0]["lm_loss"], "final_lm_loss": rows[-1]["lm_loss"],
        "max_grad_norm_before_clip": max(row["grad_norm_before_clip"] for row in rows),
        "metrics_sha256": hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "loss_smoothing": "none", "missing_steps_filled": False,
    }
    return manifest, rows, report


def plot_run(run, output_dir):
    manifest, rows, report = load_run(run)
    # Imported only when plotting. Training on the cluster does not need matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 11))
    axes[0].plot(steps, [row["lm_loss"] for row in rows], color="#2768b2", linewidth=1.25)
    axes[0].set_ylabel("Training LM loss")
    axes[1].plot(steps, [row["grad_norm_before_clip"] for row in rows], color="#a85ace", linewidth=1.0)
    axes[1].set_ylabel("Gradient norm (before clipping)")
    omitted = report["timing_warmup_steps_omitted"]
    timed = [row for row in rows if row["step"] > omitted]
    if timed:
        axes[2].plot([row["step"] for row in timed], [row["step_seconds_max_rank"] for row in timed], color="#267c70", linewidth=1.0, label=f"Max rank step time (first {omitted} omitted)")
        median = report["median_step_seconds"]
        axes[2].axhline(median, color="#666666", linestyle="--", linewidth=1, label=f"Median: {median:.3f} s")
        axes[2].legend(loc="upper left", fontsize=10)
    else:
        axes[2].text(0.5, 0.5, "No timing samples after warmup", transform=axes[2].transAxes, ha="center")
    axes[2].set_ylabel("Seconds / optimizer step")
    axes[2].set_xlabel("Optimizer iteration")
    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.tick_params(direction="out")
    if len(steps) == 1:
        axes[2].set_xlim(0, max(2, steps[-1] + 1))
    title = f"{manifest['model_name']} L{manifest['num_layers']} / E{manifest['num_experts']} on Ascend A3 - {len(rows)}/{report['requested_steps']} observed steps"
    layout = f"TP{manifest['tp']} | PP{manifest['pp']} | EP{manifest['ep']} | ETP{manifest['etp']} | CP{manifest['cp']} | DP{manifest['dp']} | SEQ{manifest['seq_length']} | GBS{manifest['global_batch_size']}"
    fig.suptitle(title + "\n" + layout + f"\n{manifest['optimizer']} | peak lr={manifest['lr']:g}, warmup+cosine | BF16 | {manifest['world_size']} logical NPUs", y=0.985, fontsize=13)
    data_label = "real text, native model tokenizer" if manifest["data_mode"] == "text" else "SYNTHETIC RANDOM TOKENS - stability test"
    footer = manifest.get("run_role", "measurement") + "; " + data_label + "; training loss only, no held-out quality evaluation. Status: " + report["status"]
    if manifest["data_mode"] == "text":
        footer += f"\nCorpus tokens: {manifest['corpus']['num_tokens']:,}; EOS-joined documents; repeated cyclically when exhausted."
    fig.text(0.5, 0.022, footer, ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.095, right=0.98, hspace=0.07)
    fig.savefig(output_dir / "report.png", dpi=180)
    fig.savefig(output_dir / "report.svg")
    plt.close(fig)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "plot-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("MEASURED_REPORT:", output_dir / "report.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plot_run(args.run_dir, args.output_dir or args.run_dir)
