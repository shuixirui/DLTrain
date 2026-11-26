"""Validate the completed AdamW launcher log and plot its 500 measured iterations."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from adamw_math import scheduled_lr


ITERATION = re.compile(
    r"^ITER (\d+)/(\d+) loss=(\S+) grad_norm=(\S+) lr=(\S+) "
    r"step_s=(\S+) tokens_s=(\S+) peak_GiB=(\S+)\s*$", re.MULTILINE,
)


def one_json(text, marker):
    values = re.findall(r"^" + re.escape(marker) + r" (.+)$", text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"Expected one {marker}, found {len(values)}")
    return json.loads(values[0])


def load_log(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    launches = list(re.finditer(r"^ADAMW_RUN_START role=measurement steps=(\d+) directory=(\S+)\s*$", text, re.MULTILINE))
    if len(launches) != 1 or int(launches[0][1]) != 500:
        raise ValueError("Expected one separate 500-step measurement launch")
    launch = launches[0]
    section = text[launch.end():]
    if re.search(r"^ADAMW_RUN_START ", section, re.MULTILINE):
        raise ValueError("Unexpected extra training run in measurement section")
    summary = one_json(section, "INTERNS2_MEASURED_TRAINING_OK:")
    measurement = one_json(section, "MEASUREMENT_RUN:")
    if not (
        summary.get("status") == "passed" and summary.get("run_role") == "measurement"
        and summary.get("completed_steps") == summary.get("requested_steps") == 500
        and summary.get("first_step_updates_verified_on_all_ranks") is True
        and summary.get("run_id") == launch[2].rsplit("/", 1)[-1] == measurement.get("run_id")
        and measurement.get("requested_steps") == 500
    ):
        raise ValueError("Completion summary, role, run ID or all-rank update check mismatch")
    exits = re.findall(r"^ADAMW_PIPELINE_EXIT_CODE=(\d+)\s*$", section, re.MULTILINE)
    if exits != ["0"] or not re.search(r"^ADAMW_500_PIPELINE_OK\s*$", section, re.MULTILINE):
        raise ValueError("Missing successful pipeline completion / exit code")
    if re.search(r"Traceback \(most recent call last\)|OutOfMemoryError|ChildFailedError", section):
        raise ValueError("Failure traceback found in formal training section")
    rows = []
    fields = ("lm_loss", "grad_norm_before_clip", "lr", "step_seconds_max_rank", "tokens_per_second", "peak_allocated_gib_max_rank")
    for match in ITERATION.finditer(section):
        if int(match[2]) != 500:
            raise ValueError("Validation iteration mixed into formal run")
        row = {"step": int(match[1]), **dict(zip(fields, map(float, match.groups()[2:])))}
        if row["step"] != len(rows) + 1:
            raise ValueError("Missing, repeated or unordered measurement step")
        if any(not math.isfinite(row[key]) for key in fields) or row["lm_loss"] < 0 or any(row[key] <= 0 for key in fields[1:]):
            raise ValueError("Invalid/non-finite measurement")
        # The launcher prints lr using .8g; validate at that precision.
        expected_lr = float(f"{scheduled_lr(row['step'], 1e-5, 1e-6, 20, 500):.8g}")
        if row["lr"] != expected_lr:
            raise ValueError("Logged learning rate differs from the recorded launcher schedule")
        rows.append(row)
    if len(rows) != 500 or len(re.findall(r"^ITER ", section, re.MULTILINE)) != 500:
        raise ValueError(f"Expected 500 complete iteration records, found {len(rows)}")
    for actual, expected, tolerance in (
        (rows[0]["lm_loss"], summary["initial_lm_loss"], 0.00000051),
        (rows[-1]["lm_loss"], summary["final_lm_loss"], 0.00000051),
        (min(row["lm_loss"] for row in rows), summary["minimum_lm_loss"], 0.00000051),
        (max(row["grad_norm_before_clip"] for row in rows), summary["maximum_grad_norm_before_clip"], 0.00000051),
        (statistics.median(row["step_seconds_max_rank"] for row in rows[4:]), summary["median_step_seconds_after_warmup"], 0.00051),
        (max(row["peak_allocated_gib_max_rank"] for row in rows), summary["peak_allocated_gib_max_rank"], 0.0051),
    ):
        if not math.isfinite(expected) or abs(actual - expected) > tolerance:
            raise ValueError("Printed metrics disagree with native summary beyond rounding tolerance")
    config = {}
    if summary.get("timing_samples_after_warmup") != 496:
        raise ValueError("Unexpected timing sample count")
    for field in ("num_layers", "num_moe_experts", "tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size", "context_parallel_size"):
        values = set(map(int, re.findall(r'"' + field + r'"\s*:\s*(\d+)', section)))
        if len(values) != 1:
            raise ValueError(f"Missing or inconsistent provider field: {field}")
        config[field] = values.pop()
    layouts = [json.loads(value) for value in re.findall(r'\{"rank":\s*\d+,\s*"use_distributed_optimizer":\s*true[^\n]*?\}', section)]
    if sorted(item["rank"] for item in layouts) != list(range(16)):
        raise ValueError("Expected optimizer layout records for all 16 ranks")
    if any(item["dense_dp_size"] != 8 or item["expert_dp_size"] != 1 or item["optimizer_types"] != ["DistributedOptimizer"] for item in layouts):
        raise ValueError("Optimizer layout differs from the measured configuration")
    if measurement["data_mode"] != "text" or measurement["seq_length"] != 64 or measurement["global_batch_size"] not in (8, 32):
        raise ValueError("Unexpected measurement batch/data configuration")
    tokens_per_step = measurement["seq_length"] * measurement["global_batch_size"]
    for row in rows:
        # Account for independently rounded step_s (.3f) and tokens_s (.1f).
        duration = row["step_seconds_max_rank"]
        throughput = row["tokens_per_second"]
        tolerance = 0.0005 * throughput + 0.05 * duration + 0.000025
        if abs(duration * throughput - tokens_per_step) > tolerance + 1e-8:
            raise ValueError("Logged throughput disagrees with batch token count")
    expected_parallelism = {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 8, "expert_tensor_parallel_size": 2, "context_parallel_size": 1}
    if any(config[key] != value for key, value in expected_parallelism.items()):
        raise ValueError("Unexpected provider parallelism")
    blocks = []
    for start in range(0, 500, 100):
        block = rows[start:start + 100]
        blocks.append({"first_step": start + 1, "last_step": start + 100,
                       "mean_loss": statistics.mean(row["lm_loss"] for row in block),
                       "median_grad_norm_before_clip": statistics.median(row["grad_norm_before_clip"] for row in block)})
    report = {
        "run_id": summary["run_id"], "status": "passed", "observed_steps": 500,
        "source_log": str(path.resolve()), "source_log_sha256": hashlib.sha256(raw).hexdigest(),
        "source_format": "launcher stdout; per-step values have printed precision, not original metrics.jsonl precision",
        "completion_status_meaning": "passed means runtime completion and sampled first-step updates; full collective and weight equivalence remain unverified",
        "numerical_equivalence_verified": False,
        "validation_steps_excluded": len(re.findall(r"^ITER \d+/3 ", text[:launch.start()], re.MULTILINE)),
        "configuration_from_log": {**measurement, **config, "logical_npus": 16, "dense_dp_size": 8, "expert_dp_size": 1},
        "launcher_context": {"corpus": "RedPajama V2 English subset", "tokenizer": "native Intern-S2 tokenizer", "peak_lr": 1e-5, "min_lr": 1e-6, "warmup_steps": 20},
        "native_summary": summary,
        "printed_metrics_statistics": {"blocks_of_100": blocks,
                                       "first_grad_norm": rows[0]["grad_norm_before_clip"],
                                       "last_grad_norm": rows[-1]["grad_norm_before_clip"],
                                       "median_step_seconds_after_first_4": statistics.median(row["step_seconds_max_rank"] for row in rows[4:]),
                                       "tokens_per_step": tokens_per_step, "total_training_tokens": tokens_per_step * len(rows)},
        "plot": {"loss": "all raw points plus explicitly labeled trailing 20-step mean", "gradient_axis": "logarithmic; raw norms before clipping", "timing_warmup_steps_omitted": 4, "missing_steps_filled": False},
        "scope": "training text loss only; no held-out evaluation; no checkpoint saved; no HF/Megatron equivalence claim",
    }
    return rows, report


def render(rows, report, output, last_step=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

    if last_step is not None:
        if not 5 <= last_step <= len(rows):
            raise ValueError("Last plotted step must be between 5 and the source run length")
        rows = rows[:last_step]
    median = (report["native_summary"]["median_step_seconds_after_warmup"]
              if len(rows) == report["observed_steps"]
              else statistics.median(row["step_seconds_max_rank"] for row in rows[4:]))
    plotted_tokens = report["printed_metrics_statistics"]["tokens_per_step"] * len(rows)
    peak = (report["native_summary"]["peak_allocated_gib_max_rank"]
            if len(rows) == report["observed_steps"]
            else max(row["peak_allocated_gib_max_rank"] for row in rows))
    report = {**report, "plot": {**report["plot"], "first_step": 1,
              "last_step": rows[-1]["step"], "plotted_steps": len(rows),
              "plotted_training_tokens": plotted_tokens,
              "timing_samples_after_warmup": len(rows) - 4,
              "median_step_seconds_after_first_4": median}}
    output.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]
    losses = [row["lm_loss"] for row in rows]
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": True})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(steps, losses, color="#296bb5", linewidth=0.9, alpha=0.60, label="Raw training LM loss")
    rolling = [statistics.mean(losses[index - 19:index + 1]) for index in range(19, len(losses))]
    axes[0].plot(steps[19:], rolling, color="#cd671b", linewidth=1.6, label="Trailing 20-step mean")
    axes[0].set_ylabel("Training LM loss")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].plot(steps, [row["grad_norm_before_clip"] for row in rows], color="#a455ca", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(LogLocator(base=10))
    axes[1].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylabel("Gradient norm before clipping\n(log scale)")
    timed = rows[4:]
    axes[2].plot([row["step"] for row in timed], [row["step_seconds_max_rank"] for row in timed], color="#237e71", linewidth=1, label="Max rank step time (first 4 omitted)")
    axes[2].axhline(median, color="#666666", linestyle="--", linewidth=1, label=f"Median: {median:.3f} s")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_ylabel("Seconds / optimizer step")
    axes[2].set_xlabel("Optimizer iteration")
    axes[2].set_xlim(0, rows[-1]["step"] * 1.01)
    for axis in axes:
        axis.grid(True, alpha=0.18)
    config = report["configuration_from_log"]
    step_label = (f"{len(rows)}/{report['observed_steps']} measured steps"
                  if len(rows) == report["observed_steps"]
                  else f"first {len(rows)} steps of a {report['observed_steps']}-step run")
    fig.suptitle(
        f"Intern-S2-Preview L{config['num_layers']} / E{config['num_moe_experts']} on Ascend A3 - {step_label}\n"
        f"16 logical NPUs | TP2 | PP1 | EP8 | ETP2 | CP1 | DP8 | SEQ{config['seq_length']} | GBS{config['global_batch_size']}\n"
        f"Distributed AdamW | BF16 | peak lr 1e-5, warmup 20 + cosine | peak allocated {peak:.2f} GiB",
        fontsize=13, y=0.985,
    )
    fig.text(0.5, 0.024,
             f"RedPajama V2 English subset; native model tokenizer; {plotted_tokens:,} training tokens in plotted interval.\n"
             "Training loss only; no held-out evaluation; no checkpoint saved. Source run completed; numerical validation pending.\n"
             "Source: launcher log (printed precision); validation run excluded; no missing steps filled.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.875, bottom=0.12, hspace=0.08)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"report.{suffix}", dpi=180)
    plt.close(fig)
    with (output / "metrics-from-log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "log-analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with ZipFile(output.with_suffix(".zip"), "w", ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                bundle.write(path, arcname=f"{output.name}/{path.name}")
        bundle.write(report["source_log"], arcname=f"{output.name}/source.log")
        bundle.write(__file__, arcname=f"{output.name}/plot_interns2_log.py")
        bundle.write(Path(__file__).with_name("adamw_math.py"), arcname=f"{output.name}/adamw_math.py")
        auditor = Path(__file__).with_name("audit_interns2_warnings.py")
        if auditor.exists():
            bundle.write(auditor, arcname=f"{output.name}/audit_interns2_warnings.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-step", type=int, help="Plot only the first N steps of the validated full run")
    args = parser.parse_args()
    records, analysis = load_log(args.log)
    render(records, analysis, args.output_dir, args.last_step)
    print(json.dumps({"run_id": analysis["run_id"], "status": analysis["status"], "source_steps": len(records), "plotted_steps": args.last_step or len(records), "outputs": str(args.output_dir.resolve())}, indent=2))
