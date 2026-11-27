"""Compare two archived runs by optimizer step and by consumed token interval."""

import argparse
import json
import statistics
from pathlib import Path

from plot_interns2_log import load_log


def compare(first, second, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_log(path) for path in (first, second)]
    budgets = [report["printed_metrics_statistics"]["tokens_per_step"] for _, report in runs]
    if sorted(budgets) != [512, 2048]:
        raise ValueError("This comparison requires the SEQ64 GBS8 and GBS32 pair")
    interval = max(budgets)
    if any(interval % budget for budget in budgets):
        raise ValueError("Token intervals must divide the common interval")
    aligned = []
    for (rows, _), budget in zip(runs, budgets):
        width = interval // budget
        aligned.append([
            {"tokens": (start + width) * budget,
             "mean_loss": statistics.mean(row["lm_loss"] for row in rows[start:start + width])}
            for start in range(0, len(rows) - width + 1, width)
        ])
    shared = min(len(items) for items in aligned)
    summary = {
        "runs": [], "common_token_interval": interval,
        "shared_token_budget": shared * interval,
        "matched_budget_last_window_tokens": 25 * interval,
        "caveat": "Training batches only, not a fixed evaluation set. Same steps consume different tokens; same tokens use different update counts and learning rates. No causal batch-size or held-out quality claim.",
    }
    for (rows, report), points in zip(runs, aligned):
        native = report["native_summary"]
        summary["runs"].append({
            "run_id": report["run_id"], "source_sha256": report["source_log_sha256"],
            "global_batch_size": report["configuration_from_log"]["global_batch_size"],
            "total_training_tokens": report["printed_metrics_statistics"]["total_training_tokens"],
            "last_100_steps_mean_loss": statistics.mean(row["lm_loss"] for row in rows[-100:]),
            "last_100_steps_median_grad_norm": statistics.median(row["grad_norm_before_clip"] for row in rows[-100:]),
            "median_step_seconds_after_first_4": native["median_step_seconds_after_warmup"],
            "wall_seconds_excluding_initialization": native["wall_seconds_excluding_initialization"],
            "tokens_per_second_over_recorded_wall": report["printed_metrics_statistics"]["total_training_tokens"] / native["wall_seconds_excluding_initialization"],
            "matched_budget_last_window_mean_loss": statistics.mean(point["mean_loss"] for point in points[shared-25:shared]),
            "matched_budget_final_step": shared * interval // report["printed_metrics_statistics"]["tokens_per_step"],
            "peak_allocated_gib_max_rank": native["peak_allocated_gib_max_rank"],
        })
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
    for (rows, report), points, color in zip(runs, aligned, ("#296bb5", "#c65c16")):
        label = f"GBS{report['configuration_from_log']['global_batch_size']}"
        losses = [row["lm_loss"] for row in rows]
        axes[0].plot([row["step"] for row in rows], losses, color=color, alpha=0.18, linewidth=0.7)
        axes[0].plot([row["step"] for row in rows[19:]],
                     [statistics.mean(losses[i-19:i+1]) for i in range(19, len(losses))],
                     color=color, label=label + " (trailing 20-step mean)")
        points = points[:shared]
        axes[1].plot([point["tokens"] / 1000 for point in points],
                     [point["mean_loss"] for point in points], color=color, alpha=0.2, linewidth=0.8)
        axes[1].plot([point["tokens"] / 1000 for point in points[24:]],
                     [statistics.mean(point["mean_loss"] for point in points[i-24:i+1]) for i in range(24, len(points))],
                     color=color, label=label + " (trailing 51,200-token mean)")
    axes[0].set_xlabel("Optimizer iteration (different token budgets)")
    axes[1].set_xlabel("Consumed target tokens (thousands; shared first 256,000)")
    for axis in axes:
        axis.set_ylabel("Training LM loss")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=9)
    fig.suptitle("Intern-S2-Preview on Ascend A3: GBS8 vs GBS32\nSEQ64 | DP8 | native distributed AdamW | 500 steps each", fontsize=13)
    fig.text(0.5, 0.02,
             "Faint lines: raw step loss (top) / mean per 2,048-token interval (bottom).\n"
             "Same tokens use different update counts and LR schedules; batches are not a held-out evaluation.\n"
             "Numerical validation pending; no interpolation or missing-step filling.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    compare(args.first, args.second, args.output_dir)
