"""Inventory warning occurrences in a launcher log without modifying any training state."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


PATTERNS = {
    "reduce_scatter_size_warning": r"For reduce_scatter, the total number of input elements",
    "accumulate_grad_stream_mismatch": r"UserWarning: The AccumulateGrad node's stream",
    "missing_hf_mapping_keys": r"WARNING: Can't find .*hf_keys",
    "hf_docstring_errors": r"\[ERROR\].*not documented",
    "bf16_to_fp32_loading": r"Dtype mismatch.*HF dtype: torch.bfloat16. Megatron dtype: torch.float32",
    "fp32_to_bf16_loading": r"Dtype mismatch.*HF dtype: torch.float32. Megatron dtype: torch.bfloat16",
    "ffn_size_fallback": r"DeprecationWarning: MLP requires ffn_hidden_size",
    "watchdog_timeout_order": r"Warning: The watchdog timeout",
    "modelopt_transformers_experimental": r"UserWarning: transformers>=5.0 support is experimental",
    "apex_fallback": r"UserWarning: Apex is not installed",
    "ddp_auto_layout": r"WARNING:.*full_param_layout not provided",
    "triton_api_deprecation": r"FutureWarning: triton.language.extra.cann.extension",
    "collective_api_deprecation": r"FutureWarning:.*torch.distributed._(?:all_gather_base|reduce_scatter_base).*private",
    "swig_deprecation": r"DeprecationWarning: builtin type swigvarlink",
    "runpy_launcher_warning": r"RuntimeWarning: 'torch.distributed.run' found in sys.modules",
    "fla_unavailable": r"^FLA is not available$",
    "hf_fla_before_adaptor_false": r"^HF_FLA_BEFORE_ADAPTOR: False$",
    "driver_padding_notice": r"Warning: The current CANN and HDK\(driver\) versions require processing for 32 padding size",
}


def audit(path):
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    split = next(n for n, line in enumerate(lines, 1) if line.startswith("ADAMW_RUN_START role=measurement "))
    categories = {}
    for name, expression in PATTERNS.items():
        hits = [(n, line) for n, line in enumerate(lines, 1) if re.search(expression, line)]
        categories[name] = {
            "total_log_occurrences": len(hits),
            "measurement_occurrences": sum(n >= split for n, _ in hits),
            "line_numbers": [n for n, _ in hits],
            "first_example": hits[0][1] if hits else None,
        }
    failures = {}
    for name, expression in {
        "traceback": r"Traceback \(most recent call last\)",
        "oom": r"OutOfMemoryError|out of memory",
        "termination_or_device_error": r"ChildFailedError|SIGTERM|SIGKILL|ERR\d{4,}",
        "nonfinite_iteration": r"^ITER .*\b(?:loss|grad_norm|step_s)=(?:nan|[-+]?inf)\b",
    }.items():
        failures[name] = [n for n, line in enumerate(lines, 1) if re.search(expression, line, re.IGNORECASE)]
    missing = [line for line in lines if re.search(PATTERNS["missing_hf_mapping_keys"], line)]
    layer_ids = sorted({int(layer) for line in missing for layer in re.findall(r"layers\.(\d+)", line)})
    docs = collections.Counter()
    for line in lines:
        if "[ERROR]" in line:
            docs[re.sub(r"^.*?\[ERROR\]\s*", "", line)] += 1
    numbers = []
    for n, line in enumerate(lines, 1):
        match = re.search(r"Got total input elements (\d+), output elements (\d+), world size (\d+), and expected total input elements (\d+)", line)
        if match:
            numbers.append({"line": n, "input_numel": int(match[1]), "output_numel": int(match[2]), "world_size": int(match[3]), "expected_numel": int(match[4])})
    return {
        "source_log": str(path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "measurement_starts_at_line": split, "categories": categories, "fatal_patterns": failures,
        "error_level_messages": docs, "missing_hf_key_layers": layer_ids,
        "missing_keys_only_on_expected_gdn_layers": bool(layer_ids) and all(layer % 4 != 3 for layer in layer_ids),
        "reduce_scatter_reported_sizes": numbers,
        "scope": "Warning occurrence counts are not operation counts; warnings may be emitted once per process. Runtime completion and optimizer probes do not establish full gradient/weight equivalence.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"fatal_pattern_counts": {key: len(value) for key, value in result["fatal_patterns"].items()}, "categories": {key: value["measurement_occurrences"] for key, value in result["categories"].items()}}, indent=2))
