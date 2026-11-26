"""Correct full-weight orientation checks before the existing ETP scatter."""

import ast
import difflib
import textwrap
from pathlib import Path

METHODS = {
    "ExpertMLPGateUpProjMapping": '''
def hf_to_megatron(self, hf_weights, megatron_module):
    if self.tp_rank != 0:
        return self._gated_mapping.hf_to_megatron(None, megatron_module)
    if hf_weights is None:
        raise ValueError("Missing HF expert gate/up weights on ETP rank 0")
    expert_number = extract_expert_number_from_param(self.megatron_param)
    expert_weight = hf_weights[expert_number] if hf_weights.ndim >= 3 else hf_weights
    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
    local_shape = tuple(target_param.shape)
    if len(local_shape) != 2 or local_shape[0] % 2:
        raise ValueError(f"Invalid local fused gate/up shape: {local_shape}")
    full_shape = (local_shape[0] * self.tp_size, local_shape[1])
    gate_shape = (full_shape[0] // 2, full_shape[1])
    if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
        gate = _align_weight_to_shape(expert_weight[0], gate_shape, "gate")
        up = _align_weight_to_shape(expert_weight[1], gate_shape, "up")
    else:
        expert_weight = _align_weight_to_shape(expert_weight, full_shape, "gate_up")
        gate, up = torch.chunk(expert_weight, 2, dim=0)
    gate = gate.to(dtype=target_param.dtype)
    up = up.to(dtype=target_param.dtype)
    return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)
''',
    "ExpertMLPDownProjMapping": '''
def hf_to_megatron(self, hf_weights, megatron_module):
    if self.tp_rank != 0:
        return super().hf_to_megatron(None, megatron_module)
    if hf_weights is None:
        raise ValueError("Missing HF expert down-projection weights on ETP rank 0")
    expert_number = extract_expert_number_from_param(self.megatron_param)
    expert_weight = hf_weights[expert_number] if hf_weights.ndim >= 3 else hf_weights
    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
    local_shape = tuple(target_param.shape)
    if len(local_shape) != 2:
        raise ValueError(f"Invalid local down-projection shape: {local_shape}")
    full_shape = (local_shape[0], local_shape[1] * self.tp_size)
    expert_weight = _align_weight_to_shape(expert_weight, full_shape, "down_proj")
    expert_weight = expert_weight.to(dtype=target_param.dtype)
    return super().hf_to_megatron(expert_weight, megatron_module)
''',
}


def patch(path):
    text = path.read_text(encoding="utf-8")
    marker = "# expert-etp-full-shape-before-scatter"
    if marker in text:
        print("EXPERT_ETP_PATCH_ALREADY_PRESENT")
        return
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in METHODS:
            continue
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "hf_to_megatron"]
        assert len(methods) == 1, node.name
        method = methods[0]
        assert not method.decorator_list and method.col_offset == 4
        replacement = textwrap.indent(textwrap.dedent(METHODS[node.name]).strip() + "\n", "    ")
        edits.append((method.lineno - 1, method.end_lineno, replacement))
    assert len(edits) == 2, "Expected both expert mapping methods"
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    updated = "".join(lines) + "\n" + marker + "\n"
    compile(updated, str(path), "exec")
    backup = path.with_name(path.name + ".before-expert-etp")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    diff = "".join(difflib.unified_diff(text.splitlines(True), updated.splitlines(True), fromfile="before/qwen3_vl_bridge.py", tofile="after/qwen3_vl_bridge.py"))
    path.with_name(path.name + ".expert-etp.patch").write_text(diff, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    print(diff)
    print("EXPERT_ETP_PATCH_OK")


if __name__ == "__main__":
    patch(Path("/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/src/Megatron-Bridge-interns2/src/megatron/bridge/models/qwen_vl/qwen3_vl_bridge.py"))
