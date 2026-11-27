"""Candidate guarded legacy-enum comparisons; preserves source formatting."""

import ast
import difflib
from pathlib import Path

path = Path(
    "/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/src/"
    "DLTrain-Megatron-Bridge-interns2/src/megatron/bridge/models/model_provider.py"
)
text = path.read_text(encoding="utf-8")
marker = "# mcore018-legacy-modeltype-comparisons"

if marker in text:
    print("MODELTYPE_COMPAT_PATCH_ALREADY_PRESENT")
else:
    tree = ast.parse(text)
    refs = {
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ModelType"
        and node.attr == "encoder_and_decoder"
    }
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def position(row, col):
        prefix = lines[row - 1].encode("utf-8")[:col].decode("utf-8")
        return offsets[row - 1] + len(prefix)

    edits, handled = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        matched = refs & {node.left, *node.comparators}
        if not matched:
            continue
        expression = ast.get_source_segment(text, node)
        exists = 'hasattr(ModelType, "encoder_and_decoder")'
        if isinstance(node.ops[0], (ast.Eq, ast.Is)):
            replacement = f"({exists} and ({expression}))"
        elif isinstance(node.ops[0], (ast.NotEq, ast.IsNot)):
            replacement = f"(not {exists} or ({expression}))"
        else:
            continue
        handled.update(matched)
        edits.append((position(node.lineno, node.col_offset), position(node.end_lineno, node.end_col_offset), replacement))
    assert refs and handled == refs, (
        "Unrecognized legacy enum usage; source unchanged",
        sorted(node.lineno for node in refs - handled),
    )
    updated = text
    for start, end, replacement in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    updated += "\n" + marker + "\n"
    compile(updated, str(path), "exec")
    backup = path.with_name(path.name + ".before-mcore018-modeltype")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    diff = "".join(difflib.unified_diff(
        text.splitlines(True), updated.splitlines(True),
        fromfile="before/model_provider.py", tofile="after/model_provider.py",
    ))
    path.with_name(path.name + ".mcore018-modeltype.patch").write_text(diff, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    print(diff)
    print("MODELTYPE_COMPAT_PATCH_OK", len(edits))
