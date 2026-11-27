"""Copy runtime entry points to the layout used by the verified cluster scripts."""

import argparse
import hashlib
import runpy
from pathlib import Path


def prepare(root, dry_run=False):
    sources = []
    for relative, pattern in (
        ("tools/runtime", "*.py"), ("tools/launch", "*.sh"),
        ("tools/environment", "fix_*.sh"),
        ("tools/environment/requirements", "*.txt"),
        ("tools/preprocess", "*.py"), ("tools/benchmark", "*.py"),
    ):
        sources.extend(sorted((root / relative).glob(pattern)))
    entries = {}
    for source in sources:
        destination = root / source.name
        if destination in entries:
            raise RuntimeError("Duplicate runtime entry-point filename: " + source.name)
        entries[destination] = source.read_bytes()
    components = runpy.run_path(str(root / "tools/environment/component_versions.py"))["COMPONENTS"]
    for component in components:
        if component.get("entry_point") and not (root / component["directory"] / component["entry_point"]).is_file():
            raise FileNotFoundError("Component source entry point missing: " + component["directory"])
        # Existing audited trainers recognize this metadata when a component
        # is vendored without its original .git directory.
        entries[root / component["directory"] / "source-commit.txt"] = (component["commit"] + "\n").encode()
    by_name = {component.get("name"): root / component["directory"] for component in components}
    if "Megatron-LM" in by_name and "Megatron-Bridge" in by_name:
        link = by_name["Megatron-Bridge"] / "3rdparty/Megatron-LM"
        target = link.readlink() if link.is_symlink() else Path(link.read_text(encoding="utf-8"))
        if (link.parent / target).resolve() != by_name["Megatron-LM"].resolve():
            raise RuntimeError("Bridge Core source link does not resolve to DLTrain-framework")
    conflicts = [path for path, data in entries.items()
                 if path.exists() and path.read_bytes() != data]
    if conflicts:
        raise RuntimeError("Refusing to replace differing existing files: " + ", ".join(str(path) for path in conflicts))
    for destination, data in entries.items():
        if not dry_run and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        print(("CHECK" if dry_run else "READY"), destination.relative_to(root),
              hashlib.sha256(data).hexdigest())
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parents[2], args.dry_run)
    print("WORKSPACE_ENTRY_POINTS", count)
