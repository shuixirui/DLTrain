"""Check repository entry points and record the vendored component versions."""

import argparse
import hashlib
import runpy
from pathlib import Path


REQUIRED_FILES = (
    "install.sh", "examples/run.sh", "examples/train_interns2.py",
    "src/DLTrain-framework/tools/prepare_interns2_corpus.py",
    "tools/validation/check_interns2_provider.py", "tools/validation/adamw_dist_update_probe.py",
    "tools/validation/adamw_math.py", "tools/validation/adamw_shard_layout.py",
    "tools/validation/optimizer_update_probe.py",
    "tools/environment/requirements/runtime-requirements.txt",
    "tools/environment/requirements/adaptor-requirements.txt",
    "tools/environment/requirements/model-requirements.txt",
    "tools/environment/requirements/bridge-compat-requirements.txt",
    "tools/environment/requirements/tokenizer-requirements.txt",
)


def prepare(root, dry_run=False):
    sources = {root / name: (root / name).read_bytes() for name in REQUIRED_FILES}
    entries = {}
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
    for source, data in sources.items():
        print("CHECK", source.relative_to(root), hashlib.sha256(data).hexdigest())
    for destination, data in entries.items():
        if not dry_run and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        print(("CHECK" if dry_run else "READY"), destination.relative_to(root),
              hashlib.sha256(data).hexdigest())
    return len(sources) + len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parents[2], args.dry_run)
    print("WORKSPACE_CHECKS", count)
