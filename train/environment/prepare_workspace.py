"""Copy runtime entry points to the layout used by the verified cluster scripts."""

import argparse
import hashlib
import runpy
from pathlib import Path


def prepare(root, dry_run=False):
    sources = []
    for relative, pattern in (
        ("train/runtime", "*.py"), ("train/launch", "*.sh"),
        ("train/environment", "fix_*.sh"),
        ("train/environment/requirements", "*.txt"),
        ("data", "*.py"), ("data", "*.sh"), ("benchmark", "*.py"),
    ):
        sources.extend(sorted((root / relative).glob(pattern)))
    entries = {}
    for source in sources:
        destination = root / source.name
        if destination in entries:
            raise RuntimeError("Duplicate runtime entry-point filename: " + source.name)
        entries[destination] = source.read_bytes()
    components = runpy.run_path(str(root / "train/environment/component_versions.py"))["COMPONENTS"]
    for component in components:
        # Existing audited trainers recognize this metadata when a component
        # is vendored without its original .git directory.
        entries[root / component["directory"] / "source-commit.txt"] = (component["commit"] + "\n").encode()
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
