"""Copy runtime entry points to the layout used by the verified cluster scripts."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(root, dry_run=False):
    sources = sorted((root / "cluster").glob("*.py")) + sorted((root / "cluster").glob("*.sh"))
    sources += sorted((root / "requirements").glob("*.txt"))
    entries = {root / source.name: source.read_bytes() for source in sources}
    lock = json.loads((root / "sources.lock.json").read_text(encoding="utf-8"))
    for component in lock["components"]:
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
    count = prepare(Path(__file__).resolve().parent.parent, args.dry_run)
    print("WORKSPACE_ENTRY_POINTS", count)
