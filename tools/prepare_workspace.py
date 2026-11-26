"""Copy runtime entry points to the layout used by the verified cluster scripts."""

import argparse
import hashlib
from pathlib import Path


def prepare(root, dry_run=False):
    sources = sorted((root / "cluster").glob("*.py")) + sorted((root / "cluster").glob("*.sh"))
    sources += sorted((root / "requirements").glob("*.txt"))
    conflicts = [root / source.name for source in sources
                 if (root / source.name).exists() and (root / source.name).read_bytes() != source.read_bytes()]
    if conflicts:
        raise RuntimeError("Refusing to replace differing existing files: " + ", ".join(str(path) for path in conflicts))
    for source in sources:
        destination = root / source.name
        if not dry_run and not destination.exists():
            destination.write_bytes(source.read_bytes())
        print(("CHECK" if dry_run else "READY"), destination.name,
              hashlib.sha256(source.read_bytes()).hexdigest())
    return len(sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parent.parent, args.dry_run)
    print("WORKSPACE_ENTRY_POINTS", count)
