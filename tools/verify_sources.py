"""Check vendored files and symlink targets against the recorded source snapshot."""

import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "sources.manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        try:
            data = (path.readlink().as_posix().encode("utf-8")
                    if name in manifest["symlinks"] and path.is_symlink()
                    else path.read_bytes())
            if hashlib.sha256(data).hexdigest() != expected:
                failures.append(name + ": content differs")
        except OSError:
            failures.append(name + ": missing or unreadable")
    if failures:
        raise RuntimeError("Source verification failed:\n" + "\n".join(failures))
    print("SOURCE_SNAPSHOT_OK", len(manifest["files"]), "files,", len(manifest["symlinks"]), "symbolic links")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent.parent)
