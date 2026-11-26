"""Import pinned Git source archives and replay audited Bridge patches.

Maintainer tool for an empty src/ tree. Normal users already have all source
after cloning DLTrain and do not need to run this tool or clone dependencies.
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent


def git(*args, cwd=None):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-c", "apply.whitespace=nowarn", *args],
        cwd=cwd or ROOT,
    )


def vendor(cache_directory):
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    targets = [ROOT / component["directory"] for component in lock["components"]]
    if any(path.exists() for path in targets):
        raise RuntimeError("Import requires empty component destinations; existing sources were not changed")
    archives = []
    for component in lock["components"]:
        cache = cache_directory / component["cache_directory"]
        raw = git("-C", str(cache), "archive", "--format=tar", component["commit"])
        archives.append((component, raw))
    symlinks, modes = {}, {}
    for component, raw in archives:
        target = ROOT / component["directory"]
        target.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for member in archive.getmembers():
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts or "\\" in member.name or ":" in member.name:
                    raise ValueError(f"Invalid source archive path: {member.name}")
                path = target / Path(*rel.parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.issym():
                    # A symlink checkout on Windows with core.symlinks=false
                    # stores its target as a regular file. Stage mode 120000
                    # later; Linux clones obtain normal symbolic links.
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(member.linkname.encode("utf-8"))
                    symlinks[path.relative_to(ROOT).as_posix()] = member.linkname
                    modes[path.relative_to(ROOT).as_posix()] = "120000"
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
                    modes[path.relative_to(ROOT).as_posix()] = "100755" if member.mode & 0o111 else "100644"
                else:
                    raise ValueError(f"Unsupported source archive entry: {member.name}")
        for patch in component["patches"]:
            git("apply", "--check", f"--directory={component['directory']}", patch)
            git("apply", f"--directory={component['directory']}", patch)
        print("IMPORTED", component["name"], component["commit"], flush=True)
    for link in lock.get("integration_links", []):
        path = ROOT / link["path"]
        if path.is_dir():
            if any(path.iterdir()):
                raise RuntimeError("Refusing to replace a nonempty integration link directory")
            path.rmdir()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(link["target"].encode("utf-8"))
        symlinks[link["path"]] = link["target"]
        modes[link["path"]] = "120000"
    manifest = {"schema_version": 1, "files": {}, "symlinks": symlinks, "file_modes": modes}
    for component in lock["components"]:
        for path in sorted((ROOT / component["directory"]).rglob("*")):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                manifest["files"][rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "sources.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_FILES", len(manifest["files"]), "SYMLINKS", len(symlinks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-directory", type=Path, required=True)
    args = parser.parse_args()
    vendor(args.cache_directory.resolve())
