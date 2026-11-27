"""Check source entry points, the Core link, and Git source changes."""

import argparse
from pathlib import Path
import runpy
import shutil
import subprocess


def verify(root, layout_only=False):
    components = runpy.run_path(str(root / "train/environment/component_versions.py"))["COMPONENTS"]
    failures = []
    for component in components:
        path = root / component["directory"] / component["entry_point"]
        if not path.is_file():
            failures.append(str(path) + ": source entry point missing")
    core = root / next(item["directory"] for item in components if item["name"] == "Megatron-LM")
    bridge = root / next(item["directory"] for item in components if item["name"] == "Megatron-Bridge")
    link = bridge / "3rdparty/Megatron-LM"
    try:
        target = link.readlink() if link.is_symlink() else Path(link.read_text(encoding="utf-8"))
        if (link.parent / target).resolve() != core.resolve():
            failures.append("Bridge Core source link points to a different component")
    except OSError:
        failures.append("Bridge Core source link missing")
    if not layout_only and (root / ".git").exists():
        binary = "/usr/bin/git" if Path("/usr/bin/git").is_file() else shutil.which("git")
        if not binary:
            failures.append("Git is required to verify tracked source changes")
        else:
            result = subprocess.run([binary, "-c", f"safe.directory={root.resolve().as_posix()}",
                                     "-C", str(root), "diff", "--exit-code", "--no-ext-diff",
                                     "--no-textconv", "HEAD", "--", "src"], capture_output=True)
            if result.returncode:
                failures.append("Tracked source changes are present; inspect git diff HEAD -- src")
    if failures:
        raise RuntimeError("Source verification failed:\n" + "\n".join(failures))
    print("SOURCE_LAYOUT_OK", len(components), "components; Core link resolves")
    if layout_only or not (root / ".git").exists():
        print("CHECK_SCOPE: layout only; Git source diff was not checked")
    else:
        print("GIT_SOURCE_DIFF_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent.parent, args.layout_only)
