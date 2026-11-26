"""Read server source state and export an audit ZIP without changing repositories."""

import argparse
import difflib
import hashlib
import io
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BASELINES = {
    "Megatron-LM": "ba7b5ebce12af60627a80985792a1449ce45f46c",
    "MegatronAdaptor": "59653dff2df7bf03b528a7fb8a7f1fb8a288f741",
    "TransformerEngineNPU-2.17": "09f6cc1a17314112bc1cdeef1e061c724f256f38",
}
BRIDGE_BASE = "a069bdffa41fdc4d7cb377e72a89d484167b8a85"
INHERITED = [
    "src/megatron/bridge/models/conversion/param_mapping.py",
    "src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py",
]
OWN_FIXES = {
    "src/megatron/bridge/models/model_provider.py": ".before-mcore018-modeltype",
    "src/megatron/bridge/models/qwen_vl/qwen3_vl_bridge.py": ".before-expert-etp",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(path, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["/usr/bin/git", "--no-optional-locks", "-c", f"safe.directory={path}",
         "-C", str(path), *args], capture_output=True, env=env, check=False,
    )


def delta(before, after, name):
    lines = difflib.unified_diff(
        before.decode("utf-8", errors="replace").splitlines(True),
        after.decode("utf-8", errors="replace").splitlines(True),
        fromfile="before/" + name, tofile="after/" + name,
    )
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                   for line in lines)


def collect(workdir, shared_bridge, output_dir):
    run_id = "dltrain-git-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    artifacts = {}
    report = {"audit_id": run_id, "workdir": str(workdir), "repositories": {},
              "scope": "Current source audit. Historical patches and backups identify known edit boundaries; current files do not prove who made every edit. Untracked repository files are listed but not copied."}

    def command_record(path, folder, label, args):
        result = git(path, *args)
        artifacts[f"{folder}/{label}.txt"] = result.stdout
        if result.stderr:
            artifacts[f"{folder}/{label}.stderr.txt"] = result.stderr
        return result

    for name, baseline in BASELINES.items():
        path = workdir / "src" / name
        info = {"path": str(path), "expected_baseline": baseline}
        report["repositories"][name] = info
        if not path.exists():
            info["status"] = "directory_missing"
            continue
        head = command_record(path, name, "HEAD", ["rev-parse", "HEAD"])
        if head.returncode:
            info["status"] = "git_unavailable"
            continue
        info["HEAD"] = head.stdout.decode().strip()
        info["commands"] = {}
        for label, args in {
            "status": ["status", "--porcelain=v1", "--untracked-files=all"],
            "diff_HEAD": ["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"],
            "diff_baseline": ["diff", "--no-ext-diff", "--no-textconv", "--binary", baseline, "--"],
            "baseline_name_status": ["diff", "--no-ext-diff", "--no-textconv", "--name-status", baseline, "--"],
            "commits_after_baseline": ["log", "--oneline", f"{baseline}..HEAD"],
        }.items():
            result = command_record(path, name, label, args)
            info["commands"][label] = {"returncode": result.returncode, "sha256": digest(result.stdout)}
            if label == "status":
                info["status_lines"] = result.stdout.decode("utf-8", errors="replace").splitlines()
            if label.startswith("diff_") and result.returncode == 0:
                info[label + "_is_empty"] = not bool(result.stdout)

    bridge = workdir / "src/Megatron-Bridge-interns2"
    info = {"path": str(bridge), "expected_baseline": BRIDGE_BASE,
            "inherited_dirty_files_from_history": INHERITED, "our_fix_files_from_history": list(OWN_FIXES)}
    report["bridge_snapshot"] = info
    for name in ("source-commit.txt", "source-changes.patch"):
        path = bridge / name
        if path.is_file():
            data = path.read_bytes()
            artifacts["Bridge/" + name] = data
            info[name] = {"sha256": digest(data)}
            if name.endswith(".txt"):
                info[name]["value"] = data.decode().strip()
        else:
            info[name] = {"missing": True}
    # The snapshot was copied without .git. Compare to the pinned object in the
    # shared Git repository, not its possibly changed current working files.
    base = git(shared_bridge, "cat-file", "-t", BRIDGE_BASE)
    info["pinned_base_object_available"] = base.returncode == 0 and base.stdout.strip() == b"commit"
    current = {p.relative_to(bridge).as_posix(): p for p in (bridge / "src").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".patch")
               and ".before-" not in p.name}
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (bridge / name).is_file():
            current[name] = bridge / name
    base_contents = {}
    if info["pinned_base_object_available"]:
        archived = git(shared_bridge, "archive", "--format=tar", BRIDGE_BASE, "src", "pyproject.toml", "README.md", "LICENSE")
        if archived.returncode:
            raise RuntimeError("Could not read pinned Bridge tree")
        with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    base_contents[member.name] = archive.extractfile(member).read()
    base_files = set(base_contents)
    hashes, changes, chunks = {}, [], []
    for name in sorted(set(current) | base_files):
        after = current[name].read_bytes() if name in current else b""
        if name in current:
            hashes[name] = digest(after)
        if not info["pinned_base_object_available"]:
            continue
        before = base_contents.get(name, b"")
        if before != after or (name in current) != (name in base_files):
            changes.append({"path": name, "status": "modified" if name in current and name in base_files else "added" if name in current else "deleted"})
            chunks.append(delta(before, after, name))
            if name in current:
                artifacts["Bridge/current/" + name] = after
    info["current_vs_pinned_base"] = changes if info["pinned_base_object_available"] else None
    info["current_file_sha256"] = hashes
    artifacts["Bridge/current_vs_pinned_base.diff"] = "".join(chunks).encode("utf-8")
    info["own_fix_backup_comparisons"] = {}
    for name, suffix in OWN_FIXES.items():
        path = bridge / name
        backup = path.with_name(path.name + suffix)
        if path.is_file() and backup.is_file():
            before, after = backup.read_bytes(), path.read_bytes()
            artifacts["Bridge/own-fixes/" + path.name + ".diff"] = delta(before, after, name).encode("utf-8")
            info["own_fix_backup_comparisons"][name] = {"before_sha256": digest(before), "after_sha256": digest(after)}
        else:
            info["own_fix_backup_comparisons"][name] = {"backup_or_current_missing": True}
    # Top-level training/helper scripts were added outside the cloned repos.
    report["workspace_scripts"] = {}
    for path in sorted(workdir.iterdir()):
        if path.is_file() and (path.suffix in (".py", ".sh") or path.name.endswith("requirements.txt")):
            data = path.read_bytes()
            artifacts["workspace-scripts/" + path.name] = data
            report["workspace_scripts"][path.name] = {"sha256": digest(data), "bytes": len(data)}
    artifacts["audit.json"] = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    artifacts["audit_git_changes.py"] = Path(__file__).read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (run_id + ".zip")
    with target.open("xb") as handle:
        with ZipFile(handle, "w", ZIP_DEFLATED) as bundle:
            for name, data in sorted(artifacts.items()):
                bundle.writestr(run_id + "/" + name, data)
    print("AUDIT_ZIP:", target, flush=True)
    for name, item in report["repositories"].items():
        print(name, json.dumps({k: item.get(k) for k in ("HEAD", "diff_HEAD_is_empty", "diff_baseline_is_empty", "status_lines")}, ensure_ascii=False), flush=True)
    print("BRIDGE_CHANGED_FILES:", json.dumps(info["current_vs_pinned_base"], ensure_ascii=False), flush=True)
    print("WORKSPACE_SCRIPT_COUNT:", len(report["workspace_scripts"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/mnt/afs-ptdata-shared/sxr/dltrain-mcore018"))
    parser.add_argument("--shared-bridge", type=Path, default=Path("/mnt/afs-a3-weight-share/slime4sec_qwen35/Megatron-Bridge-slime"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/workspace"))
    args = parser.parse_args()
    collect(args.workdir, args.shared_bridge, args.output_dir)
