"""Simulator-free selection of curated versioned demos or explicit raw sessions."""

import json
from pathlib import Path
import sys
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/dexverse"))
from dexverse.recording_compat import normalize_recording_identity, recording_output_label  # noqa: E402

from demo_release import baseline_entries, baseline_key, contained_path, read_manifest, sha256


class PickleGroup(NamedTuple):
    label: str
    output_stem: str
    anchor_dir: Path
    pickles: list[Path]
    expected_task: str | None = None


def normalize_demo_root(root):
    """Accept a download root, prepared release root, or explicit v0/v1 root."""
    root = Path(root).expanduser().resolve()
    if (root / "demonstrations/release_manifest.json").is_file():
        root /= "demonstrations"
    return root


def discover_groups(root, *, tasks=(), files=(), all_tasks=False, version="all", legacy_collections=False):
    """Curated selection never descends into local_collection, backups or _review.

    Return groups plus explicit coverage warnings for --all. Individual missing
    tasks fail with download guidance. File overrides are deliberate, not fallback.
    """
    root = normalize_demo_root(root)
    if version not in {"all", "v0", "v1"}:
        raise ValueError(f"Invalid version {version}")
    if not (tasks or files or all_tasks):
        raise ValueError("Nothing selected. Pass --task Dexverse-PushT-v1, --all, or --file PATH.")
    if files and (tasks or all_tasks or version != "all" or legacy_collections):
        raise ValueError("Use --file separately from --task/--all/--version/--legacy-collections.")
    if tasks and all_tasks:
        raise ValueError("Choose --task or --all, not both.")
    groups, warnings = [], []
    if files:
        for filename in dict.fromkeys(files):
            path = Path(filename).expanduser().resolve()
            if not path.is_file() or path.suffix != ".pkl":
                raise FileNotFoundError(f"Expected a pickle file: {path}")
            groups.append(PickleGroup(
                recording_output_label(path.name), recording_output_label(path.stem), path.parent, [path],
            ))
        return groups, warnings
    if legacy_collections:
        if version != "all":
            raise ValueError("--version is for curated datasets; raw collections use explicit directory paths.")
        directories = [root] if all_tasks else [contained_path(root, task) for task in tasks]
        parents = set()
        for directory in directories:
            if not directory.is_dir():
                raise FileNotFoundError(f"Collection directory not found: {directory}")
            for path in directory.rglob("*.pkl"):
                relative = path.relative_to(root)
                if any(p == "_review" or "backup" in p.lower() or "copy" in p.lower() for p in relative.parts):
                    continue
                contained_path(root, relative.as_posix())
                parents.add(path.resolve().parent)
        for directory in sorted(parents):
            groups.append(PickleGroup(
                recording_output_label(directory.name), recording_output_label(directory.name),
                directory, sorted(directory.glob("*.pkl")),
            ))
        if not groups:
            raise FileNotFoundError("No raw collection pickles selected; use --file for an explicit recording.")
        return groups, warnings

    root_version = root.name if root.name in {"v0", "v1"} else None
    if root_version and version not in ("all", root_version):
        raise ValueError("--demos-root version conflicts with --version")
    selected_version = root_version or version
    keys = baseline_entries(selected_version) if all_tasks else list(dict.fromkeys(baseline_key(t, selected_version) for t in tasks))
    manifest_path = root.parent / "release_manifest.json" if root_version else root / "release_manifest.json"
    records = {r["key"]: r for r in read_manifest(manifest_path)["tasks"]} if manifest_path.is_file() else None
    for key in keys:
        relative = key.split("/", 1)[1] if root_version else key
        path = contained_path(root, relative + "/demos.pkl")
        task = key.split("/")[-1]
        record = records[key] if records is not None else None
        if record is not None and not record["files"]:
            message = f"{task}: {record['status']}. {record['reason']}"
        elif not path.is_file():
            message = (f"{task}: curated demo not found: {path}. Download with "
                       f"python scripts/demo_tools/download_demos.py --repo OWNER/DATASET --task {task}; "
                       "or point --demos-root at a prepared release. Raw sessions require --file or --legacy-collections.")
        else:
            if record is not None:
                entry = record["files"][0]
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Release checksum/size mismatch: {path}")
            groups.append(PickleGroup(key, task, path.parent, [path], task))
            if record is not None and record["status"] != "complete":
                warnings.append(f"{task}: {record['status']} ({record['episodes']}/50)")
            continue
        if not all_tasks:
            raise FileNotFoundError(message)
        warnings.append(message)
    if not groups:
        raise FileNotFoundError("No curated demos available. " + "\n".join(warnings))
    return groups, warnings


def validate_selected_identity(payload, expected_task):
    """Reject misplaced files before constructing an environment, even with overrides."""
    if expected_task is None:
        return
    payload = normalize_recording_identity(payload)
    identities = [payload.get(k) for k in ("task", "env_name") if payload.get(k) is not None]
    if not identities or any(task != expected_task for task in identities):
        raise ValueError(f"Selected {expected_task}, but the curated pickle identifies {identities!r}")
    version = int(expected_task[-1])
    if payload.get("task_version", version) != version:
        raise ValueError(f"Curated pickle task_version does not match {expected_task}")


def worker_arguments(group, arguments):
    """Retain render/output options, replacing all selection with one exact group."""
    values = {"--task", "--file", "--version", "--demos-root", "--worker-group"}
    switches = {"--all", "--legacy-collections", "--dry-run"}
    retained, skip = [], False
    for arg in arguments:
        if skip:
            skip = False
            continue
        name = arg.split("=", 1)[0]
        if name in values:
            skip = "=" not in arg
        elif name not in switches:
            retained.append(arg)
    data = group._asdict()
    data["anchor_dir"] = str(group.anchor_dir)
    data["pickles"] = [str(p) for p in group.pickles]
    return retained + ["--worker-group", json.dumps(data)]
