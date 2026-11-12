"""Copy trusted NumPy trajectory pickles into version-separated local task folders.

Planning is read-only except for the JSON inventory under outputs/. Apply never
overwrites inputs or existing different outputs. Remote access is download-only;
credentials are supplied by the user's existing Hugging Face login, never saved.
"""

from __future__ import annotations

import argparse
import builtins
import collections
import hashlib
import json
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from dexverse.benchmark import BASELINE_PAIRS, LEGACY_UPGRADE_TASKS, task_identity  # noqa: E402
from dexverse.recording_compat import recorded_task_name  # noqa: E402

DEST = ROOT / "source/dexverse/demonstrations"
IDENTITY_KEYS = (
    "task",
    "env_name",
    "format",
    "benchmark_revision",
    "task_version",
    "task_source_revision",
    "action_layout",
)


class NumpyCompatUnpickler(pickle.Unpickler):
    """Allow only the NumPy/container constructors used by these recordings.

    NumPy 2 pickles refer to numpy._core, unavailable in NumPy 1.26. Remapping
    GLOBAL references at unpickle time preserves values; byte substitution would
    corrupt pickle FRAME lengths. Re-serialization uses the active NumPy version.
    This is a narrow compatibility loader, not a general untrusted-pickle sandbox.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self.remapped = set()

    def find_class(self, module, name):
        original = module
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
            self.remapped.add((original, module, name))
        allowed = {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"),
        }
        if (module, name) in allowed:
            return super().find_class(module, name)
        if module == "builtins" and name in {"set", "frozenset", "slice", "complex", "bytearray"}:
            return getattr(builtins, name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "_codecs" and name == "encode":
            return _latin1_encode
        raise pickle.UnpicklingError(f"Unsupported pickle constructor: {module}.{name}")


def _latin1_encode(value, encoding="latin1", errors="strict"):
    """Old pickle protocols represent byte arrays through this constructor."""
    if encoding not in {"latin1", "latin-1"} or errors != "strict":
        raise pickle.UnpicklingError("Only strict latin1 byte reconstruction is supported")
    return value.encode("latin1")


def load_compatible(path):
    with Path(path).open("rb") as stream:
        loader = NumpyCompatUnpickler(stream)
        payload = loader.load()
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError("Expected a trajectory dict with episodes")
    return payload, sorted(loader.remapped)


def canonical_v0(task):
    return recorded_task_name(task.replace("Dexbench-FixateThenManipulate-", "Dexverse-").replace("Dexbench-", "Dexverse-"))


def category(task):
    for line in (DEST / "baseline_manifest.txt").read_text().splitlines():
        if (
            line
            and not line.startswith("#")
            and line.split("/")[-1].removesuffix("-v0") == task.removesuffix("-v1").removesuffix("-v0")
        ):
            return line.split("/")[0]
    raise ValueError(f"Task is not in the baseline manifest: {task}")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(value):
    """Digest numerical contents, dtype, shape, and container structure."""
    h = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise ValueError("Object arrays need manual review")
            h.update(str((obj.dtype.str, obj.shape)).encode())
            h.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(str((type(obj).__name__, len(obj))).encode())
            for child in obj:
                visit(child)
        elif isinstance(obj, np.generic):
            visit(np.asarray(obj))
        else:
            h.update(repr(obj).encode())

    visit(value)
    return h.hexdigest()


def inspect_payload(payload):
    episodes = payload["episodes"]
    dims, entities, issues = set(), set(), set()
    for ep in episodes:
        actions = np.asarray(ep.get("actions", []))
        if actions.ndim != 2 or not len(actions):
            issues.add("empty_or_invalid_actions")
            continue
        dims.add(actions.shape[-1])
        if not np.isfinite(actions).all():
            issues.add("nonfinite_actions")
        states = ep.get("states")
        if states is None or len(states) != len(actions) + 1:
            issues.add("missing_or_misaligned_recorded_states")
        for values in ep.get("initial_state", {}).values():
            if isinstance(values, dict):
                entities.update(values)
    if len(dims) != 1:
        issues.add("mixed_action_dimensions")
    if not episodes:
        issues.add("empty_recording")
    robot = payload.get("robot_type", "")
    expected = {"floating_shadow_right": 28, "floating_shadow_left": 28, "floating_shadow_bimanual": 56}.get(robot)
    if expected is None or dims != {expected}:
        issues.add("unsupported_or_mismatched_robot_layout")
    return {
        "episodes": len(episodes),
        "recorded_successes": sum(bool(e.get("success")) for e in episodes),
        "action_dims": sorted(dims),
        "scene_entities": sorted(entities),
        "issues": sorted(issues),
        "task_state_episodes": sum(bool(e.get("task_state")) for e in episodes),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def plan_local(private_root):
    # Prefer the latest collection per task. Tests, backups, renders and robot
    # experiments are not demonstration sources. Keep dated session filenames.
    collections_by_task = collections.defaultdict(list)
    allowed = {
        "0821_yunchao_hammer_withta_table",
        "0823_zhuxiu_teleop",
        "0824_yunchao_actual",
        "0829_yunchao",
        "0905_zhuxiu",
    }
    for directory in sorted(private_root.iterdir()):
        if directory.name not in allowed:
            continue
        for task_dir in sorted(directory.iterdir()):
            if task_dir.is_dir() and list(task_dir.glob("*.pkl")):
                collections_by_task[task_dir.name].append(task_dir)
    items, excluded = [], []
    for source_task, directories in sorted(collections_by_task.items()):
        target = LEGACY_UPGRADE_TASKS.get(source_task, BASELINE_PAIRS.get(source_task))
        if target is None:
            continue
        latest = directories[-1]
        excluded.extend(str(p) for directory in directories[:-1] for p in directory.glob("*.pkl"))
        for path in sorted(latest.glob("*.pkl")):
            payload, remaps = load_compatible(path)
            if payload.get("task") != source_task:
                raise ValueError(f"Task/path disagreement: {path}")
            summary = inspect_payload(payload)
            reasons = list(summary["issues"])
            if target == "Dexverse-OpenDoor-v1":
                reasons.append("retired_tilted_door_not_cutaway_latch_v1")
            if target == "Dexverse-OpenLaptop-v1" and any(
                n in summary["scene_entities"] for n in ("laptop_goal", "laptop_open_region")
            ):
                reasons.append("older_laptop_placement_goal_not_current_v1")
            if target == "Dexverse-BimanualLiftCarton-v1" and summary["task_state_episodes"] != summary["episodes"]:
                reasons.append("missing_recorded_carton_target_face")
            items.append(
                {
                    "source": str(path),
                    "source_task": source_task,
                    "target_task": target,
                    "version": 1,
                    "collection": latest.parent.name,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "summary": summary,
                    "numpy_remaps": remaps,
                    "review_reasons": reasons,
                }
            )
    return items, excluded


def plan_remote(repo):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True, timeout=30)
    reruns, state = collections.defaultdict(list), collections.defaultdict(list)
    for entry in info.siblings:
        name = entry.rfilename
        if not name.endswith(".pkl"):
            continue
        parts = name.split("/")
        task = canonical_v0(parts[-2])
        if task not in BASELINE_PAIRS:
            continue
        if name.startswith("baseline_rerun/"):
            reruns[task].append(entry)
        elif name.startswith(("pkl/state/floating_shadow_right/", "pkl/state/floating_shadow_bimanual/")):
            state[task].append(entry)
    items = []
    for task in BASELINE_PAIRS:
        for entry in sorted(reruns.get(task) or state.get(task, []), key=lambda x: x.rfilename):
            items.append(
                {
                    "remote_file": entry.rfilename,
                    "repo": repo,
                    "revision": info.sha,
                    "target_task": task,
                    "version": 0,
                    "collection": "baseline_rerun" if entry.rfilename.startswith("baseline_rerun/") else "state_shadow",
                    "bytes": entry.size,
                    "remote_sha256": entry.lfs.sha256 if entry.lfs else None,
                }
            )
    return items, info.sha


def convert(item, source, dest):
    payload, remaps = load_compatible(source)
    summary = inspect_payload(payload)
    reasons = sorted(set(item.get("review_reasons", []) + summary["issues"]))
    source_task = payload.get("env_name") or payload.get("task")
    if item["version"] == 0 and canonical_v0(source_task) != item["target_task"]:
        raise ValueError(f"Unexpected task in remote file: {source_task}")
    source_hash = sha256(source)
    if item.get("sha256") and source_hash != item["sha256"]:
        raise ValueError("Source changed since planning")
    if item.get("remote_sha256") and source_hash != item["remote_sha256"]:
        raise ValueError("Remote hash mismatch")
    original_identity = {key: payload.get(key) for key in IDENTITY_KEYS}
    data_digest = array_digest({k: v for k, v in payload.items() if k not in IDENTITY_KEYS})
    target = item["target_task"]
    version = item["version"]
    identity = task_identity(target)
    payload.update(
        task=target,
        env_name=target,
        format="dexverse_trajectory",
        task_version=version,
        benchmark_revision=identity["benchmark_revision"],
    )
    # Do not invent recording-time source revisions or joint ordering.
    payload["task_source_revision"] = original_identity["task_source_revision"]
    provenance = {
        "source": item.get("source") or item["remote_file"],
        "source_sha256": source_hash,
        "original_identity": original_identity,
        "numpy_module_remaps": remaps,
        "normalized_with_numpy": np.__version__,
        "target_task_source_revision": identity["task_source_revision"],
        "compatibility": "needs_review" if reasons else "not_simulation_verified",
        "review_reasons": reasons,
        "array_and_episode_digest": data_digest,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if "repo" in item:
        provenance.update(remote_repo=item["repo"], remote_revision=item["revision"])
    payload["demo_import"] = provenance
    directory = dest / ("_review" if reasons else f"v{version}")
    if reasons:
        directory = directory / f"v{version}"
    directory = directory / category(target) / target / item["collection"]
    filename = Path(source).name.replace(source_task, target)
    if not filename.startswith(target):
        filename = target + "__" + filename
    output = directory / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing, _ = load_compatible(output)
        if existing.get("demo_import", {}).get("source_sha256") != source_hash:
            raise FileExistsError(f"Refusing to overwrite {output}")
        status = "already_imported"
    else:
        # Publish only a complete file. Hard-link creation is atomic and refuses
        # to replace an existing destination, including a concurrent import.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                pickle.dump(payload, stream, protocol=4)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        status = "imported"
    with output.open("rb") as stream:
        restored = pickle.load(stream)
    assert restored["task"] == target and restored["task_version"] == version
    assert (
        array_digest({k: v for k, v in restored.items() if k not in IDENTITY_KEYS and k != "demo_import"})
        == data_digest
    )
    return {
        "output": str(output.relative_to(dest)),
        "status": status,
        "source_sha256": source_hash,
        "output_sha256": sha256(output),
        "target_task": target,
        "version": version,
        "summary": summary,
        "numpy_remaps": remaps,
        "review_reasons": reasons,
    }


def apply_plan(plan, dest, workspace):
    from huggingface_hub import hf_hub_download

    results, hashes = [], {}
    report = workspace / "import_report.json"
    for item in plan["items"]:
        try:
            if "remote_file" in item:
                source = Path(
                    hf_hub_download(
                        repo_id=item["repo"],
                        repo_type="dataset",
                        revision=item["revision"],
                        filename=item["remote_file"],
                        local_dir=workspace / "remote_raw",
                    )
                )
            else:
                source = Path(item["source"])
            digest = sha256(source)
            key = (item["version"], item["target_task"], digest)
            if key in hashes:
                results.append({"status": "duplicate_source", "source": str(source), "duplicate_of": hashes[key]})
            else:
                result = convert(item, source, dest)
                hashes[key] = result["output"]
                results.append(result)
                print(
                    json.dumps({k: result[k] for k in ("status", "target_task", "output", "review_reasons")}),
                    flush=True,
                )
        except Exception as exc:
            results.append({"status": "failed", "item": item, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {item['target_task']}: {type(exc).__name__}: {exc}", flush=True)
        write_json(report, {"destination": str(dest), "results": results})
    return int(any(r["status"] == "failed" for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--hf-repo", help="Optional private HF dataset to download; no default or upload path.")
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs/demo_import")
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--apply", action="store_true", help="Apply an existing reviewed plan.json")
    args = parser.parse_args()
    path = args.workspace / "plan.json"
    if args.apply:
        return apply_plan(json.loads(path.read_text()), args.dest, args.workspace)
    items, excluded = plan_local(args.private_root) if args.private_root else ([], [])
    revision = None
    if args.hf_repo:
        remote, revision = plan_remote(args.hf_repo)
        items.extend(remote)
    write_json(path, {"items": items, "excluded_older_local_sessions": excluded, "remote_revision": revision})
    print(
        json.dumps(
            {
                "plan": str(path),
                "files": len(items),
                "bytes": sum(i["bytes"] for i in items),
                "tasks_v1": sorted({i["target_task"] for i in items if i["version"] == 1}),
                "tasks_v0": sorted({i["target_task"] for i in items if i["version"] == 0}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
