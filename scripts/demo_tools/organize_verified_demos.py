# Copyright (c) 2025-2026, The DexVerse Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Install previously verified local collections into the category tree.

This packages saved evidence; it does not run physics, repair actions, download
new data, or promote experimental long-horizon recordings. Existing baseline
files are preserved. Use the DexVerse Python environment for NumPy pickles.

    python scripts/demo_tools/organize_verified_demos.py --dry-run
    python scripts/demo_tools/organize_verified_demos.py

The default evidence is the October 8-15 local audits. Keep those audit folders
when reproducing this selection. Per-task provenance and a collection manifest
are installed beside the data; original archives remain in place.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

from demo_layout import (
    DEFAULT_DEST, REPO_ROOT, baseline_entries, contained_path, install_file,
    sha256, sync_baseline_links, task_categories,
)

COVERED = {"public_baseline_user_confirmed", "repaired_complete", "original_complete", "new_action_replay_confirmed_50"}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def action_digest(actions):
    array = np.asarray(actions, dtype=np.float64)
    return hashlib.sha256(b"array" + str(array.shape).encode() + array.dtype.str.encode()
                          + np.ascontiguousarray(array).tobytes()).hexdigest()


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def select_verified_payload(payload, rows):
    """Select exactly the audited episodes; only names/indices are renumbered."""
    if len(rows) != 50 or len({r["actions_sha256"] for r in rows}) != 50:
        raise ValueError("A collection requires 50 distinct verified action sequences")
    result = dict(payload)
    result["episodes"] = []
    for index, row in enumerate(rows):
        if not (row["success"] and row["complete"] and row["max_success_streak"] >= 10):
            raise ValueError("Cannot promote an unsuccessful or incomplete episode")
        episode = copy.deepcopy(payload["episodes"][row["prepared_index"]])
        if action_digest(episode["actions"]) != row["actions_sha256"]:
            raise ValueError("Prepared actions differ from audited actions")
        if len(episode["actions"]) != row["actual_steps"]:
            raise ValueError("Audited horizon differs from source")
        episode["episode_index"] = index
        episode["episode_name"] = f"demo_{index:04d}"
        result["episodes"].append(episode)
    result["num_episodes"] = 50
    return result


def ready_checksum(source, task):
    delivery = next(p for p in source.parents if p.name == "delivery")
    sums = delivery / "SHA256SUMS.json"
    if sums.exists():
        return read_json(sums)[source.relative_to(delivery).as_posix()]
    summary = read_json(delivery / "summary.json")
    rows = summary.get("tasks", summary.get("collections", []))
    row = next(r for r in rows if r["task"] == task)
    return row.get("pkl_sha256", row.get("sha256"))


def snapshot_stats(payload):
    episodes = payload["episodes"]
    return {
        "episodes": len(episodes),
        "initial_state_episodes": sum(e.get("initial_state") is not None for e in episodes),
        "full_state_episodes": sum(e.get("states") is not None and len(e["states"]) == len(e["actions"]) + 1 for e in episodes),
        "distinct_action_sequences": len({action_digest(e["actions"]) for e in episodes}),
    }


def install_json(value, target, staging):
    source = staging / "metadata.json"
    source.write_text(json.dumps(value, indent=2) + "\n")
    install_file(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-root", type=Path, default=REPO_ROOT / "outputs/new_demos_probe_20261015")
    parser.add_argument("--legacy-root", type=Path, default=REPO_ROOT / "outputs/legacy_replay_audit_20261008")
    parser.add_argument("--baseline-source", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dest = args.dest.expanduser().resolve()
    categories = task_categories()
    status = read_json(args.probe_root / "consolidated_status.json")
    task_rows = [t for t in status["tasks"] if t["demos_required"] and t["category"] in COVERED
                 and categories[t["task"]] != "long_horizon"]
    new_selection = collections.defaultdict(list)
    for row in read_json(args.probe_root / "successful_50_selection.json"):
        new_selection[row["task"]].append(row)
    legacy = {r["task"]: r for r in read_json(args.legacy_root / "full_session_verification.json")}
    baseline = set(baseline_entries())
    manifest_rows = []
    source_hashes = {}
    with tempfile.TemporaryDirectory(prefix="dexverse-organize-") as temporary:
        staging = Path(temporary)
        for task in sorted(task_rows, key=lambda t: t["task"]):
            name = task["task"]
            task_id = f"Dexverse-{name}-v0"
            relative = f"{categories[name]}/{task_id}/{task_id}.pkl"
            target = contained_path(args.dest, relative)
            kind = task["category"]
            provenance = {"task": task_id, "selection_status": kind, "evidence": task["evidence"] if "evidence" in task else None}
            if kind == "public_baseline_user_confirmed":
                source = args.baseline_source / relative
                provenance.update(source=str(source.resolve()), source_sha256=sha256(source),
                                  validation="Existing accepted baseline release; not re-audited in this organization pass")
                payload = read_pickle(source)
            elif task.get("ready_pickle"):
                source = Path(task["ready_pickle"])
                expected = ready_checksum(source, name)
                if not expected or sha256(source) != expected:
                    raise ValueError(f"Verified delivery checksum mismatch: {source}")
                payload = read_pickle(source)
                provenance.update(source=str(source), source_sha256=expected,
                                  previous_motion_repairs=task.get("modified", 0))
            elif kind == "original_complete":
                evidence = legacy[name]
                if evidence["successes"] != 50 or not evidence["all_episodes_complete"] or not evidence["actions_equal_original"]:
                    raise ValueError(f"Incomplete legacy evidence for {name}")
                original = Path(evidence["source"])
                if sha256(original) != evidence["source_sha256"]:
                    raise ValueError(f"Legacy source checksum mismatch: {original}")
                source = Path(evidence["normalized_source"])
                payload = read_pickle(source)
                raw = read_pickle(original)
                for selected, old in zip(payload["episodes"], raw["episodes"]):
                    for key in ("actions", "initial_state", "goal_pose", "task_state", "multi_assets", "multi_usds"):
                        if not equal(selected.get(key), old.get(key)):
                            raise ValueError(f"Normalized source changed {name}/{key}")
                provenance.update(source=str(source), source_sha256=sha256(source), original_source=str(original),
                                  original_sha256=evidence["source_sha256"], source_indices=list(range(50)),
                                  evidence_h5=evidence["h5"])
            else:
                source_prepared = args.probe_root / "prepared" / name / "sources.pkl"
                prepared = read_pickle(source_prepared)
                rows = new_selection[name]
                payload = select_verified_payload(prepared, rows)
                mapping = read_json(source_prepared.with_name("mapping.json"))
                provenance_rows = []
                for row in rows:
                    mapped = mapping[row["prepared_index"]]
                    original = Path(mapped["original_file"])
                    if original not in source_hashes:
                        source_hashes[original] = sha256(original)
                    if source_hashes[original] != mapped["original_file_sha256"]:
                        raise ValueError(f"Archived source checksum mismatch: {original}")
                    provenance_rows.append({**row, "original_file": str(original), "original_file_sha256": source_hashes[original]})
                source = staging / f"{task_id}.pkl"
                with source.open("wb") as stream:
                    pickle.dump(payload, stream, protocol=4)
                roundtrip = read_pickle(source)
                if not equal(roundtrip, payload):
                    raise ValueError(f"Serialization changed {name}")
                provenance.update(source=str(source_prepared.resolve()), source_sha256=sha256(source_prepared),
                                  previous_motion_repairs=0, selected_episodes=provenance_rows)
            if payload.get("env_name") != task_id:
                raise ValueError(f"Task identity mismatch: {name}")
            stats = snapshot_stats(payload)
            if kind != "public_baseline_user_confirmed" and (stats["episodes"] != 50 or stats["distinct_action_sequences"] != 50):
                raise ValueError(f"Incomplete or duplicate collection: {name}")
            if stats["initial_state_episodes"] != stats["episodes"]:
                raise ValueError(f"Missing initial states: {name}")
            source_digest = sha256(source)
            # Check all existing destinations even in dry-run mode.
            if target.is_symlink() or (target.exists() and sha256(target) != source_digest):
                raise FileExistsError(f"Different local data exists: {target}")
            row = {"task": task_id, "category": categories[name], "file": relative, "sha256": source_digest,
                   **stats, "baseline": "/".join(relative.split("/")[:2]) in baseline,
                   "selection_status": kind, "action_replay_verified": kind != "public_baseline_user_confirmed",
                   "set_state_replay_verified": task.get("set_state_successful") == 50,
                   "previous_motion_repairs": task.get("modified", 0),
                   "evidence": provenance["evidence"], "provenance": str(Path(relative).with_name("provenance.json"))}
            if not args.dry_run:
                install_file(source, target, source_digest)
                install_json(provenance, target.with_name("provenance.json"), staging)
            manifest_rows.append(row)
            print(f"{'Would install' if args.dry_run else 'Organized'} {relative}: {stats['episodes']} episodes ({kind})", flush=True)
            if source.parent == staging:
                source.unlink()
        long_horizon = sorted(t for t, category in categories.items() if category == "long_horizon")
        covered_names = {r["task"].removeprefix("Dexverse-").removesuffix("-v0") for r in manifest_rows}
        missing_short = sorted(set(categories) - set(long_horizon) - covered_names)
        manifest = {
            "schema": "dexverse-local-collection-v1", "as_of": "2026-10-15",
            "scope": "Existing v0 baselines and complete previously verified short-horizon collections",
            "short_horizon_task_families": len(categories) - len(long_horizon),
            "covered_short_horizon_task_families": len(manifest_rows),
            "missing_short_horizon_tasks": missing_short,
            "missing_long_horizon_tasks": long_horizon,
            "long_horizon_collections_available": 0,
            "baseline_task_families": len(baseline),
            "baseline_collections_available": sum(r["baseline"] for r in manifest_rows),
            "canonical_files": len(manifest_rows), "episodes": sum(r["episodes"] for r in manifest_rows),
            "new_repairs_performed": False, "tasks": manifest_rows,
        }
        if not args.dry_run:
            install_json(manifest, args.dest / "collection_manifest.json", staging)
            links = sync_baseline_links(args.dest, baseline_entries())
            manifest["baseline_symlinks"] = len(links["links"])
        print(json.dumps({k: v for k, v in manifest.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
