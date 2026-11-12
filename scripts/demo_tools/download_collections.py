# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Named state-demo collections and category-layout compatibility helpers.

    python scripts/demo_tools/download_demos.py --baseline
    python scripts/demo_tools/download_demos.py --main
    python scripts/demo_tools/download_demos.py --cross_20
    python scripts/demo_tools/download_demos.py --all --version all
    python scripts/demo_tools/download_demos.py --category articulation
    python scripts/demo_tools/download_demos.py --task contact_rich/Dexverse-PlugCharger-v0
    python scripts/demo_tools/download_demos.py --link-baseline-only

The default v0 selection matches this checkout. v1 files keep their -v1 task IDs.
Both demonstrations/v0/<category>/<task>/... and the older unversioned remote
layout install into <dest>/<category>/<task>/.... Baseline is a local symlink
view of those files. Different local recordings are never overwritten.
The main download contains the 42 approved v0 Shadow tasks with complete
saved states. It installs under <dest>/main/<category>/<task>/, separate from
baseline and v1 data. The 20 held-out RelocateObject validation PKLs are hosted
online but excluded from --main. Replay main demos with --set-state.
The cross_20 collection has 7 tasks x 6 hands x 20 saved-state episodes (840).
It installs under <dest>/cross_20/<category>/<task>/<robot>/, preserving main,
baseline and v1 data. Replay cross_20 demos with --set-state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import json
from pathlib import Path
import re

from demo_layout import (
    BASELINE_MANIFEST, CATEGORIES, COLLECTIONS, CROSS_HANDS, DEFAULT_DEST, TASK_ID, baseline_entries,
    contained_path, install_file, local_release_path, safe_relative, sha256,
    sync_baseline_links, task_categories,
)

DEFAULT_REPO = "dexverse/DexVerse_release"
REMOTE_ROOT = "demonstrations"
RELEASE_MANIFEST = f"{REMOTE_ROOT}/release_manifest.json"
MAIN_MANIFEST = Path(__file__).with_name("main_demos_manifest.json")
CROSS_20_MANIFEST = Path(__file__).with_name("cross_20_demos_manifest.json")


def _local_task(local: str) -> str:
    parts = local.split("/")
    return "/".join(parts[1:3] if parts[0] in COLLECTIONS else parts[:2])


def _named_collection(local: str) -> bool:
    return local.split("/", 1)[0] in COLLECTIONS


def _main_split(local: str) -> str:
    parts = local.split("/")
    return parts[3] if len(parts) == 5 else "train"


def main_metadata() -> dict[str, dict]:
    """Checksums ship with the downloader so the dataset upload can be PKL-only."""
    manifest = json.loads(MAIN_MANIFEST.read_text())
    result = {}
    tasks = set()
    episodes = 0
    for entry in manifest["files"]:
        path = entry["path"]
        local = local_release_path(path)
        if not local.startswith("main/") or path in result:
            raise ValueError(f"Invalid or duplicate main demo path: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid main demo checksum: {path}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Invalid main demo size: {path}")
        if type(entry["episodes"]) is not int or entry["episodes"] <= 0:
            raise ValueError(f"Invalid main episode count: {path}")
        if entry.get("split") != _main_split(local):
            raise ValueError(f"Main split does not match its directory: {path}")
        result[path] = entry
        tasks.add(_local_task(local))
        episodes += entry["episodes"]
    if len(tasks) != manifest["tasks"] or episodes != manifest["episodes"]:
        raise ValueError("Main demo manifest totals do not match its files")
    return result


def _load_baseline_manifest() -> list[str]:
    return baseline_entries()


def cross_20_metadata() -> dict[str, dict]:
    """Require the complete 7-task, 6-hand, 20-episode matrix and PKL checksums."""
    manifest = json.loads(CROSS_20_MANIFEST.read_text())
    result, pairs = {}, Counter()
    for entry in manifest["files"]:
        path = entry["path"]
        local = local_release_path(path)
        if not local.startswith("cross_20/") or path in result:
            raise ValueError(f"Invalid or duplicate cross_20 demo path: {path}")
        _, category, task, robot, filename = local.split("/")
        if entry["task"] != task or entry["robot_type"] != robot or not task.endswith("-v0"):
            raise ValueError(f"Cross_20 task/hand does not match its directory: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid cross_20 demo checksum: {path}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Invalid cross_20 demo size: {path}")
        if type(entry["episodes"]) is not int or entry["episodes"] <= 0:
            raise ValueError(f"Invalid cross_20 episode count: {path}")
        result[path] = entry
        pairs[(task, robot)] += entry["episodes"]
    tasks = {task for task, robot in pairs}
    required = {(task, f"floating_{hand}_right") for task in tasks for hand in CROSS_HANDS}
    if (len(tasks) != 7 or set(pairs) != required or set(pairs.values()) != {20}
            or manifest["tasks"] != 7 or manifest["episodes"] != 840
            or manifest["task_hand_pairs"] != 42 or manifest["episodes_per_task_hand"] != 20
            or set(manifest["hands"]) != CROSS_HANDS):
        raise ValueError("Cross_20 manifest must contain 7 tasks x 6 hands x 20 episodes")
    return result


def _task_selector(value: str, version: str) -> str:
    parts = list(safe_relative(value).parts)
    if parts[0] in {"v0", "v1"}:
        explicit = parts.pop(0)
        if version not in {"all", explicit}:
            raise ValueError(f"Task version conflicts with --version: {value}")
        if not parts or not parts[-1].endswith("-" + explicit):
            raise ValueError(f"Version disagrees with task ID: {value}")
    if len(parts) == 1 and TASK_ID.fullmatch(parts[0]):
        family = re.sub(r"-v\d+$", "", parts[0]).removeprefix("Dexverse-")
        category = task_categories().get(family)
        if category is None:
            raise ValueError(f"Unknown task {value}; supply category/Dexverse-Task-v0")
        parts.insert(0, category)
    if len(parts) != 2 or parts[0] not in CATEGORIES or not TASK_ID.fullmatch(parts[1]):
        raise ValueError(f"Expected category/Dexverse-Task-v0: {value}")
    if version != "all" and not parts[1].endswith("-" + version):
        raise ValueError(f"Task version conflicts with --version: {value}")
    return "/".join(parts)


def select_files(files: list[str], args) -> list[dict]:
    baseline = set(baseline_entries(args.version)) if args.baseline else set()
    tasks = {_task_selector(t, args.version) for t in args.task or []}
    categories = set(args.category or [])
    if not categories <= CATEGORIES:
        raise ValueError(f"Unknown categories: {sorted(categories - CATEGORIES)}")
    for pattern in args.pattern or []:
        if not pattern.startswith("demonstrations/") or ".." in pattern.split("/") or "\\" in pattern:
            raise ValueError("Patterns must stay under demonstrations/")
    selected, targets = [], {}
    for remote in sorted(files):
        if not remote.startswith("demonstrations/"):
            continue
        try:
            local = local_release_path(remote)
        except ValueError:
            # Root manifests and legacy baseline copies are not canonical data.
            continue
        key = _local_task(local)
        category, task = key.split("/")
        if args.version != "all" and not task.endswith("-" + args.version):
            continue
        if not (args.all or (getattr(args, "main", False) and local.startswith("main/") and _main_split(local) == "train")
                or (getattr(args, "cross_20", False) and local.startswith("cross_20/"))
                or (key in baseline and not _named_collection(local)) or key in tasks or category in categories
                or any(fnmatch.fnmatchcase(remote, p) for p in args.pattern or [])):
            continue
        if local in targets:
            raise ValueError(f"Multiple remote files map to {local}: {targets[local]}, {remote}")
        targets[local] = remote
        selected.append({"path": remote, "local_path": local})
    found_tasks = {_local_task(e["local_path"]) for e in selected}
    absent = tasks - found_tasks
    if absent:
        raise ValueError(f"No remote demonstration files for: {', '.join(sorted(absent))}")
    return selected


def release_metadata(manifest: dict) -> dict[str, dict]:
    """Validate published checksums, allowing future tasks beyond the baseline."""
    result = {}
    for task in manifest.get("tasks", []):
        for entry in task.get("files", []):
            path = entry["path"]
            local_release_path(path)
            digest = entry["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid release checksum: {path}")
            if type(entry["bytes"]) is not int or entry["bytes"] <= 0 or path in result:
                raise ValueError(f"Invalid or duplicate release entry: {path}")
            result[path] = entry
    return result


def fetch_selected(entries: list[dict], dest: Path, fetch, metadata: dict | None = None) -> None:
    metadata = metadata or {}
    for entry in entries:
        target = contained_path(dest, entry["local_path"])
        proof = metadata.get(entry["path"], {})
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise FileExistsError(f"Refusing to replace {target}")
            if proof:
                if target.stat().st_size != proof["bytes"] or sha256(target) != proof["sha256"]:
                    raise FileExistsError(f"Different local data exists: {target}. Use a separate --dest.")
                print(f"Already verified: {target}")
                continue
        cached = Path(fetch(entry["path"]))
        if proof and cached.stat().st_size != proof["bytes"]:
            raise ValueError(f"Downloaded size mismatch: {entry['path']}")
        install_file(cached, target, proof.get("sha256"))
        print(f"Installed: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--repo-type", default="dataset", choices=("dataset", "model", "space"))
    parser.add_argument("--revision", default="main", help="Remote branch, tag, or commit")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Download the baseline tasks into their categories")
    parser.add_argument("--main", action="store_true", help="Download all 42 approved v0 Shadow state-demo tasks into <dest>/main")
    parser.add_argument("--cross_20", action="store_true", help="Download 7 v0 tasks with 20 state demos per hand (6 hands, 840 episodes) into <dest>/cross_20")
    parser.add_argument("--version", choices=("v0", "v1", "all"), default="v0")
    parser.add_argument("--category", action="append", help="Category name; repeatable")
    parser.add_argument("--task", action="append", help="category/Dexverse-Task-v0 or a registered task ID; repeatable")
    parser.add_argument("--pattern", action="append", help="Additional remote demonstrations/ glob; repeatable")
    parser.add_argument("--list", dest="list_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--link-baseline-only", action="store_true", help="Create links to locally installed demos; no network")
    args = parser.parse_args()
    args.dest = args.dest.expanduser().resolve()
    try:
        if args.link_baseline_only:
            if any((args.all, args.baseline, args.main, args.cross_20, args.category, args.task, args.pattern, args.list_only, args.dry_run)):
                parser.error("--link-baseline-only cannot be combined with download/list/dry-run options")
            result = sync_baseline_links(args.dest, baseline_entries(args.version))
            print(f"Baseline links: {len(result['links'])}; unavailable tasks: {len(result['missing'])}")
            for entry in result["missing"]:
                print(f"Unavailable locally: {entry}")
            return 0
        if not any((args.all, args.baseline, args.main, args.cross_20, args.category, args.task, args.pattern, args.list_only)):
            parser.error("Select --all, --baseline, --main, --cross_20, --category, --task, --pattern, --list, or --link-baseline-only")
        if args.main and args.version == "v1":
            parser.error("The main Shadow collection contains v0 tasks; use --version v0 or all")
        if args.cross_20 and args.version == "v1":
            parser.error("The cross_20 collection contains v0 tasks; use --version v0 or all")
        # Validate selectors before accessing the network.
        for task in args.task or []:
            _task_selector(task, args.version)
        if not set(args.category or []) <= CATEGORIES:
            parser.error(f"Categories: {', '.join(sorted(CATEGORIES))}")
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub to download demos; --link-baseline-only works offline.") from exc
        # The release is public; a stale local login must not block its downloads.
        token = False if args.repo == DEFAULT_REPO and args.repo_type == "dataset" else None
        api = HfApi(token=token)
        revision = api.repo_info(repo_id=args.repo, repo_type=args.repo_type, revision=args.revision).sha
        files = api.list_repo_files(repo_id=args.repo, repo_type=args.repo_type, revision=revision)
        if args.list_only and not any((args.main, args.cross_20, args.all, args.baseline, args.category, args.task, args.pattern)):
            for remote in sorted(files):
                try:
                    local = local_release_path(remote)
                except ValueError:
                    continue
                if args.version == "all" or _local_task(local).split("/")[1].endswith("-" + args.version):
                    print(f"{remote} -> {local}")
            return 0
        entries = select_files(files, args)
        main_proof = main_metadata() if any(e["local_path"].startswith("main/") for e in entries) or args.main else {}
        unknown_main = {e["path"] for e in entries if e["local_path"].startswith("main/")} - set(main_proof)
        if unknown_main:
            raise ValueError("Main demo checksums are unavailable; update this checkout's main_demos_manifest.json")
        if args.main:
            required = {path for path in main_proof if _main_split(local_release_path(path)) == "train"}
            missing = required - {e["path"] for e in entries}
            if missing:
                raise ValueError(f"Main collection is incomplete at {revision}: {len(missing)} required PKLs missing")
        cross_proof = cross_20_metadata() if any(e["local_path"].startswith("cross_20/") for e in entries) or args.cross_20 else {}
        unknown_cross = {e["path"] for e in entries if e["local_path"].startswith("cross_20/")} - set(cross_proof)
        if unknown_cross:
            raise ValueError("Cross_20 demo checksums are unavailable; update this checkout's cross_20_demos_manifest.json")
        if args.cross_20:
            missing = set(cross_proof) - {e["path"] for e in entries}
            if missing:
                raise ValueError(f"Cross_20 collection is incomplete at {revision}: {len(missing)} required PKLs missing")
        print(f"Repo: {args.repo}@{revision}\nDestination: {args.dest}\nSelected files: {len(entries)}")
        if args.baseline:
            present = {_local_task(e["local_path"]) for e in entries
                       if e["path"].endswith(".pkl") and not _named_collection(e["local_path"])}
            for entry in sorted(set(baseline_entries(args.version)) - present):
                print(f"Unavailable baseline task on remote: {entry}")
        if not entries:
            raise ValueError("No demonstration files match the selection")
        if args.dry_run or args.list_only:
            for entry in entries:
                print(f"{entry['path']} -> {entry['local_path']}")
            return 0

        def fetch(name):
            return hf_hub_download(repo_id=args.repo, repo_type=args.repo_type, revision=revision,
                                   filename=name, cache_dir=str(args.cache_dir) if args.cache_dir else None,
                                   token=token)

        metadata = release_metadata(json.loads(Path(fetch(RELEASE_MANIFEST)).read_text())) if RELEASE_MANIFEST in files else {}
        metadata.update(main_proof)
        metadata.update(cross_proof)
        fetch_selected(entries, args.dest, fetch, metadata)
        if any(not _named_collection(e["local_path"]) for e in entries):
            links = sync_baseline_links(args.dest)
            print(f"Done. Category files at {args.dest}; {len(links['links'])} baseline symlinks.")
        if any(e["local_path"].startswith("main/") for e in entries):
            print(f"Main Shadow state demos: {args.dest / 'main'}. Replay with --set-state.")
        if any(e["local_path"].startswith("cross_20/") for e in entries):
            print(f"Cross_20 state demos: {args.dest / 'cross_20'}. Replay with --set-state.")
        return 0
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
