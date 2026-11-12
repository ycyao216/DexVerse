# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Download versioned demos without overwriting differing local recordings.

  python scripts/demo_tools/download_demos.py --repo OWNER/DATASET --baseline
  python scripts/demo_tools/download_demos.py --baseline --version v1 --dry-run
  python scripts/demo_tools/download_demos.py --category functional --version v0

--baseline selects BOTH v0 and v1 by default, using the remote release manifest.
Absent/skipped tasks are reported, never silently substituted with another version.
For older unversioned datasets, use --legacy explicitly (v0 only). No pickle is
unpickled during download. The default release is public and needs no login.
Authenticate with `hf auth login` only when using a gated/private --repo override.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from demo_release import DEMOS, MANIFEST, baseline_entries, baseline_key, contained_path, install_file, read_manifest, safe_relative, sha256

DEFAULT_REPO = "dexverse/DexVerse_release"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_DEST = DEMOS
REMOTE_ROOT = "demonstrations"
BASELINE_MANIFEST = DEMOS / "baseline_manifest.txt"


def _load_baseline_manifest(version="all"):
    return baseline_entries(version)


def _build_patterns(args):
    """Unversioned selectors expand into both directories; explicit ones stay explicit."""
    version = getattr(args, "version", "all")
    legacy = getattr(args, "legacy", False)
    versions = ("v0", "v1") if version == "all" else (version,)
    patterns = []
    if legacy and version == "v1":
        raise ValueError("Legacy layout has no v1 demos")
    if args.all:
        patterns.extend(["demonstrations/**"] if legacy else [f"demonstrations/{v}/**" for v in versions])
    if args.baseline:
        for entry in baseline_entries("v0" if legacy else version):
            patterns.append(f"demonstrations/{entry.split('/', 1)[1] if legacy else entry}/**")
    for selector in args.task or []:
        if not legacy:
            key = baseline_key(selector, version)
            patterns.append(f"demonstrations/{key}/**")
        else:
            safe_relative(selector)
            if selector.endswith("-v1") or selector.split("/")[0] in {"v0", "v1"}:
                raise ValueError("Legacy layout only supports unversioned v0 paths")
            if "/" not in selector:
                selector = baseline_key(selector, "v0").split("/", 1)[1]
            patterns.append(f"demonstrations/{selector}/**")
    for selector in args.category or []:
        safe_relative(selector)
        if selector.split("/")[0] in {"v0", "v1"}:
            if legacy:
                raise ValueError("Do not use a version prefix with --legacy")
            if selector.split("/")[0] not in versions:
                raise ValueError("Explicit task version conflicts with --version")
            patterns.append(f"demonstrations/{selector}/**")
        else:
            selected_versions = versions
            if selector.endswith(("-v0", "-v1")):
                selected_versions = tuple(v for v in versions if selector.endswith("-" + v))
            patterns.extend([f"demonstrations/{selector}/**"] if legacy else
                            [f"demonstrations/{v}/{selector}/**" for v in selected_versions])
    for pattern in args.pattern or []:
        if not pattern.startswith("demonstrations/") or ".." in pattern.split("/") or "\\" in pattern:
            raise ValueError("Patterns must stay under demonstrations/")
        patterns.append(pattern)
    return list(dict.fromkeys(patterns))


def select_release(manifest, patterns):
    files, unavailable = [], []
    for task in manifest["tasks"]:
        probe = f"demonstrations/{task['key']}/demos.pkl"
        if not any(fnmatch.fnmatchcase(probe, pattern) for pattern in patterns):
            continue
        files.extend(task["files"])
        if task["status"] != "complete":
            unavailable.append(task)
    return files, unavailable


def fetch_selected(entries, dest, fetch):
    """Checksum validate cached downloads and atomically install without clobbering."""
    for entry in entries:
        rel = safe_relative(entry["path"])
        if rel.parts[0] != REMOTE_ROOT:
            raise ValueError("Remote entry outside demonstrations/")
        target = contained_path(dest, str(rel.relative_to(REMOTE_ROOT)))
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise FileExistsError(f"Refusing to replace {target}")
            if entry.get("sha256"):
                if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                    raise FileExistsError(f"Different local file exists: {target}; choose another --dest")
                print(f"Already verified: {target}")
                continue
        cached = Path(fetch(entry["path"]))
        if entry.get("bytes") is not None and cached.stat().st_size != entry["bytes"]:
            raise ValueError(f"Downloaded size mismatch: {entry['path']}")
        install_file(cached, target, entry.get("sha256"))
        print(f"Installed: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument("--revision", default="main", help="Branch, tag, or pinned commit")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--baseline", action="store_true", help="Both baseline versions by default")
    parser.add_argument("--version", choices=("all", "v0", "v1"), default="all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--category", action="append")
    parser.add_argument("--task", action="append", help="Versioned ID (Dexverse-PushT-v1) or [v0|v1/]<category>/<task>; repeatable")
    parser.add_argument("--pattern", action="append", help="demonstrations/ glob; restricted to manifest files")
    parser.add_argument("--list", dest="list_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read remote inventory, no demo downloads")
    parser.add_argument("--legacy", action="store_true", help="Explicit old unversioned v0 dataset layout")
    parser.add_argument("--require-complete", action="store_true", help="Fail before fetching if selected tasks are missing/partial/skipped")
    parser.add_argument("--main", action="store_true", help="Download the approved v0 Shadow state-demo collection")
    parser.add_argument("--cross_20", action="store_true", help="Download the six-hand, seven-task state-demo collection")
    parser.add_argument("--link-baseline-only", action="store_true", help="Create baseline links for existing category-layout demos, offline")
    args = parser.parse_args()
    if args.main or args.cross_20 or args.link_baseline_only:
        if any((args.baseline, args.all, args.category, args.task, args.pattern, args.legacy, args.require_complete)):
            parser.error("Download named collections separately from baseline/versioned selections")
        from download_collections import main as download_collections
        return download_collections()
    try:
        patterns = _build_patterns(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not patterns and not args.list_only:
        parser.error("Select --baseline, --all, --category, --task, --pattern, or --list")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    # Pin inventory and downloads to the same commit even when a branch advances.
    revision = api.repo_info(repo_id=args.repo, repo_type=args.repo_type, revision=args.revision).sha
    def fetch(name):
        return hf_hub_download(repo_id=args.repo, repo_type=args.repo_type, revision=revision,
                               filename=name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if args.legacy:
        if args.require_complete:
            parser.error("--require-complete needs the versioned release manifest")
        paths = api.list_repo_files(repo_id=args.repo, repo_type=args.repo_type, revision=revision)
        paths = [p for p in paths if p.startswith("demonstrations/") and p.split("/")[1] not in {"v0", "v1", "_review"}]
        if args.list_only:
            print("\n".join(paths))
            return 0
        entries = [{"path": p} for p in paths if any(fnmatch.fnmatchcase(p, pat) for pat in patterns)]
        unavailable = []
        print("Legacy layout: only unversioned v0; release hashes and v1 coverage unavailable.")
    else:
        from huggingface_hub.errors import EntryNotFoundError
        try:
            manifest = read_manifest(fetch(MANIFEST))
        except EntryNotFoundError as exc:
            raise SystemExit("Remote has no versioned release manifest. Upload a prepared release first, or use --legacy for the old v0-only dataset.") from exc
        if args.list_only:
            for task in manifest["tasks"]:
                print(f"{task['key']}: {task['status']} ({task['episodes']})")
            return 0
        entries, unavailable = select_release(manifest, patterns)
    print(f"Dataset {args.repo}@{revision}: {len(entries)} selected files")
    for task in unavailable:
        print(f"Unavailable/incomplete: {task['key']} ({task['status']}, {task['episodes']}/50). {task['reason']}")
    if unavailable and args.require_complete:
        raise SystemExit("Selected baseline set is incomplete; nothing downloaded")
    if not entries:
        raise SystemExit("No demo files match this selection")
    if args.dry_run:
        print("\n".join(entry["path"] for entry in entries))
        return 0
    fetch_selected(entries, args.dest, fetch)
    # A remote manifest is not installed as local coverage: selection may be partial.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
