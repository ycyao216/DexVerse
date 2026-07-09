# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync teleoperation demonstrations from the DexVerse Hugging Face
dataset into the in-repo ``demonstrations/`` directory.

The HF repo ``dexverse/DexVerse_Dataset`` keeps demos under
``demonstrations/<category>/<task>/...``. This script mirrors that tree
into ``source/dexverse/demonstrations/`` (or a path of your choosing).

Examples
--------
    # Download everything under demonstrations/
    python scripts/demo_tools/download_demos.py --all

    # Only the rigid category
    python scripts/demo_tools/download_demos.py --category rigid

    # A specific task
    python scripts/demo_tools/download_demos.py --task contact_rich/Dexverse-PlugCharger-v0

    # The curated baseline demo set (tasks listed in baseline_manifest.txt,
    # each fetched from its own real <category>/<task> location -- no
    # separate "baseline" category, so a later --all won't re-download them)
    python scripts/demo_tools/download_demos.py --baseline

    # Just list what is available on the remote
    python scripts/demo_tools/download_demos.py --list

The dataset is gated. Run ``huggingface-cli login`` once (and accept the
terms on the dataset page in your browser) before using this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "dexverse/DexVerse_Dataset"
DEFAULT_REPO_TYPE = "dataset"
REMOTE_ROOT = "demonstrations"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "source" / "dexverse" / "demonstrations"
# The curated baseline set is a list of <category>/<task> entries (see the
# file for details), each a real category/task location -- not a separate
# flat "baseline" category. Keeps the remote free of duplicate copies and
# lets a later --all skip files --baseline already fetched.
BASELINE_MANIFEST = DEFAULT_DEST / "baseline_manifest.txt"


def _load_baseline_manifest() -> list[str]:
    if not BASELINE_MANIFEST.is_file():
        raise SystemExit(f"--baseline manifest not found: {BASELINE_MANIFEST}")
    entries = []
    for line in BASELINE_MANIFEST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line.strip("/"))
    return entries


def _load_hf():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed. Run `pip install huggingface_hub`.") from exc
    return HfApi, snapshot_download


def _list_remote(repo: str, repo_type: str) -> list[str]:
    HfApi, _ = _load_hf()
    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type=repo_type)
    return [f for f in files if f.startswith(REMOTE_ROOT + "/")]


def _print_listing(files: list[str]) -> None:
    if not files:
        print(f"No files found under {REMOTE_ROOT}/ on the remote.")
        return

    categories: dict[str, dict[str, int]] = {}
    for f in files:
        parts = f.split("/")
        if len(parts) < 3:
            continue
        category = parts[1]
        task = parts[2] if len(parts) >= 4 else "<root>"
        categories.setdefault(category, {}).setdefault(task, 0)
        categories[category][task] += 1

    print(f"Available demos under {REMOTE_ROOT}/ in remote repo:")
    for category in sorted(categories):
        print(f"  {category}/")
        for task, count in sorted(categories[category].items()):
            print(f"    {task}/  ({count} file{'s' if count != 1 else ''})")


def _build_patterns(args) -> list[str]:
    patterns: list[str] = []
    if args.all:
        patterns.append(f"{REMOTE_ROOT}/**")
    if args.baseline:
        for entry in _load_baseline_manifest():
            patterns.append(f"{REMOTE_ROOT}/{entry}/**")
    for cat in args.category or []:
        patterns.append(f"{REMOTE_ROOT}/{cat.strip('/')}/**")
    for task in args.task or []:
        patterns.append(f"{REMOTE_ROOT}/{task.strip('/')}/**")
    for pat in args.pattern or []:
        patterns.append(pat)
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Local destination directory (default: source/dexverse/demonstrations)",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir override.")
    parser.add_argument("--all", action="store_true", help=f"Download everything under {REMOTE_ROOT}/.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Download the curated baseline demo set (the tasks "
            f"listed in {BASELINE_MANIFEST.name}), each from its "
            "own real <category>/<task> location."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Download a category subdir, e.g. `--category rigid` or `--category articulation`. Repeatable.",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Download a specific task, e.g. `--task contact_rich/Dexverse-PlugCharger-v0`. Repeatable.",
    )
    parser.add_argument("--pattern", action="append", help="Extra HF allow_patterns glob (advanced). Repeatable.")
    parser.add_argument(
        "--list", dest="list_only", action="store_true", help="List what's available on the remote and exit."
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without fetching.")
    args = parser.parse_args()

    if args.list_only:
        files = _list_remote(args.repo, args.repo_type)
        _print_listing(files)
        return 0

    patterns = _build_patterns(args)
    if not patterns:
        parser.error(
            "Nothing selected. Use --all, --baseline, --category <name>, "
            "--task <cat/name>, --pattern <glob>, or --list."
        )

    print(f"Repo:        {args.repo} ({args.repo_type})")
    print(f"Destination: {args.dest}")
    print("Patterns:")
    for p in patterns:
        print(f"  {p}")

    if args.dry_run:
        files = _list_remote(args.repo, args.repo_type)
        import fnmatch

        matched = sorted({f for f in files for p in patterns if fnmatch.fnmatch(f, p)})
        print(f"\n[dry-run] {len(matched)} file(s) would be downloaded:")
        for f in matched:
            print(f"  {f}")
        return 0

    _, snapshot_download = _load_hf()
    args.dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        allow_patterns=patterns,
        local_dir=str(args.dest),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    # snapshot_download mirrors the repo paths, so files arrive at
    # `<dest>/demonstrations/...`. Promote them up to `<dest>/...` so the
    # caller's dest contains the categories directly. Promote per-file so
    # newly fetched tasks merge into pre-existing category dirs instead of
    # being silently stranded under `<dest>/demonstrations/`.
    import shutil

    nested = args.dest / REMOTE_ROOT
    if nested.is_dir():
        for src_file in nested.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(nested)
            dst_file = args.dest / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            src_file.rename(dst_file)
        shutil.rmtree(nested, ignore_errors=True)

    print(f"\nDone. Demos at: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
