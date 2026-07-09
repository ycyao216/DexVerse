# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pull ``robot_agents/`` binary asset bundles from the DexVerse Hugging Face
dataset and extract them back into the in-package ``robot_agents/`` directory.

Inverse of ``upload_robot_agents.py``. The bundles live under the
``robot_agents/`` prefix of ``dexverse/DexVerse_Dataset`` as one ``<subdir>.zip``
per hand directory (currently: shadow), each rooted at ``robot_agents/``. The
text sources for these agents (Python + yaml configs) are tracked in git, so
this script only restores the large binary assets (USD, URDF, meshes). Each
zip's arcnames are package-relative (``shadow/floating_shadow_right/...``), so
extraction drops every asset straight into the directory its ``floating.py``
references — no post-processing needed for the robot agents to load.

Examples
--------
    # Everything
    python scripts/asset_tools/download_robot_agents.py --all

    # A specific hand bundle
    python scripts/asset_tools/download_robot_agents.py --bundle shadow

    # List what's available on the remote
    python scripts/asset_tools/download_robot_agents.py --list

The dataset is gated. Run ``huggingface-cli login`` once (and accept the terms
on the dataset page) before using this script.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

DEFAULT_REPO = "dexverse/DexVerse_Dataset"
DEFAULT_REPO_TYPE = "dataset"
REMOTE_PREFIX = "robot_agents"
MANIFEST_NAME = "MANIFEST.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_AGENTS_ROOT = REPO_ROOT / "source" / "dexverse" / "dexverse" / "robot_agents"


def _load_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed. Run `pip install huggingface_hub`.") from exc
    return HfApi, hf_hub_download


def _download(filename: str, *, repo: str, repo_type: str, cache_dir: Path | None) -> Path:
    _, hf_hub_download = _load_hf()
    remote = f"{REMOTE_PREFIX}/{filename}"
    print(f"  downloading {remote} from {repo} ...", flush=True)
    local = hf_hub_download(
        repo_id=repo,
        repo_type=repo_type,
        filename=remote,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def _remote_bundles(repo: str, repo_type: str) -> list[str]:
    """Bundle names (without .zip) found under robot_agents/ on the remote."""
    HfApi, _ = _load_hf()
    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type=repo_type)
    prefix = REMOTE_PREFIX + "/"
    return sorted(f[len(prefix) : -len(".zip")] for f in files if f.startswith(prefix) and f.endswith(".zip"))


def _load_manifest(repo: str, repo_type: str, cache_dir: Path | None) -> dict | None:
    try:
        path = _download(MANIFEST_NAME, repo=repo, repo_type=repo_type, cache_dir=cache_dir)
    except Exception:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract(zip_path: Path, dest_root: Path) -> None:
    print(f"  extracting {zip_path.name} -> {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument("--all", action="store_true", help="Download every bundle on the remote.")
    parser.add_argument(
        "--bundle",
        action="append",
        help="Download a single bundle by name (e.g. `--bundle leap`). Repeatable. Run --list to see names.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        default=ROBOT_AGENTS_ROOT,
        help="Where bundles extract to (default: package robot_agents dir).",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir override.")
    parser.add_argument("--no-extract", action="store_true", help="Download zips but skip extraction.")
    parser.add_argument(
        "--list", dest="list_only", action="store_true", help="List bundles available on the remote and exit."
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without fetching.")
    args = parser.parse_args()

    available = _remote_bundles(args.repo, args.repo_type)

    if args.list_only:
        manifest = _load_manifest(args.repo, args.repo_type, args.cache_dir)
        info = (manifest or {}).get("bundles", {})
        if not available:
            print(f"No bundles found under {REMOTE_PREFIX}/ on {args.repo}.")
            return 0
        print(f"Bundles available under {REMOTE_PREFIX}/ on {args.repo}:")
        for name in available:
            meta = info.get(name, {})
            size = meta.get("size_bytes_compressed")
            size_str = f"  ({size / 1e6:.1f} MB)" if size else ""
            count = meta.get("file_count")
            count_str = f"  {count} files" if count else ""
            print(f"  {name}{count_str}{size_str}")
        return 0

    if args.bundle:
        wanted = {b.strip("/") for b in args.bundle}
        unknown = wanted - set(available)
        if unknown:
            raise SystemExit(
                f"Unknown bundle(s): {', '.join(sorted(unknown))}. Available: {', '.join(available) or '(none)'}"
            )
    elif args.all:
        wanted = set(available)
    else:
        parser.error("Nothing selected. Pass --all, --bundle <name>, or --list.")

    if not wanted:
        print(f"No bundles to download under {REMOTE_PREFIX}/.")
        return 0

    print(f"Repo:        {args.repo} ({args.repo_type})")
    print(f"Extract to:  {args.extract_to}")
    print(f"Bundles:     {', '.join(sorted(wanted))}")

    if args.dry_run:
        print("\n[dry-run] would download:")
        for name in sorted(wanted):
            print(f"  {REMOTE_PREFIX}/{name}.zip")
        return 0

    args.extract_to.mkdir(parents=True, exist_ok=True)
    for name in sorted(wanted):
        zip_local = _download(f"{name}.zip", repo=args.repo, repo_type=args.repo_type, cache_dir=args.cache_dir)
        if args.no_extract:
            print(f"  [no-extract] kept at {zip_local}")
            continue
        _extract(zip_local, args.extract_to)

    print(f"\nDone. Assets at: {args.extract_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
