# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pull ``robot_agents/`` binary asset bundles from the DexVerse Hugging Face
dataset and extract them back into the in-package ``robot_agents/`` directory.

The bundles live under the
``robot_agents/`` prefix of ``dexverse/DexVerse_release`` as one ``<subdir>.zip``
per hand directory (Shadow, Allegro, Inspire, LEAP, Sharpa, Wuji). The
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

The dataset is public; no Hugging Face login is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_REPO = "dexverse/DexVerse_release"
DEFAULT_REPO_TYPE = "dataset"
REMOTE_PREFIX = "robot_agents"
MANIFEST_NAME = "MANIFEST.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_AGENTS_ROOT = REPO_ROOT / "source" / "dexverse" / "dexverse" / "robot_agents"
REQUIREMENTS = Path(__file__).with_name("robot_asset_requirements.json")


def _load_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed. Run `pip install huggingface_hub`.") from exc
    return HfApi, hf_hub_download


def _download(filename: str, *, repo: str, repo_type: str, cache_dir: Path | None, revision: str) -> Path:
    _, hf_hub_download = _load_hf()
    remote = f"{REMOTE_PREFIX}/{filename}"
    print(f"  downloading {remote} from {repo} ...", flush=True)
    local = hf_hub_download(
        repo_id=repo,
        repo_type=repo_type,
        filename=remote,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def _remote_bundles(repo: str, repo_type: str, revision: str) -> list[str]:
    """Bundle names (without .zip) found under robot_agents/ on the remote."""
    HfApi, _ = _load_hf()
    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type=repo_type, revision=revision)
    prefix = REMOTE_PREFIX + "/"
    return sorted(f[len(prefix) : -len(".zip")] for f in files if f.startswith(prefix) and f.endswith(".zip"))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_bundle(zip_path: Path, bundle: str, metadata: dict, requirements: dict | None = None) -> list[str]:
    """Check every member before writing anything, including required physics fixes."""
    if metadata.get("sha256") and _sha256(zip_path) != metadata["sha256"]:
        raise ValueError(f"Checksum mismatch for {bundle}.zip; nothing extracted.")
    if requirements is None:
        requirements = json.loads(REQUIREMENTS.read_text())["bundles"]
    with zipfile.ZipFile(zip_path) as zf:
        names = []
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != bundle or "\\" in info.filename
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Invalid asset archive member: {info.filename}")
            if info.is_dir():
                continue
            if path.suffix in (".py", ".pyc", ".yaml", ".yml") or "__pycache__" in path.parts:
                raise ValueError(f"Asset archive must not overwrite Git configuration: {info.filename}")
            names.append(info.filename)
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate members in {bundle}.zip")
        for name, expected in requirements.get(bundle, {}).get("required_files", {}).items():
            if name not in names or hashlib.sha256(zf.read(name)).hexdigest() != expected:
                raise ValueError(
                    f"{bundle}.zip is incompatible with this checkout: missing or different {name}. "
                    "Use the corrected robot-asset release described in docs/embodiment_release.md. "
                    "Nothing from this bundle was extracted.")
        bad = zf.testzip()
        if bad:
            raise ValueError(f"Corrupt asset archive member: {bad}")
        return names


def _extract(zip_path: Path, dest_root: Path, names: list[str]) -> None:
    print(f"  extracting {zip_path.name} -> {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=True)
    # Resolve all paths first: an existing symlink must not redirect extraction.
    for name in names:
        if not (dest_root / name).resolve().is_relative_to(dest_root.resolve()):
            raise ValueError(f"Asset destination escapes extraction root: {name}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in names:
            dest = dest_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument("--revision", help="HF commit, tag, or branch; defaults to the matching release pin, or main before publication.")
    parser.add_argument("--local-bundles", type=Path, help="Read staged ZIPs and MANIFEST.json locally for release verification.")
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

    if args.local_bundles:
        available = sorted(p.stem for p in args.local_bundles.glob("*.zip"))
        manifest = json.loads((args.local_bundles / MANIFEST_NAME).read_text())
        revision = "local staged release"
    else:
        HfApi, _ = _load_hf()
        requirements = json.loads(REQUIREMENTS.read_text())
        pin = requirements.get("hf_revision") if args.repo == requirements.get("repo_id") else None
        requested_revision = args.revision or pin or "main"
        revision = HfApi().repo_info(repo_id=args.repo, repo_type=args.repo_type, revision=requested_revision).sha
        available = _remote_bundles(args.repo, args.repo_type, revision)
        path = _download(MANIFEST_NAME, repo=args.repo, repo_type=args.repo_type,
                         cache_dir=args.cache_dir, revision=revision)
        manifest = json.loads(path.read_text())

    if args.list_only:
        info = manifest.get("bundles", {})
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
    print(f"Revision:    {revision}")
    print(f"Extract to:  {args.extract_to}")
    print(f"Bundles:     {', '.join(sorted(wanted))}")

    if args.dry_run:
        print("\n[dry-run] would download:")
        for name in sorted(wanted):
            print(f"  {REMOTE_PREFIX}/{name}.zip")
        return 0

    prepared = []
    for name in sorted(wanted):
        zip_local = (args.local_bundles / f"{name}.zip" if args.local_bundles else
                     _download(f"{name}.zip", repo=args.repo, repo_type=args.repo_type,
                               cache_dir=args.cache_dir, revision=revision))
        metadata = manifest["bundles"][name]
        members = _validate_bundle(zip_local, name, metadata)
        prepared.append((zip_local, members))
        if args.no_extract:
            print(f"  [no-extract] verified at {zip_local}")
    if not args.no_extract:
        for zip_local, members in prepared:
            _extract(zip_local, args.extract_to, members)

    print(f"\nDone. Assets at: {args.extract_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
