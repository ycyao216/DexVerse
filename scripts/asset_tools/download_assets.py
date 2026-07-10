# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reverse of :mod:`upload_to_hf`: pull asset bundles from Hugging Face
and extract them into the in-package ``assets/`` directory.

By default downloads every redistributed bundle (core + long_horizon_extra +
hdris) and fetches the ManiTwin object pool from its upstream repo. Pass one
or more flags to fetch only a subset:

  --core                core_assets.zip: USD set supporting most environments
                        (~410 MB).
  --mani-twin           ManiTwin-100K object pool used by the default
                        grasping / object-pool tasks (~2.2 GB). NOT
                        redistributed in our HF repo: fetched straight from
                        the upstream ``ManiTwin/ManiTwin-100K`` dataset via
                        download_selected_objects.py.
  --long-horizon-extra  Large meshes used only by long-horizon tasks
                        (cook_foods, trash_drawer_sort_simple).
  --hdris               Polyhaven HDRI environment maps (~1.8 GB).
  --all                 Every bundle plus the ManiTwin pool (same as the
                        no-flag default).

Any other external datasets listed in ``MANIFEST.json`` are not fetched
automatically; this script just prints the helper script named for each.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_REPO = "dexverse/DexVerse_release"
DEFAULT_REPO_TYPE = "dataset"
# Subdir within the HF repo that the asset bundles live under.
REPO_SUBDIR = "assets"

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "source" / "dexverse" / "dexverse" / "assets"

BUNDLE_FILES = {
    "core": "core_assets.zip",
    "long_horizon_extra": "long_horizon_extra.zip",
    "hdris": "polyhaven_hdris.zip",
}
# Fetched from upstream (not our HF repo) via download_selected_objects.py.
MANI_TWIN_KEY = "mani_twin"
ALL_KEYS = set(BUNDLE_FILES) | {MANI_TWIN_KEY}
MANIFEST_FILE = "MANIFEST.json"


def _load_hf_api():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed. Run `pip install huggingface_hub`.") from exc
    return hf_hub_download


def _entry_not_found_error():
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub layouts
        from huggingface_hub.utils import EntryNotFoundError
    return EntryNotFoundError


def _download(filename: str, *, repo: str, repo_type: str, cache_dir: Path | None) -> Path:
    hf_hub_download = _load_hf_api()
    print(f"  downloading {filename} from {repo} ...", flush=True)
    local = hf_hub_download(
        repo_id=repo,
        repo_type=repo_type,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def _extract(zip_path: Path, dest_root: Path) -> None:
    print(f"  extracting {zip_path.name} -> {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_root)


def _fetch_mani_twin(extract_to: Path) -> bool:
    """Fetch the selected ManiTwin-100K objects from the upstream HF dataset.

    The pool is not redistributed in our repo; download_selected_objects.py
    pulls the selected object folders straight from ManiTwin/ManiTwin-100K.
    Returns True on success.
    """
    script = Path(__file__).resolve().with_name("download_selected_objects.py")
    dest = extract_to / "mani_twin_selected"
    print(f"  fetching ManiTwin pool from upstream (ManiTwin/ManiTwin-100K) -> {dest}")
    proc = subprocess.run([sys.executable, str(script), str(dest)])
    if proc.returncode != 0:
        print(f"  !! download_selected_objects.py exited with code {proc.returncode}")
    return proc.returncode == 0


def _print_externals(manifest: dict) -> None:
    externals = manifest.get("external_datasets", [])
    if not externals:
        return
    print("\nExternal datasets (not in this HF repo — fetch separately if needed):")
    for ext in externals:
        size = ext.get("size_estimate_gb")
        size_str = f"  ({size} GB)" if size else ""
        print(f"  - {ext['name']}{size_str}")
        print(f"      extract to: assets/{ext['extract_to']}/")
        print(f"      fetch with: {ext['fetch_with']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument(
        "--core", action="store_true", help="Download core_assets.zip (USD set for most environments, ~410 MB)."
    )
    parser.add_argument(
        "--mani-twin",
        action="store_true",
        dest="mani_twin",
        help=(
            "Fetch the ManiTwin-100K object pool (~2.2 GB) for the "
            "default grasping/object-pool tasks, straight from the "
            "upstream ManiTwin/ManiTwin-100K dataset (not our repo)."
        ),
    )
    parser.add_argument("--hdris", action="store_true", help="Download polyhaven_hdris.zip (~1.8 GB).")
    parser.add_argument(
        "--long-horizon-extra",
        action="store_true",
        dest="long_horizon_extra",
        help="Download long_horizon_extra.zip (large meshes for cook_foods / trash_drawer_sort_simple).",
    )
    parser.add_argument("--all", action="store_true", help="Download every bundle plus the upstream ManiTwin pool.")
    parser.add_argument(
        "--extract-to", type=Path, default=ASSETS_ROOT, help="Where bundles extract to (default: package assets dir)."
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir override.")
    parser.add_argument("--no-extract", action="store_true", help="Download zips but skip extraction.")
    parser.add_argument(
        "--print-externals", action="store_true", help="Print the external-dataset section of MANIFEST.json and exit."
    )
    args = parser.parse_args()

    if args.all:
        wanted = set(ALL_KEYS)
    else:
        wanted = set()
        if args.core:
            wanted.add("core")
        if args.mani_twin:
            wanted.add(MANI_TWIN_KEY)
        if args.hdris:
            wanted.add("hdris")
        if args.long_horizon_extra:
            wanted.add("long_horizon_extra")
    # Default to everything if nothing chosen and not just printing externals.
    if not wanted and not args.print_externals:
        wanted = set(ALL_KEYS)

    # Always pull the manifest so we can show externals.
    manifest_local = _download(
        f"{REPO_SUBDIR}/{MANIFEST_FILE}", repo=args.repo, repo_type=args.repo_type, cache_dir=args.cache_dir
    )
    manifest = json.loads(manifest_local.read_text())

    if args.print_externals:
        _print_externals(manifest)
        return 0

    args.extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting bundles into: {args.extract_to}")

    missing_bundles: list[str] = []
    for key in sorted(wanted - {MANI_TWIN_KEY}):
        filename = f"{REPO_SUBDIR}/{BUNDLE_FILES[key]}"
        try:
            zip_local = _download(filename, repo=args.repo, repo_type=args.repo_type, cache_dir=args.cache_dir)
        except _entry_not_found_error():
            missing_bundles.append(key)
            print(
                f"  !! {BUNDLE_FILES[key]} is not in {args.repo} — skipping. "
                "(Publish it with scripts/asset_tools/archived/zip_for_hf.py + upload_to_hf.py.)"
            )
            continue
        if args.no_extract:
            print(f"  [no-extract] kept at {zip_local}")
            continue
        _extract(zip_local, args.extract_to)

    if MANI_TWIN_KEY in wanted:
        if args.no_extract:
            print(
                "  [no-extract] skipping mani_twin — it is fetched from upstream "
                "as loose object folders, so there is no zip to keep."
            )
        elif not _fetch_mani_twin(args.extract_to):
            missing_bundles.append(MANI_TWIN_KEY)

    # Sanity check: dexverse.assets needs __init__.py to expose path constants
    # (otherwise it becomes a PEP 420 namespace package). It is NOT shipped in
    # the bundles — it is a source-controlled file that comes from the git
    # checkout, so it should already be present at the default extract target.
    if not args.no_extract and ("core" in wanted):
        init_path = args.extract_to / "__init__.py"
        if not init_path.is_file():
            print(
                f"\nWARNING: {init_path} is missing — `from dexverse.assets import ...` "
                "will fail. The bundles do not ship __init__.py; restore it from the git "
                "checkout (it lives at source/dexverse/dexverse/assets/__init__.py) or "
                "extract into a clean git working tree."
            )

    _print_externals(manifest)
    if missing_bundles:
        print(f"\nDone, but {len(missing_bundles)} bundle(s) were unavailable: {', '.join(sorted(missing_bundles))}.")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
