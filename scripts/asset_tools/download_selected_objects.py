#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download selected ManiTwin-100K object folders from the Hugging Face Hub.

This script is the benchmark-facing entry point. It:

1. Downloads the selected object folders from the ``ManiTwin/ManiTwin-100K``
   dataset into a temporary staging directory.
2. Optionally unpacks each ``base_rescaled.usdz`` and patches the Mesh prims
   with ``UsdPhysics.CollisionAPI`` + ``MeshCollisionAPI(approximation=...)``.
3. Moves the processed object folders into the benchmark's asset directory
   (``dexverse/source/dexverse/dexverse/assets/mani_twin_selected``), which
   is the location hard-coded by the environment configs.

The final install directory defaults to that path (resolved relative to this
script), so running the script with no arguments just works inside a
dexverse checkout.

Supported manifest formats
--------------------------
1. A CSV with a ``path`` column whose values look like ``category/instance``
   (e.g. ``adjustment_notches/002``). This matches
   ``selected_100_medium_diverse_seed42.csv``.
2. A CSV (or single-column text file) whose values are just the object name,
   e.g. ``adjustment_notches`` or ``adjustment_notches/002``. In the former
   case, every instance under that category will be downloaded.

Use ``--column`` to force a specific column name.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as exc:
    print(
        "huggingface_hub is required. Install it with `pip install huggingface_hub`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_REPO_ID = "ManiTwin/ManiTwin-100K"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_REPO_SUBDIR = "ManiTwin-100K"
USDZ_FILENAME = "base_rescaled.usdz"
UNPACK_DIRNAME = "base_rescaled_unpacked"
VALID_APPROXIMATIONS = (
    "convexDecomposition",
    "convexHull",
    "boundingSphere",
    "boundingCube",
    "meshSimplification",
    "none",
)

# Default install location for the processed assets, resolved relative to this
# script so the benchmark environments (which hard-code the path) keep working.
# scripts/asset_tools/   -> script_dir
# ../../source/dexverse/dexverse/assets/mani_twin_selected -> install target
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INSTALL_DIR = (
    SCRIPT_DIR / ".." / ".." / "source" / "dexverse" / "dexverse" / "assets" / "mani_twin_selected"
).resolve()

# Reuse the canonical patcher (author_usd_mesh_collision.py) that lives next to
# this script in scripts/asset_tools/ instead of duplicating it inline. That
# directory isn't a Python package, so we extend sys.path.
_ASSET_SCRIPT_DIR = SCRIPT_DIR
if str(_ASSET_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSET_SCRIPT_DIR))


def _resolve_default_csv() -> Path:
    """Look for the manifest CSV next to the script, then in the repo's
    ``mani_twin/manitwin/`` folder as a fallback."""
    candidates = [
        SCRIPT_DIR / "selected_100_medium_diverse_seed42.csv",
        SCRIPT_DIR.parent.parent / "mani_twin" / "manitwin" / "selected_100_medium_diverse_seed42.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def parse_args() -> argparse.Namespace:
    default_csv = _resolve_default_csv()

    parser = argparse.ArgumentParser(
        description="Download selected object folders from the ManiTwin-100K Hugging Face dataset using a manifest CSV."
    )
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=DEFAULT_INSTALL_DIR,
        help=(
            "Final install directory for the processed object folders. "
            f"Defaults to {DEFAULT_INSTALL_DIR} (relative to this script), "
            "which is the location the dexverse task configs expect."
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=default_csv,
        help=f"Manifest CSV to read (default: {default_csv}).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help=(
            "Name of the CSV column that lists objects. If omitted, the script "
            "tries 'path', then 'object', 'instance', 'name', and finally falls "
            "back to the first column."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo id (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default=DEFAULT_REPO_TYPE,
        choices=["dataset", "model", "space"],
        help=f"Hugging Face repo type (default: {DEFAULT_REPO_TYPE}).",
    )
    parser.add_argument(
        "--repo-subdir",
        type=str,
        default=DEFAULT_REPO_SUBDIR,
        help=(
            "Path prefix inside the repo where object folders live "
            f"(default: '{DEFAULT_REPO_SUBDIR}'). Set to '' if your manifest "
            "entries are already relative to the repo root."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Directory where files are downloaded and processed before being "
            "moved into the destination. If omitted, a temporary directory is "
            "created and removed on exit (see --keep-staging)."
        ),
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help=(
            "Keep the staging directory after the run. When the staging "
            "directory is auto-created and this flag is not set, it is deleted."
        ),
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help=(
            "Leave processed objects in the staging directory instead of "
            "moving them into the destination. Useful for dry inspection."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "When moving into the destination, remove any existing instance "
            "folder at the same path first (default: skip existing and warn)."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision / branch / tag to download from.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token. Defaults to the value stored by "
            "`huggingface-cli login` or the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional list of filename patterns to keep within each object folder "
            "(e.g. base_rescaled.glb caption.json). Defaults to all files in the "
            "folder."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download patterns without contacting the Hub.",
    )
    parser.add_argument(
        "--unpack",
        action="store_true",
        help=(
            "After downloading, unpack each base_rescaled.usdz next to it as "
            "base_rescaled_unpacked/ and patch every Mesh prim with "
            "UsdPhysics.CollisionAPI + MeshCollisionAPI(approximation=...). "
            "Requires the `pxr` module (install `usd-core` or use an Isaac Sim env)."
        ),
    )
    parser.add_argument(
        "--approx",
        choices=sorted(VALID_APPROXIMATIONS),
        default="convexDecomposition",
        help="Collision approximation applied by --unpack (default: convexDecomposition).",
    )
    parser.add_argument(
        "--force-unpack",
        action="store_true",
        help="Re-extract and re-patch even if the unpacked folder already exists.",
    )
    parser.add_argument(
        "--keep-usdz",
        action="store_true",
        help=(
            "Keep the original base_rescaled.usdz after unpacking (default is to "
            "keep it; use --no-keep-usdz to remove it)."
        ),
    )
    parser.add_argument(
        "--no-keep-usdz",
        dest="keep_usdz",
        action="store_false",
        help="Delete base_rescaled.usdz after successful unpack+patch.",
    )
    parser.set_defaults(keep_usdz=True)
    return parser.parse_args()


def _pick_column(fieldnames: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column '{explicit}' not in CSV. Available columns: {fieldnames}")
        return explicit

    for candidate in ("path", "object", "instance", "name"):
        if candidate in fieldnames:
            return candidate

    if fieldnames:
        return fieldnames[0]

    raise ValueError("CSV has no columns to read.")


def read_selected_entries(csv_path: Path, column: str | None) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest CSV is empty: {csv_path}")

    header = rows[0]
    known_headers = {"path", "object", "instance", "name", "category", "folder"}
    has_header = column is not None or any(cell.strip().lower() in known_headers for cell in header)

    entries: list[str] = []
    if has_header:
        col_name = _pick_column(header, column)
        col_idx = header.index(col_name)
        data_rows = rows[1:]
    else:
        col_idx = 0
        data_rows = rows

    for row in data_rows:
        if col_idx >= len(row):
            continue
        value = row[col_idx].strip().strip("/")
        if value:
            entries.append(value)

    if not entries:
        raise ValueError(f"No object entries found in {csv_path}")

    # Preserve order but deduplicate.
    seen: set[str] = set()
    unique_entries: list[str] = []
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            unique_entries.append(entry)
    return unique_entries


def build_allow_patterns(
    entries: list[str],
    includes: list[str] | None,
    repo_subdir: str,
) -> list[str]:
    """Build ``allow_patterns`` for ``snapshot_download``.

    - ``entry`` with a slash (``category/instance``) matches that single folder.
    - ``entry`` without a slash (``category``) matches every instance under it.
    - ``includes`` (optional) restricts which filenames are downloaded inside
      each matched folder. If omitted, everything in the folder is fetched.
    - ``repo_subdir`` is prepended to every pattern so it aligns with the
      repository's on-disk layout.
    """
    prefix = repo_subdir.strip("/")
    patterns: list[str] = []

    for entry in entries:
        base = entry.strip("/").replace("\\", "/")
        has_instance = "/" in base
        folder = base if has_instance else f"{base}/*"
        full_folder = f"{prefix}/{folder}" if prefix else folder

        if includes:
            for name in includes:
                patterns.append(f"{full_folder}/{name}")
        else:
            patterns.append(f"{full_folder}/*")

    seen: set[str] = set()
    unique: list[str] = []
    for pat in patterns:
        if pat not in seen:
            seen.add(pat)
            unique.append(pat)
    return unique


@dataclass
class UnpackResult:
    usdz: Path
    unpacked_dir: Path | None = None
    root_usd: Path | None = None
    meshes_patched: int = 0
    skipped_reason: str | None = None
    error: str | None = None

    def ok(self) -> bool:
        return self.error is None and self.skipped_reason is None


def _find_root_usd(folder: Path) -> Path | None:
    """.usdz archives put the root layer near the top. Pick the shallowest
    .usdc/.usda/.usd we can find."""
    if not folder.exists():
        return None
    for ext in (".usdc", ".usda", ".usd"):
        shallow = sorted(folder.glob(f"*{ext}"))
        if shallow:
            return shallow[0]
    for ext in (".usdc", ".usda", ".usd"):
        deep = sorted(folder.rglob(f"*{ext}"))
        if deep:
            return deep[0]
    return None


def _unpack_usdz(usdz: Path, dest: Path, force: bool) -> Path | None:
    existing = _find_root_usd(dest) if dest.exists() else None
    if existing is not None and not force:
        return existing

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(usdz, "r") as zf:
        zf.extractall(dest)

    return _find_root_usd(dest)


def _patch_usd(usd_path: Path, approx: str) -> int:
    """Apply CollisionAPI + MeshCollisionAPI(approximation=approx) to every
    Mesh prim in the stage, saving in place. Returns number of meshes patched.

    Delegates to ``scripts/asset_tools/author_usd_mesh_collision.py`` so the in-place
    authoring logic stays in one place.
    """
    from author_usd_mesh_collision import patch_usd as _author_patch_usd  # lazy

    result = _author_patch_usd(
        usd_path,
        output_usd=None,
        approximation=approx,
        in_place=True,
        overwrite=False,
        flatten=False,
        mesh_regex=None,
        visible_only=True,
        max_convex_hulls=None,
    )
    if result.status == "failed":
        raise RuntimeError(f"author_usd_mesh_collision failed for {usd_path}: {result.message}")
    return int(result.patched_meshes)


def unpack_and_patch_tree(
    root: Path,
    *,
    approx: str,
    force: bool,
    keep_usdz: bool,
) -> list[UnpackResult]:
    """Find every ``base_rescaled.usdz`` under ``root`` and unpack + patch it."""
    usdz_files = sorted(root.rglob(USDZ_FILENAME))
    print(f"\n[unpack] Found {len(usdz_files)} {USDZ_FILENAME} file(s) under {root}")
    if not usdz_files:
        return []

    results: list[UnpackResult] = []
    for i, usdz in enumerate(usdz_files, 1):
        rel = usdz.relative_to(root)
        result = UnpackResult(usdz=usdz)
        unpacked_dir = usdz.parent / UNPACK_DIRNAME
        result.unpacked_dir = unpacked_dir

        try:
            root_usd = _unpack_usdz(usdz, unpacked_dir, force=force)
            if root_usd is None:
                result.error = "No .usd/.usdc/.usda found inside archive"
            else:
                result.root_usd = root_usd
                result.meshes_patched = _patch_usd(root_usd, approx=approx)
                if result.meshes_patched == 0:
                    result.skipped_reason = "no Mesh prims found"
                elif not keep_usdz:
                    with contextlib.suppress(OSError):
                        usdz.unlink()
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"

        results.append(result)

        if result.error:
            status, detail = "ERR ", result.error
        elif result.skipped_reason:
            status, detail = "WARN", result.skipped_reason
        else:
            status, detail = "OK  ", f"meshes={result.meshes_patched}"
        print(f"[unpack {i:3d}/{len(usdz_files)}] {status} {rel}  ({detail})")

    return results


def iter_object_dirs(source_root: Path) -> Iterator[Path]:
    """Yield every ``<category>/<instance>`` directory under ``source_root``."""
    if not source_root.is_dir():
        return
    for category in sorted(source_root.iterdir()):
        if not category.is_dir() or category.name.startswith("."):
            continue
        for instance in sorted(category.iterdir()):
            if instance.is_dir():
                yield instance


def move_objects_into_destination(
    source_root: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> tuple[int, int]:
    """Move every ``category/instance`` folder from ``source_root`` into
    ``destination``, preserving sibling instances that already exist there.

    Returns (moved, skipped).
    """
    moved = 0
    skipped = 0

    destination.mkdir(parents=True, exist_ok=True)
    for obj_dir in iter_object_dirs(source_root):
        rel = obj_dir.relative_to(source_root)
        target = destination / rel

        if target.exists():
            if not overwrite:
                print(f"[install] skip existing (use --overwrite): {target}")
                skipped += 1
                continue
            shutil.rmtree(target) if target.is_dir() else target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(obj_dir), str(target))
        print(f"[install] moved {rel} -> {target}")
        moved += 1

    return moved, skipped


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    destination = args.destination.resolve()

    try:
        entries = read_selected_entries(csv_path, args.column)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    allow_patterns = build_allow_patterns(entries, args.include, args.repo_subdir)

    # Resolve staging directory. Auto-create a temp one if not supplied.
    auto_staging = args.staging_dir is None
    if auto_staging:
        staging_dir = Path(tempfile.mkdtemp(prefix="manitwin_dl_"))
    else:
        staging_dir = args.staging_dir.resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest:    {csv_path}")
    print(f"Repo:        {args.repo_id} ({args.repo_type})")
    print(f"Entries:     {len(entries)} (e.g. {entries[:3]}{'...' if len(entries) > 3 else ''})")
    print(f"Staging dir: {staging_dir}{' (auto-temp)' if auto_staging else ''}")
    print(f"Install dir: {destination}")
    print(f"Patterns:    {len(allow_patterns)}")

    if args.dry_run:
        for pat in allow_patterns:
            print(f"  + {pat}")
        if auto_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
        return 0

    exit_code = 0
    try:
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=str(staging_dir),
                allow_patterns=allow_patterns,
                token=args.token,
                max_workers=args.max_workers,
            )
        except HfHubHTTPError as exc:
            print(f"Hugging Face download failed: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected error during download: {exc}", file=sys.stderr)
            return 2

        # The repo stores objects under `<repo_subdir>/<category>/<instance>/...`.
        # Treat that subdirectory as the tree to unpack/patch and move from.
        source_root = staging_dir / args.repo_subdir.strip("/") if args.repo_subdir else staging_dir
        if not source_root.is_dir():
            print(
                f"[error] Expected downloaded tree at {source_root} but it does not exist. Check --repo-subdir.",
                file=sys.stderr,
            )
            return 2

        if args.unpack:
            try:
                import pxr  # noqa: F401
            except ImportError:
                print(
                    "[unpack] Cannot import pxr. Install `usd-core` or run "
                    "inside an Isaac Sim / USD Python environment to enable "
                    "--unpack.",
                    file=sys.stderr,
                )
                return 2

            results = unpack_and_patch_tree(
                source_root,
                approx=args.approx,
                force=args.force_unpack,
                keep_usdz=args.keep_usdz,
            )
            n_ok = sum(1 for r in results if r.ok())
            n_warn = sum(1 for r in results if r.skipped_reason and not r.error)
            n_err = sum(1 for r in results if r.error)
            total_meshes = sum(r.meshes_patched for r in results)
            print(
                f"[unpack summary] ok={n_ok} warn={n_warn} err={n_err} "
                f"meshes_patched={total_meshes} approx={args.approx}"
            )
            if n_err:
                exit_code = 2

        if args.no_move:
            print(f"[install] --no-move set; leaving processed objects under {source_root}")
        else:
            moved, skipped = move_objects_into_destination(
                source_root,
                destination,
                overwrite=args.overwrite,
            )
            print(f"[install summary] moved={moved} skipped={skipped} destination={destination}")
    finally:
        if auto_staging and not args.keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"[cleanup] removed temporary staging dir {staging_dir}")
        elif args.keep_staging:
            print(f"[cleanup] keeping staging dir {staging_dir}")

    print(f"Done. Processed {len(entries)} object selections.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
