# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge per-session teleop pickles into one consolidated pickle per task.

Background
----------
``record_demos.py`` writes one timestamped ``*.pkl`` per teleop session, so
a single task folder ends up with N pickles even though they all target the
same env and the same object. Replaying them through ``create_demo_files.py``
would force Isaac Sim to tear down and rebuild the env between pickles,
which Isaac Lab does not handle robustly.

This helper scans a demonstrations tree, groups pickles by task folder,
verifies that the per-session pickles share compatible top-level metadata
(task name, ``usd_path``, ``robot_type``, ``json_path``, and the first
episode's ``multi_assets`` / ``multi_usds`` bindings), concatenates their
episodes with monotonically re-indexed ``episode_index`` fields, and writes
a single merged pickle.

Usage
-----
::

    # Merge every task folder under the default demos root
    python scripts/demo_tools/merge_session_pickles.py --all

    # One task only
    python scripts/demo_tools/merge_session_pickles.py --task grasping/Dexverse-BimanualLiftBasket-v0

The output filename is fixed as ``<task>.pkl`` inside each task folder.
The per-session inputs are left untouched.
"""

from __future__ import annotations

import argparse
import copy
import pickle
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMOS_ROOT = REPO_ROOT / "source" / "dexverse" / "demonstrations"

# Top-level pickle keys that MUST agree across sessions before we merge them.
# These keys describe the env / asset / robot identity of the recording; if
# they disagree, the demos are logically different datasets.
_REQUIRED_MATCH_KEYS: tuple[str, ...] = (
    "format",
    "env_name",
    "task",
    "usd_path",
    "robot_type",
    "json_path",
)

# Canonical top-level keys for single-env trajectory pickles written by
# ``record_demos.py``. We intentionally keep only these keys in merged output
# so ad-hoc metadata (e.g. ``augmented*`` markers) does not leak through.
_CANONICAL_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "format",
    "schema_version",
    "task",
    "env_name",
    "record_state",
    "usd_path",
    "robot_type",
    "json_path",
    "num_episodes",
    "episodes",
)


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: not a pickle dict")
    if payload.get("format") != "dexverse_trajectory":
        raise ValueError(
            f"{path}: format={payload.get('format')!r}; this merger only handles "
            "single-env 'dexverse_trajectory' pickles."
        )
    return payload


def _normalized_match_signature(payload: dict) -> tuple:
    """Stable key for equality comparison between pickles."""
    return tuple(payload.get(k) for k in _REQUIRED_MATCH_KEYS)


def _first_episode_bindings(payload: dict) -> tuple:
    """Capture the per-episode binding shape of the first episode for comparison."""
    eps = payload.get("episodes") or []
    if not eps:
        return ((), ())
    first = eps[0]
    return (
        # Bindings are usually small lists of dicts; pickling them to bytes
        # gives us a hashable, deep-equality-safe signature.
        pickle.dumps(first.get("multi_assets"), protocol=4),
        pickle.dumps(first.get("multi_usds"), protocol=4),
    )


def _validate_intra_pickle_bindings(payload: dict, path: Path) -> None:
    """Episodes within one pickle must agree on multi_* bindings (same env build)."""
    eps = payload.get("episodes") or []
    if len(eps) < 2:
        return
    sig0 = _first_episode_bindings(payload)
    for i, ep in enumerate(eps[1:], start=1):
        sig = (
            pickle.dumps(ep.get("multi_assets"), protocol=4),
            pickle.dumps(ep.get("multi_usds"), protocol=4),
        )
        if sig != sig0:
            raise ValueError(
                f"{path}: episode {i} has different multi_assets/multi_usds bindings "
                "than episode 0. Pickle is internally inconsistent; split it first."
            )


def _format_mismatch(
    a: dict,
    b: dict,
    path_a: Path,
    path_b: Path,
) -> str:
    diffs = []
    for k in _REQUIRED_MATCH_KEYS:
        if a.get(k) != b.get(k):
            diffs.append(f"  {k}: {a.get(k)!r} (from {path_a.name}) vs {b.get(k)!r} (from {path_b.name})")
    if not diffs:
        diffs.append("  (multi_assets / multi_usds bindings differ)")
    return "\n".join(diffs)


def _discover_pickles(demos_root: Path, task_relpaths: list[str], do_all: bool) -> list[Path]:
    found: set[Path] = set()
    for task in task_relpaths:
        task_dir = (demos_root / task).resolve()
        if not task_dir.is_dir():
            raise FileNotFoundError(f"--task directory not found: {task_dir}")
        found.update(task_dir.rglob("*.pkl"))
    if do_all:
        if not demos_root.is_dir():
            raise FileNotFoundError(f"--demos-root not found: {demos_root}")
        found.update(demos_root.rglob("*.pkl"))
    if not found:
        raise SystemExit("Nothing selected. Pass --all or --task <relpath> (repeatable).")
    return sorted(found)


def _group_by_task_dir(pickles: Iterable[Path]) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for p in pickles:
        groups.setdefault(p.parent.resolve(), []).append(p)
    return {k: sorted(v) for k, v in groups.items()}


def _looks_like_merged_output(path: Path, output_suffix: str) -> bool:
    return path.name.endswith(output_suffix)


def _merge_one_task(
    task_dir: Path,
    sources: list[Path],
    output_name: str,
    overwrite: bool,
) -> tuple[Path | None, int, str]:
    """Return ``(written_path, episode_count, status)`` for one task folder."""
    # Skip prior merged outputs so re-runs are idempotent.
    sources = [p for p in sources if not _looks_like_merged_output(p, output_name)]
    if not sources:
        return (None, 0, "no_inputs")

    out_path = task_dir / output_name
    if out_path.is_file() and not overwrite:
        return (out_path, 0, "skip_exists")

    if len(sources) == 1:
        # Nothing to merge; just sanity-check and either copy or no-op.
        only = _load_pickle(sources[0])
        _validate_intra_pickle_bindings(only, sources[0])
        if sources[0] == out_path:
            return (out_path, len(only.get("episodes") or []), "single_already_named")
        # Don't overwrite a sole input file silently; require explicit overwrite
        # since the user might just have one session and not need merging.
        return (None, len(only.get("episodes") or []), "single_input_no_action")

    base_payload: dict | None = None
    base_path: Path | None = None
    merged_episodes: list = []

    for src in sources:
        payload = _load_pickle(src)
        _validate_intra_pickle_bindings(payload, src)
        eps = payload.get("episodes") or []
        if not eps:
            continue
        if base_payload is None:
            base_payload = payload
            base_path = src
        else:
            if _normalized_match_signature(payload) != _normalized_match_signature(base_payload):
                raise ValueError(
                    f"{task_dir.name}: incompatible top-level metadata between {base_path.name} and {src.name}:\n"
                    + _format_mismatch(base_payload, payload, base_path, src)
                )
            if _first_episode_bindings(payload) != _first_episode_bindings(base_payload):
                raise ValueError(
                    f"{task_dir.name}: per-episode multi_assets/multi_usds bindings "
                    f"differ between {base_path.name} and {src.name}. These sessions "
                    "used different object pools and cannot be merged into one pickle."
                )
        merged_episodes.extend(eps)

    if base_payload is None or not merged_episodes:
        return (None, 0, "no_episodes")

    merged_payload = {k: copy.copy(base_payload.get(k)) for k in _CANONICAL_TOP_LEVEL_KEYS if k in base_payload}
    new_episodes = []
    for new_idx, ep in enumerate(merged_episodes):
        ep_copy = dict(ep)
        ep_copy["episode_index"] = new_idx
        if "episode_name" not in ep_copy or not ep_copy["episode_name"]:
            ep_copy["episode_name"] = f"demo_{new_idx}"
        new_episodes.append(ep_copy)
    merged_payload["episodes"] = new_episodes
    merged_payload["num_episodes"] = len(new_episodes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fp:
        pickle.dump(merged_payload, fp, protocol=4)
    return (out_path, len(new_episodes), "written")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demos-root",
        type=Path,
        default=DEFAULT_DEMOS_ROOT,
        help="Root directory of teleop pickles (default: source/dexverse/demonstrations).",
    )
    parser.add_argument(
        "--task", action="append", default=[], help="Task subdirectory relative to --demos-root. Repeatable."
    )
    parser.add_argument("--all", action="store_true", help="Process every task folder under --demos-root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing merged outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be merged without writing files.")
    args = parser.parse_args()

    demos_root = args.demos_root.expanduser().resolve()
    pickles = _discover_pickles(demos_root, args.task, args.all)
    groups = _group_by_task_dir(pickles)
    print(f"Found {len(pickles)} pickle(s) across {len(groups)} task folder(s).")

    total_written = 0
    total_episodes = 0
    for task_dir, sources in sorted(groups.items()):
        rel = task_dir.relative_to(demos_root) if task_dir.is_relative_to(demos_root) else task_dir
        output_name = f"{task_dir.name}.pkl"
        print(f"\n[{rel}]  ({len(sources)} pickle{'s' if len(sources) != 1 else ''})")
        if args.dry_run:
            for src in sources:
                print(f"    - {src.name}")
            print(f"    -> would write {output_name} (if >=2 inputs and not skipped)")
            continue
        try:
            out_path, count, status = _merge_one_task(
                task_dir,
                sources,
                output_name=output_name,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            print(f"    [error] {exc}")
            continue

        if status == "written":
            total_written += 1
            total_episodes += count
            print(f"    -> wrote {out_path.name} ({count} episodes)")
        elif status == "skip_exists":
            print(f"    [skip] {out_path.name} already exists (pass --overwrite to replace).")
        elif status == "single_input_no_action":
            print(f"    [skip] only one input session ({sources[0].name}); nothing to merge.")
        elif status == "single_already_named":
            print(f"    [skip] only input is already the merged output ({sources[0].name}).")
        elif status == "no_inputs":
            print("    [skip] all inputs are existing merged outputs.")
        elif status == "no_episodes":
            print("    [skip] no episodes in any input pickle.")

    if not args.dry_run:
        print(f"\nDone. Wrote {total_written} merged pickle(s), {total_episodes} episodes total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
