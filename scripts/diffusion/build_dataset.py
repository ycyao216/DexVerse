#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn a replayed demonstration HDF5 into a diffusion-policy training file.

``create_demo_files_sequential.py`` writes observations split per group and per
term (``obs/policy/<term>``, ``obs/proprio/<term>``, ...). Training wants one
flat state vector per step. This script does that flattening offline -- no Isaac
Sim launch -- using the same group order the eval runner uses at runtime
(``dexverse.IL.diffusion.obs_state``), and copies each episode's recorded
``initial_state`` across so evaluation can pin episode starts.

Usage::

    python scripts/diffusion/build_dataset.py \
        --in_file  outputs/replay/Dexverse-GraspKettle-v0.state.seq.demo.h5 \
        --out_file datasets/diffusion/Dexverse-GraspKettle-v0.h5
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexverse.IL.diffusion.obs_state import OPTIONAL_GROUPS, REQUIRED_GROUPS
from dexverse.recording_compat import recorded_task_name


def build_dataset(in_file: Path, out_file: Path, *, task: str | None, pkl_file: Path | None) -> dict[str, Any]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(in_file, "r") as src, h5py.File(out_file, "w") as dst:
        if "data" not in src:
            raise ValueError(f"'{in_file}' has no top-level 'data' group; not a replay HDF5.")
        task_name = recorded_task_name(task or str(src.attrs.get("task", "")))
        term_order = _read_term_order(src, in_file)
        initial_states = _read_initial_states(pkl_file or _source_pickle(src))

        data = dst.create_group("data")
        layout: list[dict[str, Any]] = []
        dims: set[tuple[int, int]] = set()
        for name in sorted(src["data"]):
            episode = src["data"][name]
            if "obs" not in episode or "actions" not in episode:
                continue
            state, layout = _flatten_state(episode["obs"], term_order)
            actions = np.asarray(episode["actions"][...], dtype=np.float32)
            length = min(state.shape[0], actions.shape[0])
            if length == 0:
                continue

            dims.add((state.shape[1], actions.shape[1]))
            if len(dims) > 1:
                raise ValueError(f"Episode '{name}' disagrees with earlier episodes on (state_dim, action_dim): {dims}.")

            demo = data.create_group(name)
            demo.create_dataset("actions", data=actions[:length], compression="gzip")
            demo.create_dataset("obs/state", data=state[:length], compression="gzip")
            if name in initial_states:
                _write_tree(demo.create_group("initial_state"), initial_states[name])

        if not dims:
            raise ValueError(f"No episodes with both 'obs' and 'actions' in '{in_file}'.")

        state_dim, action_dim = dims.pop()
        summary = {
            "task": task_name,
            "source_file": str(in_file),
            "episodes": len(data),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_layout": layout,
        }
        dst.attrs["meta"] = json.dumps(summary)
    return {**summary, "out_file": str(out_file)}


def _flatten_state(obs: h5py.Group, term_order: dict[str, list[str]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Concatenate the state observation groups, mirroring ``obs_state.build_state``."""
    missing = [group for group in REQUIRED_GROUPS if group not in obs]
    if missing:
        raise KeyError(f"Replay HDF5 is missing required obs group(s) {missing} (have: {sorted(obs)}).")

    chunks, layout = [], []
    for group in REQUIRED_GROUPS + tuple(group for group in OPTIONAL_GROUPS if group in obs):
        flat = _concat_terms(obs[group], term_order.get(group))
        chunks.append(flat)
        layout.append({"group": group, "dim": int(flat.shape[1])})
    return np.concatenate(chunks, axis=1), layout


def _concat_terms(group: h5py.Group, order: list[str] | None) -> np.ndarray:
    """Concatenate a group's per-term datasets along the feature axis."""
    names = order if order is not None else sorted(group)
    missing = [name for name in names if name not in group]
    if missing:
        raise KeyError(f"term_order names {missing} absent from {group.name} (have: {sorted(group)}).")
    columns = []
    for name in names:
        term = np.asarray(group[name][...], dtype=np.float32)
        if term.ndim == 0:
            raise ValueError(f"Observation term '{group.name}/{name}' is a scalar; expected a (T, ...) array.")
        columns.append(term.reshape(term.shape[0], -1))
    return np.concatenate(columns, axis=1)


def _read_term_order(src: h5py.File, in_file: Path) -> dict[str, list[str]]:
    """Per-group term declaration order, as recorded by the replay script.

    This is the order the live env's ``concatenate_terms=True`` groups use. Without
    it we fall back to alphabetical, which silently produces a different layout for
    any group whose declaration order is not alphabetical -- training on one layout
    and evaluating on another.
    """
    raw = src.attrs.get("term_order")
    if raw is None:
        print(
            f"[warn] '{in_file}' has no 'term_order' attribute, falling back to alphabetical term order. "
            "Re-run the replay with a current create_demo_files_sequential.py to avoid a layout mismatch."
        )
        return {}
    return json.loads(raw if isinstance(raw, str) else raw.decode())


def _source_pickle(src: h5py.File) -> Path | None:
    raw = src.attrs.get("source_pickles")
    paths = json.loads(raw) if isinstance(raw, str) else None
    return Path(paths[0]) if paths else None


def _read_initial_states(pkl_file: Path | None) -> dict[str, Any]:
    """Map episode name -> recorded initial scene state, for ``--reset_from_dataset``."""
    if pkl_file is None or not pkl_file.is_file():
        print(f"[warn] source pickle {pkl_file} unavailable; output will have no initial_state entries.")
        return {}
    with pkl_file.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        str(episode.get("episode_name") or f"demo_{index}"): episode["initial_state"]
        for index, episode in enumerate(payload["episodes"])
        if episode.get("initial_state") is not None
    }


def _write_tree(group: h5py.Group, tree: Any) -> None:
    for key, value in tree.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(str(key)), value)
        else:
            group.create_dataset(str(key), data=np.asarray(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in_file", type=Path, required=True, help="Replay HDF5 from create_demo_files_sequential.py.")
    parser.add_argument("--out_file", type=Path, required=True, help="Destination training HDF5.")
    parser.add_argument("--task", type=str, default=None, help="Override the task name (default: the input's attribute).")
    parser.add_argument(
        "--pkl_file",
        type=Path,
        default=None,
        help="Demonstration pickle to copy initial_state from (default: the input's 'source_pickles' attribute).",
    )
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.in_file, args.out_file, task=args.task, pkl_file=args.pkl_file), indent=2))


if __name__ == "__main__":
    main()
