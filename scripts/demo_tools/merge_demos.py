# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge single-trajectory pickles into a multi-env batch pickle.

Each input pickle is one recorded by ``record_demos.py`` (``format =
"dexverse_trajectory"``) and may contain one or more episodes.  All episodes
across all inputs are concatenated -- in input order -- into a single batch
pickle (``format = "dexverse_trajectory_batch"``) whose shape is

    num_envs       = sum of episode counts across inputs
    max_num_steps  = longest action sequence among selected episodes
    action_dim     = shared across all episodes (validated)

The resulting pickle stores:

    - ``usd_paths``        : list[str]               length num_envs
    - ``initial_states``   : list[dict]              length num_envs
    - ``goal_poses``       : list[np.ndarray|None]   length num_envs
    - ``multi_assets``     : list[list[dict]]        length num_envs
    - ``multi_usds``       : list[list[dict]]        length num_envs
    - ``actions``          : np.ndarray (T, N, D)    zero-padded per env
    - ``states``           : list[list[dict]]        optional; only present when every merged
                                                     episode provides per-step states with len T+1
    - ``action_lengths``   : np.ndarray (N,) int32   real action count per env
    - ``success_flags``    : list[bool|None]         length num_envs
    - ``source``           : list[dict]              provenance per env

``replay_demos.py`` detects this format, builds the env with one USD per
parallel environment via ``MultiUsdFileCfg`` (deterministic, in order),
restores each env's recorded initial state via ``env.reset_to``, and rolls
out the padded action tensor -- freezing per-env observation capture as each
trajectory runs out of actions.

Typical use::

    python merge_demos.py \\
        --output merged.pkl \\
        --inputs can.pkl apple_a.pkl apple_b.pkl

    # Then replay with:
    python replay_demos.py --dataset_file merged.pkl --save_obs_action_pkl
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np

TRAJECTORY_FORMAT = "dexverse_trajectory"
BATCH_FORMAT = "dexverse_trajectory_batch"
BATCH_SCHEMA_VERSION = 3

# Per-episode active-object-subset field recorded by randomized-pool tasks
# (e.g. TrashDrawerSort, GraspTwoItems). Preserved through the merge so batch
# replay can restore each episode's recorded object subset. Mirrors
# _active_object_masks.ACTIVE_EPISODE_FIELDS (kept local: this tool must not
# import torch).
_ACTIVE_EPISODE_FIELDS: tuple[str, ...] = ("active_object_metadata",)


def load_trajectory_pickle(path: str) -> dict:
    """Load a trajectory pickle (single or batch) and validate the format tag."""
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{path!r} is not a valid pickle dictionary.")
    fmt = payload.get("format")
    if fmt not in {TRAJECTORY_FORMAT, BATCH_FORMAT}:
        raise ValueError(f"{path!r} has unknown trajectory format {fmt!r}.")
    return payload


def is_batch_pickle(payload: dict) -> bool:
    return isinstance(payload, dict) and payload.get("format") == BATCH_FORMAT


def _parse_selection(selection: list[str] | None) -> dict[str, list[int] | None]:
    """Parse ``--select path:idx[,idx...]`` CLI arguments into a dict.

    Keys are resolved pickle paths; values are either an explicit list of
    episode indices (sorted, duplicates removed) or ``None`` meaning "all
    episodes in this pickle".
    """
    resolved: dict[str, list[int] | None] = {}
    if not selection:
        return resolved
    for entry in selection:
        if ":" not in entry:
            raise ValueError(
                f"--select entry {entry!r} must be of the form 'path:idx[,idx...]' (e.g. 'apple.pkl:0,1')."
            )
        path_part, idx_part = entry.split(":", 1)
        path_key = str(Path(path_part).expanduser().resolve())
        idx_part = idx_part.strip()
        if idx_part == "" or idx_part == "*":
            resolved[path_key] = None
            continue
        indices = sorted({int(i) for i in idx_part.split(",") if i != ""})
        resolved[path_key] = indices
    return resolved


def _select_episodes(
    path: str,
    episodes: list[dict],
    selection: dict[str, list[int] | None],
) -> list[tuple[int, dict]]:
    """Return ``(original_index, episode)`` pairs honoring the user's selection."""
    path_key = str(Path(path).expanduser().resolve())
    if path_key not in selection:
        return list(enumerate(episodes))
    wanted = selection[path_key]
    if wanted is None:
        return list(enumerate(episodes))
    result: list[tuple[int, dict]] = []
    for idx in wanted:
        if idx < 0 or idx >= len(episodes):
            raise IndexError(f"--select index {idx} is out of range for {path!r} (has {len(episodes)} episodes).")
        result.append((idx, episodes[idx]))
    return result


def _extract_active_fields(episode: dict) -> dict:
    out: dict = {}
    for key in _ACTIVE_EPISODE_FIELDS:
        if key in episode:
            out[key] = copy.deepcopy(episode.get(key))
    return out


def _normalize_episode_binding_list(value) -> list[dict]:
    """Return a clean ``list[dict]`` for episode binding fields."""
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


def merge_trajectory_pickles(
    input_paths: list[str],
    output_path: str,
    *,
    selection: list[str] | None = None,
    allow_mixed_robot: bool = False,
    allow_mixed_task: bool = False,
) -> dict:
    """Merge single-trajectory pickles into a ``dexverse_trajectory_batch`` pickle.

    Args:
        input_paths: Pickle files produced by ``record_demos.py``.
        output_path: Destination pickle path.
        selection: Optional list of ``path:idx[,idx...]`` entries restricting
            which episodes from each pickle are included. Paths without an
            entry contribute *all* their episodes.
        allow_mixed_robot: If ``False`` (default), abort when inputs disagree
            on ``robot_type``. Actions from different robot configs are almost
            certainly not replay-compatible so this is opt-in.
        allow_mixed_task: If ``False`` (default), abort when inputs disagree
            on ``task`` / ``env_name``. Opt-in for advanced use cases.

    Returns:
        The assembled batch payload dict (also written to ``output_path``).
    """
    if len(input_paths) == 0:
        raise ValueError("Need at least one input pickle to merge.")

    selection_map = _parse_selection(selection)
    payloads = [load_trajectory_pickle(p) for p in input_paths]

    for path, payload in zip(input_paths, payloads):
        if payload.get("format") != TRAJECTORY_FORMAT:
            raise ValueError(
                f"{path!r} is not a single trajectory pickle (format="
                f"{payload.get('format')!r}); only format={TRAJECTORY_FORMAT!r} "
                "inputs can be merged."
            )

    task = payloads[0].get("task")
    env_name = payloads[0].get("env_name")
    robot_type = payloads[0].get("robot_type")
    json_path = payloads[0].get("json_path")
    for path, payload in zip(input_paths[1:], payloads[1:]):
        if not allow_mixed_task and payload.get("task") != task:
            raise ValueError(
                f"Task mismatch in {path!r}: {payload.get('task')!r} != {task!r} (pass --allow_mixed_task to override)."
            )
        if not allow_mixed_task and payload.get("env_name") != env_name:
            raise ValueError(
                f"env_name mismatch in {path!r}: {payload.get('env_name')!r} != {env_name!r} "
                "(pass --allow_mixed_task to override)."
            )
        if not allow_mixed_robot and payload.get("robot_type") != robot_type:
            raise ValueError(
                f"robot_type mismatch in {path!r}: {payload.get('robot_type')!r} "
                f"!= {robot_type!r} (pass --allow_mixed_robot to override)."
            )

    entries: list[dict] = []
    action_dim: int | None = None
    for path, payload in zip(input_paths, payloads):
        usd_path = payload.get("usd_path")
        episodes = payload.get("episodes", [])
        for original_idx, ep in _select_episodes(path, episodes, selection_map):
            actions = np.asarray(ep.get("actions", []))
            if actions.ndim != 2 or actions.shape[0] == 0:
                print(f"[merge_demos] skipping empty episode {ep.get('episode_name')!r} from {path!r}")
                continue
            if action_dim is None:
                action_dim = int(actions.shape[1])
            elif int(actions.shape[1]) != action_dim:
                raise ValueError(
                    f"Action dim mismatch in {path!r} episode "
                    f"{ep.get('episode_name')!r}: {actions.shape[1]} != {action_dim}."
                )
            states = ep.get("states") if "states" in ep else None
            if states is not None:
                if not isinstance(states, list):
                    raise ValueError(
                        f"Episode {ep.get('episode_name')!r} in {path!r} has invalid 'states' type "
                        f"{type(states)!r}. Expected list with length T+1 when present."
                    )
                expected_states_len = int(actions.shape[0]) + 1
                if len(states) != expected_states_len:
                    raise ValueError(
                        f"Episode {ep.get('episode_name')!r} in {path!r} has invalid states length "
                        f"{len(states)} (expected {expected_states_len}, i.e. T+1)."
                    )
            entries.append({
                "source_pickle": str(Path(path).expanduser().resolve()),
                "source_episode_index": int(ep.get("episode_index", original_idx)),
                "source_episode_name": str(ep.get("episode_name", f"demo_{original_idx}")),
                "usd_path": usd_path,
                "initial_state": copy.deepcopy(ep.get("initial_state")),
                "goal_pose": copy.deepcopy(ep.get("goal_pose")),
                "multi_assets": _normalize_episode_binding_list(ep.get("multi_assets")),
                "multi_usds": _normalize_episode_binding_list(ep.get("multi_usds")),
                "actions": actions.astype(np.float32, copy=False),
                "states": copy.deepcopy(states) if states is not None else None,
                "success": ep.get("success"),
                "active_fields": _extract_active_fields(ep),
            })

    if not entries:
        raise ValueError("No non-empty episodes were selected from the provided inputs.")

    num_envs = len(entries)
    max_steps = max(int(entry["actions"].shape[0]) for entry in entries)

    assert action_dim is not None  # Guaranteed by the non-empty check above.
    action_tensor = np.zeros((max_steps, num_envs, action_dim), dtype=np.float32)
    action_lengths = np.zeros((num_envs,), dtype=np.int32)
    for i, entry in enumerate(entries):
        length = int(entry["actions"].shape[0])
        action_tensor[:length, i, :] = entry["actions"]
        action_lengths[i] = length

    batch_payload = {
        "format": BATCH_FORMAT,
        "schema_version": BATCH_SCHEMA_VERSION,
        "task": task,
        "env_name": env_name,
        "robot_type": robot_type,
        "json_path": json_path,
        "num_envs": num_envs,
        "max_num_steps": max_steps,
        "action_dim": action_dim,
        "usd_paths": [entry["usd_path"] for entry in entries],
        "initial_states": [entry["initial_state"] for entry in entries],
        "goal_poses": [entry["goal_pose"] for entry in entries],
        "multi_assets": [entry["multi_assets"] for entry in entries],
        "multi_usds": [entry["multi_usds"] for entry in entries],
        "actions": action_tensor,
        "action_lengths": action_lengths,
        "success_flags": [entry["success"] for entry in entries],
        "source": [
            {
                "pickle": entry["source_pickle"],
                "episode_index": entry["source_episode_index"],
                "episode_name": entry["source_episode_name"],
                **entry.get("active_fields", {}),
            }
            for entry in entries
        ],
    }
    # Convenience top-level mirrors for batch tooling that doesn't inspect
    # ``source`` entries.
    for field in _ACTIVE_EPISODE_FIELDS:
        values = [entry.get("active_fields", {}).get(field) for entry in entries]
        if any(v is not None for v in values):
            batch_payload[field] = values
    num_with_states = sum(1 for entry in entries if entry["states"] is not None)
    if num_with_states == len(entries):
        batch_payload["states"] = [entry["states"] for entry in entries]
    else:
        print(
            f"[merge_demos] per-step states coverage: {num_with_states}/{len(entries)} episodes. "
            "At least one episode has no per-step 'states'; "
            "omitting batch['states']. Replay with --set_state will be unavailable for this merged file."
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as fp:
        pickle.dump(batch_payload, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[merge_demos] wrote batch pickle: {output_path}")
    print(f"  task          : {task}")
    print(f"  env_name      : {env_name}")
    print(f"  robot_type    : {robot_type}")
    print(f"  num_envs      : {num_envs}")
    print(f"  max_num_steps : {max_steps}")
    print(f"  action_dim    : {action_dim}")
    print("  per-env usd_path / action_length / source:")
    for i, entry in enumerate(entries):
        short_usd = entry["usd_path"] or "<default>"
        print(
            f"    [{i:3d}] len={int(action_lengths[i]):4d}  usd={short_usd}  "
            f"src={entry['source_pickle']}#{entry['source_episode_name']}"
        )

    return batch_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge trajectory pickles into a multi-env batch pickle.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output batch pickle path (format='dexverse_trajectory_batch').",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=str,
        help="One or more trajectory pickles produced by record_demos.py.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help=(
            "Optional per-input episode filter, given as 'path:idx[,idx...]' "
            "entries. Paths without an entry contribute all their episodes. "
            "Example: --select apple.pkl:0,1 can.pkl:0"
        ),
    )
    parser.add_argument(
        "--allow_mixed_robot",
        action="store_true",
        help="Permit merging inputs that disagree on robot_type.",
    )
    parser.add_argument(
        "--allow_mixed_task",
        action="store_true",
        help="Permit merging inputs that disagree on task / env_name.",
    )
    args = parser.parse_args()

    merge_trajectory_pickles(
        args.inputs,
        args.output,
        selection=args.select,
        allow_mixed_robot=args.allow_mixed_robot,
        allow_mixed_task=args.allow_mixed_task,
    )


if __name__ == "__main__":
    main()
