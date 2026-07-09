# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restore per-episode active-object subsets recorded by randomized-pool tasks.

Some envs (e.g. TrashDrawerSort, GraspTwoItems) randomize which subset of a
fixed object pool participates in each episode. ``record_demos.py`` stores
that choice in the episode's ``active_object_metadata`` field (produced by the
env's ``get_active_object_metadata`` hook)::

    {"groups": {<group>: {"object_names": [...], "active_mask": [...]}}}

On replay the recorded masks must be written back into the env's
``_active_<group>_objects`` buffers (matched by group name) and per-object
visibility right after the initial ``reset_to`` -- otherwise the env
re-randomizes the subset and the replayed scene no longer matches the
recording.

``apply_recorded_active_masks`` is a no-op for episodes without the field and
for envs without the matching buffers, so callers can invoke it
unconditionally for every task.

Recordings made by the retired ``record_demos_long_horizon.py`` used flat
``active_drawer_mask``/``active_trash_mask`` fields instead; migrate them once
with ``archived/fix_legacy_active_masks.py`` -- the loaders here understand
only the current format.
"""

from __future__ import annotations

import numpy as np
import torch

# Episode field that carries active-subset data (preserved by merge_demos.py).
ACTIVE_EPISODE_FIELDS: tuple[str, ...] = ("active_object_metadata",)


def _normalize_bool_mask(raw_mask, expected_len: int) -> list[bool] | None:
    if raw_mask is None:
        return None
    mask = np.asarray(raw_mask, dtype=np.bool_).reshape(-1)
    if expected_len <= 0 or mask.size != expected_len:
        return None
    return [bool(v) for v in mask.tolist()]


def _groups_from_metadata(episode: dict) -> list[tuple[str, list[str], list[bool]]]:
    meta = episode.get("active_object_metadata")
    if not isinstance(meta, dict):
        return []
    groups: list[tuple[str, list[str], list[bool]]] = []
    for group_name, group in (meta.get("groups") or {}).items():
        if not isinstance(group, dict):
            continue
        names = [str(x) for x in (group.get("object_names") or [])]
        mask = _normalize_bool_mask(group.get("active_mask"), len(names))
        if names and mask is not None:
            groups.append((str(group_name), names, mask))
    return groups


def apply_recorded_active_masks(env, episode: dict, env_ids: torch.Tensor) -> None:
    for group_name, names, mask in _groups_from_metadata(episode):
        buffer = getattr(env, f"_active_{group_name}_objects", None)
        if isinstance(buffer, torch.Tensor):
            mask_t = torch.as_tensor(mask, device=env.device, dtype=torch.bool)
            buffer[env_ids, : mask_t.numel()] = mask_t.unsqueeze(0).expand(env_ids.numel(), -1)
        for name, is_active in zip(names, mask):
            try:
                env.scene[name].set_visibility(bool(is_active), env_ids=env_ids)
            except Exception:
                continue
