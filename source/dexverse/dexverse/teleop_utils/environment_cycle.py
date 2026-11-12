# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select one preloaded object environment for teleoperation and recording."""

from __future__ import annotations

import copy

import numpy as np
import torch

from .relocate_collection import object_id_from_asset


def object_pool_paths(env_cfg) -> list[str]:
    spawn = env_cfg.scene.object.spawn
    if hasattr(spawn, "assets_cfg"):
        return [str(asset.usd_path) for asset in spawn.assets_cfg]
    paths = getattr(spawn, "usd_path", None)
    return [paths] if isinstance(paths, str) else list(paths or [])


def configure_single_collection_object(env_cfg, object_id: str):
    """Load exactly one pool asset and its annotated upright placement."""
    paths = object_pool_paths(env_cfg)
    matches = [i for i, path in enumerate(paths)
               if object_id_from_asset({"usd_path": path}) == object_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous collection object: {object_id}")
    index = matches[0]
    placements = getattr(env_cfg, "object_placements", None)
    if not placements or len(placements) != len(paths):
        raise ValueError("Single-object collection requires the pool's upright annotations.")
    placement = placements[index]
    spawn = env_cfg.scene.object.spawn
    if hasattr(spawn, "assets_cfg"):
        spawn.assets_cfg = [copy.deepcopy(spawn.assets_cfg[index])]
    else:
        spawn.usd_path = [paths[index]]
    spawn.random_choice = False
    env_cfg.scene.num_envs = 1
    env_cfg.teleop_object_cycle_size = 1
    env_cfg.object_placements = (placement,)
    params = env_cfg.events.reset_object.params
    params["per_env_positions"] = [list(placement.position_offset)]
    params["per_env_quats"] = [list(placement.quat)]
    env_cfg.scene.object.init_state.pos = (
        placement.position_offset[0], placement.position_offset[1],
        params["support_z"] + placement.position_offset[2],
    )
    env_cfg.scene.object.init_state.rot = tuple(placement.quat)
    env_cfg.collection_object_asset = {
        "object_id": object_id, "usd_path": paths[index], "source_env_index": 0,
    }
    return env_cfg.collection_object_asset


def configure_object_cycle(env_cfg, requested_size: int | None = None) -> int:
    """Size a deterministic bank; an explicit single-object override stays single."""
    size = requested_size if requested_size is not None else getattr(env_cfg, "teleop_object_cycle_size", 1)
    if size < 1:
        raise ValueError("object_cycle_size must be at least 1")
    if size == 1:
        env_cfg.teleop_object_cycle_size = 1
        return 1
    if not getattr(env_cfg, "teleop_object_cycle_size", None):
        raise ValueError("This task does not support cycling object environments.")
    paths = object_pool_paths(env_cfg)
    size = min(size, len(paths))
    if size < 2:
        env_cfg.teleop_object_cycle_size = 1
        return 1
    if len(set(paths[:size])) != size:
        raise ValueError("The object bank must contain distinct USD paths.")
    spawn = env_cfg.scene.object.spawn
    spawn.random_choice = False
    # Multi-asset spawning builds templates for every configured asset, even
    # when only a few environments use them. Load only this bank's objects.
    if hasattr(spawn, "assets_cfg"):
        spawn.assets_cfg = spawn.assets_cfg[:size]
    else:
        spawn.usd_path = paths[:size]
    env_cfg.scene.num_envs = size
    env_cfg.teleop_object_cycle_size = size
    return size


def select_scene_state(state, env_index: int):
    """Keep a one-row scene snapshot compatible with ordinary single-env demos."""
    if isinstance(state, dict):
        return {key: select_scene_state(value, env_index) for key, value in state.items()}
    if isinstance(state, torch.Tensor):
        return state[env_index : env_index + 1].clone()
    raise TypeError(f"Unexpected scene-state value: {type(state).__name__}")


def single_env_bindings(bindings, env_index: int):
    """Select traced bindings and rebase their prim paths to replay environment 0."""
    result = []
    prefix = f"/env_{env_index}/"
    for binding in bindings or []:
        path = str(binding.get("resolved_prim_path", ""))
        if prefix in path:
            item = dict(binding)
            item["resolved_prim_path"] = path.replace(prefix, "/env_0/")
            result.append(item)
    return result or None


class ObjectEnvironmentCycle:
    """Keep XR, retargeter coordinates, desktop view and control on one env."""

    def __init__(self, env, device=None, *, enabled=False):
        self.env = env
        self.device = device
        self.enabled = enabled and env.num_envs > 1
        self.index = 0
        self._completed: set[int] = set()
        self._origin = np.zeros(3, dtype=np.float32)
        self._anchor_position = None
        if self.enabled and device is not None and getattr(device, "_xr_cfg", None) is not None:
            if device._xr_cfg.anchor_prim_path is not None:
                raise ValueError("Object cycling requires a static XR anchor.")
            self._anchor_position = np.array(device._xr_cfg.anchor_pos, dtype=np.float64)
        self.select(0)

    def select(self, index: int):
        if not 0 <= index < self.env.num_envs:
            raise IndexError(index)
        self.index = index
        if not self.enabled:
            return
        origin = self.env.scene.env_origins[index].detach().cpu().numpy()
        delta = origin - self._origin
        controller = getattr(self.env, "viewport_camera_controller", None)
        if controller is not None:
            controller.set_view_env_index(index)
        if self._anchor_position is not None:
            from isaacsim.core.prims import SingleXFormPrim

            position = self._anchor_position + origin
            self.device._xr_cfg.anchor_pos = tuple(position)
            SingleXFormPrim(self.device._xr_anchor_headset_path).set_world_pose(position=position)
        for retargeter in getattr(self.device, "_retargeters", ()):
            # Relative retargeting keeps calibration in world coordinates.
            for pose in getattr(retargeter, "retarget_base_wrist_poses", {}).values():
                pose[:3] += delta
            # Absolute retargeting subtracts the selected robot's joint origin.
            for frame in getattr(retargeter, "_wrist_joint_origins", {}).values():
                frame["pos"] += delta
        self._origin = origin.copy()
        asset = self.object_asset()
        print(f"Active object environment {index + 1}/{self.env.num_envs}: {asset['usd_path']}")

    def mark_success(self):
        """Retire this object's slot after saving its successful demonstration."""
        if self.enabled:
            self._completed.add(self.index)

    def advance(self) -> bool:
        """Visit the next unfinished object; return False when all are complete.

        Failed/skipped objects remain eligible for the next pass through the
        pool. Completed objects never produce duplicate demos in this session.
        Teleop does not mark successes, so it keeps cycling through the pool.
        """
        if not self.enabled:
            return True
        for offset in range(1, self.env.num_envs + 1):
            index = (self.index + offset) % self.env.num_envs
            if index not in self._completed:
                self.select(index)
                return True
        return False

    def actions(self, action):
        if not self.enabled:
            return action.repeat(self.env.num_envs, 1)
        # Keep inactive hands at their last command instead of copying teleop
        # motions into every environment.
        batched = self.env.action_manager.action.clone()
        batched[self.index] = action
        return batched

    def scene_state(self):
        return select_scene_state(self.env.scene.get_state(is_relative=True), self.index)

    def object_asset(self):
        if not self.enabled:
            return getattr(self.env.cfg, "collection_object_asset", None)
        return {"usd_path": object_pool_paths(self.env.cfg)[self.index], "source_env_index": self.index}


def configure_replay_object_bank(env_cfg, episodes):
    """Bind each recorded object to a fixed replay slot, preserving USD identity."""
    assets = [episode.get("object_asset") for episode in episodes]
    if not any(assets):
        return
    if not all(assets):
        raise ValueError("Cannot mix object-cycling episodes with episodes missing object_asset metadata.")
    paths = list(dict.fromkeys(str(asset["usd_path"]) for asset in assets))
    original_paths = object_pool_paths(env_cfg)
    spawn = env_cfg.scene.object.spawn
    if any(path not in original_paths for path in paths):
        raise ValueError("A recorded object USD is missing from the replay task's object pool.")
    indices = [original_paths.index(path) for path in paths]
    if hasattr(spawn, "assets_cfg"):
        spawn.assets_cfg = [copy.deepcopy(spawn.assets_cfg[index]) for index in indices]
    elif isinstance(getattr(spawn, "usd_path", None), list):
        spawn.usd_path = paths
    else:
        raise ValueError("Recorded object cycling requires a multi-object replay task.")
    spawn.random_choice = False
    env_cfg.scene.num_envs = len(paths)
    # Keep startup/reset placements aligned with the reordered object bank.
    params = env_cfg.events.reset_object.params
    placements = getattr(env_cfg, "object_placements", None)
    if placements and len(placements) == len(original_paths):
        # A one-env config may only have object 0 in its reset arrays. The
        # full pool annotations are the source of truth for any selected ID.
        selected = tuple(placements[index] for index in indices)
        env_cfg.object_placements = selected
        params["per_env_positions"] = [list(p.position_offset) for p in selected]
        params["per_env_quats"] = [list(p.quat) for p in selected]
    else:
        for name in ("per_env_positions", "per_env_quats"):
            if name in params:
                params[name] = [params[name][index] for index in indices]
    env_cfg.replay_object_paths = paths


def replay_object_env_index(env_cfg, episode):
    asset = episode.get("object_asset")
    if asset is None:
        return 0
    return env_cfg.replay_object_paths.index(str(asset["usd_path"]))
