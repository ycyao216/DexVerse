# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic-safe rigid objects for explicitly opted-in scenes.

Isaac Lab 2.3's scene.reset_to writes velocities to every RigidObject, but PhysX
rejects these writes for kinematic bodies, including zero velocities. Use this
adapter in replay or CPU collection scenes: poses, collision properties and dynamic-body
velocity writes remain unchanged. No Isaac Lab classes are patched globally.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import UsdPhysics


class ReplayRigidObject(RigidObject):
    """Ignore unsupported velocity writes, using the actual spawned USD flags."""

    def _initialize_impl(self):
        super()._initialize_impl()
        stage = sim_utils.get_current_stage()
        dynamic = []
        for path in self.root_physx_view.prim_paths:
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
            if not body:
                raise RuntimeError(f"Replay body has no RigidBodyAPI: {path}")
            dynamic.append(not bool(body.GetKinematicEnabledAttr().Get()))
        self._replay_all_dynamic = all(dynamic)
        self._replay_all_kinematic = not any(dynamic)
        self._replay_dynamic_mask = torch.tensor(dynamic, dtype=torch.bool, device=self.device)

    def _dynamic_velocity_rows(self, root_velocity, env_ids):
        if self._replay_all_dynamic:
            return root_velocity, env_ids
        if self._replay_all_kinematic:
            return None
        ids = torch.arange(len(self._replay_dynamic_mask), device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        keep = self._replay_dynamic_mask[ids]
        if not bool(keep.any()):
            return None
        return root_velocity[keep.to(root_velocity.device)], ids[keep]

    def write_root_com_velocity_to_sim(self, root_velocity, env_ids=None):
        # write_root_velocity_to_sim and write_root*_state_to_sim delegate here.
        selected = self._dynamic_velocity_rows(root_velocity, env_ids)
        if selected is not None:
            velocity, ids = selected
            super().write_root_com_velocity_to_sim(velocity, env_ids=ids)

    def write_root_link_velocity_to_sim(self, root_velocity, env_ids=None):
        # Filter before the base method updates its link-velocity buffers.
        selected = self._dynamic_velocity_rows(root_velocity, env_ids)
        if selected is not None:
            velocity, ids = selected
            super().write_root_link_velocity_to_sim(velocity, env_ids=ids)


def configure_replay_rigid_objects(scene_cfg):
    """Opt ordinary rigid objects into the adapter before gym.make.

    Custom asset classes, articulations and rigid-object collections are left
    untouched. Kinematic flags are read once at physics initialization, including
    flags inherited from USD files, and refreshed if the assets are initialized
    again. Callers must not toggle these flags while the scene is running.
    """
    for cfg in vars(scene_cfg).values():
        if isinstance(cfg, RigidObjectCfg) and cfg.class_type is RigidObject:
            cfg.class_type = ReplayRigidObject
