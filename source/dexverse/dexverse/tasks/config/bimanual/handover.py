# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist workspace constraints and continuous-transfer state for handover."""

from __future__ import annotations

import math

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class HandoverWristAction(JointPositionAction):
    """Clamp wrist targets to the physical limits installed for this task."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions = self._processed_actions.clamp(limits[..., 0], limits[..., 1])


def _axis_joint_limits(origin: float, axis: float, bounds: tuple[float, float]) -> tuple[float, float]:
    """Map a table-axis interval to joint coordinates, including reversed axes."""
    if not math.isclose(abs(axis), 1.0, abs_tol=1e-4):
        raise ValueError("Handover requires wrist translation axes aligned with the table.")
    return tuple(sorted((bound - origin) / axis for bound in bounds))


def constrain_handover_wrists(env, env_ids=None):
    """Install physical wrist stops from the spawned USD's joint frames.

    Use the actual per-hand mounts and axis signs, rather than assuming that
    every embodiment has Shadow's +/-0.3 m mount offsets. Cartesian wrist
    translations precede wrist rotations in the supported floating hands.
    """
    robot = env.scene["robot"]
    cached = getattr(env, "_handover_wrist_constraints", None)
    if cached is not None:
        ids, limits = cached
        # The recorder calls sim.reset(), rebuilding PhysX from the USD and
        # discarding runtime limits. Restore them on every environment reset.
        robot.write_joint_position_limit_to_sim(
            limits if env_ids is None else limits[env_ids], joint_ids=ids, env_ids=env_ids,
        )
        return
    # The PhysX articulation root can be a body below Robot, while the USD
    # joint prims are siblings of that body under the complete robot asset.
    root = env.sim.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/Robot")
    transforms = UsdGeom.XformCache()
    table = env.cfg.scene.table
    table_x = table.init_state.pos[0] + float(env.scene.env_origins[0, 0])
    table_y = table.init_state.pos[1] + float(env.scene.env_origins[0, 1])
    half_length, half_width = table.spawn.size[0] / 2, table.spawn.size[1] / 2
    bounds = {
        ("right", 0): (table_x - half_length, table_x + half_length),
        ("left", 0): (table_x - half_length, table_x + half_length),
        ("right", 1): (table_y - half_width, table_y),
        ("left", 1): (table_y, table_y + half_width),
    }
    found = {}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        name = prim.GetName()
        side = next(
            (
                side
                for side, prefixes in (("right", ("rh_", "right_")), ("left", ("lh_", "left_")))
                if name.startswith(prefixes)
            ),
            None,
        )
        if side is None or name not in robot.joint_names:
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        parent = env.sim.stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        parent_world = transforms.GetLocalToWorldTransform(parent)
        axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[joint.GetAxisAttr().Get()]
        axis = Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(axis)
        axis_world = parent_world.TransformDir(axis).GetNormalized()
        world_axis = max(range(3), key=lambda index: abs(axis_world[index]))
        if world_axis == 2:
            continue
        origin = parent_world.Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))[world_axis]
        key = (side, world_axis)
        limits = _axis_joint_limits(origin, axis_world[world_axis], bounds[key])
        if key in found:
            raise ValueError(f"Multiple wrist joints found for {key}.")
        found[key] = (robot.joint_names.index(name), limits)
    if set(found) != set(bounds):
        raise ValueError(f"Could not constrain both handover wrists in X/Y; resolved: {sorted(found)}")
    ids = [found[key][0] for key in bounds]
    limits = torch.tensor([found[key][1] for key in bounds], device=env.device)
    limits = limits.unsqueeze(0).repeat(env.num_envs, 1, 1)
    original = robot.data.joint_pos_limits[:, ids]
    limits[..., 0] = torch.maximum(limits[..., 0], original[..., 0])
    limits[..., 1] = torch.minimum(limits[..., 1], original[..., 1])
    if torch.any(limits[..., 0] >= limits[..., 1]):
        raise ValueError("Handover wrist workspace does not intersect the robot's joint limits.")
    env._handover_wrist_constraints = (ids, limits.clone())
    robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)


class HandoverState:
    """Latch pickup and failure, but require a live, sustained receiver hold.

    A failed attempt stays invalid until reset. Repeated evaluations in one
    environment step do not advance timers. All timers use simulation time.
    """

    def __init__(self, num_envs, device, step_dt, receiver_hold_s=0.3, contact_gap_s=0.1):
        self.hold_steps = max(1, math.ceil(receiver_hold_s / step_dt))
        self.gap_steps = max(1, math.ceil(contact_gap_s / step_dt))
        self.picked_up = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros_like(self.picked_up)
        self.success = torch.zeros_like(self.picked_up)
        self.hold_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gap_count = torch.zeros_like(self.hold_count)
        self.last_step = torch.full_like(self.hold_count, -1)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.picked_up, self.failed, self.success, self.hold_count, self.gap_count):
            value[ids] = 0
        self.last_step[ids] = -1

    def update(self, step, source_only, receiver_only, lifted, clear_of_table, supported, stable):
        rolled_back = step < self.last_step
        if rolled_back.any():
            self.reset(rolled_back)
        fresh = step != self.last_step
        # Require the source hand to be the first hand to lift the object.
        self.failed |= fresh & ~self.picked_up & lifted & receiver_only
        self.picked_up |= fresh & ~self.failed & source_only & lifted
        active = self.picked_up & ~self.failed
        gap = torch.where(active & ~supported, self.gap_count + 1, 0)
        self.gap_count = torch.where(fresh, gap, self.gap_count)
        self.failed |= fresh & active & (~clear_of_table | (self.gap_count >= self.gap_steps))
        holding = self.picked_up & ~self.failed & receiver_only & lifted & stable
        count = torch.where(holding, self.hold_count + 1, 0)
        self.hold_count = torch.where(fresh, count, self.hold_count)
        self.success = (self.hold_count >= self.hold_steps) & holding
        self.last_step = step.clone()
        return self.success


def reset_handover_state(env, env_ids=None):
    """Clear history even when the recorder evaluates success outside its manager."""
    state = getattr(env, "_handover_state", None)
    if state is not None:
        visuals = getattr(env, "_handover_visuals", None)
        if visuals is not None:
            visuals.on_reset(state, env_ids)
        state.reset(env_ids)
    update_handover_visuals(env)


def update_handover_visuals(env):
    """Refresh operator feedback without evaluating or changing task state."""
    if not getattr(env.cfg, "enable_debug_vis", False):
        return
    visuals = getattr(env, "_handover_visuals", None)
    if visuals is None:
        from .handover_visualization import HandoverVisuals

        visuals = env._handover_visuals = HandoverVisuals(env)
    visuals.update(getattr(env, "_handover_state", None), env.num_envs)
