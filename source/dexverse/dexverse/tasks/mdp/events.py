# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event helpers for manager-based tasks (non-termination utilities)."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .resets import reset_joints_to_init
from .utils import resolve_env_ids, resolve_joint_ids


def reset_board_and_switch_xy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    board_cfg: SceneEntityCfg,
    switch_cfg: SceneEntityCfg,
    switch_joint_cfg: SceneEntityCfg | None = None,
    xy_range: tuple[float, float] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Reset board and switch positions with the same XY offset."""
    board = env.scene[board_cfg.name]
    switch = env.scene[switch_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)

    if x_range is None:
        if xy_range is None:
            raise ValueError("reset_board_and_switch_xy requires xy_range or x_range.")
        x_range = xy_range
    if y_range is None:
        if xy_range is None:
            raise ValueError("reset_board_and_switch_xy requires xy_range or y_range.")
        y_range = xy_range

    # sample offsets
    offsets = torch.zeros((env_ids_t.shape[0], 3), device=env.device)
    offsets[:, 0].uniform_(x_range[0], x_range[1])
    offsets[:, 1].uniform_(y_range[0], y_range[1])

    board_state = board.data.default_root_state[env_ids_t].clone()
    switch_state = switch.data.default_root_state[env_ids_t].clone()
    board_state[:, :3] += offsets
    switch_state[:, :3] += offsets

    board.write_root_pose_to_sim(board_state[:, 0:7], env_ids=env_ids_t)
    switch.write_root_pose_to_sim(switch_state[:, 0:7], env_ids=env_ids_t)
    if switch_joint_cfg is not None:
        reset_joints_to_init(env, env_ids_t, switch_joint_cfg)


def reset_book_cluster_and_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    left_book_cfg: SceneEntityCfg,
    right_book_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    target_pos_x: float,
    target_pos_z: float,
    target_pitch_rad: float,
    command_name: str = "object_pose",
):
    """Reset the three-book cluster with one shared Y offset and align the goal pose."""
    env_ids_t = resolve_env_ids(env, env_ids)
    num_envs = env_ids_t.shape[0]
    if num_envs == 0:
        return

    y_offsets = torch.empty(num_envs, device=env.device)
    y_offsets.uniform_(y_range[0], y_range[1])

    target_book = env.scene[object_cfg.name]
    left_book = env.scene[left_book_cfg.name]
    right_book = env.scene[right_book_cfg.name]

    # Target book is dynamic: reset pose and zero velocity.
    target_root_state = target_book.data.default_root_state[env_ids_t].clone()
    target_root_state[:, 1] += y_offsets
    target_root_state[:, 7:13] = 0.0
    target_book.write_root_pose_to_sim(target_root_state[:, 0:7], env_ids=env_ids_t)
    target_book.write_root_velocity_to_sim(target_root_state[:, 7:13], env_ids=env_ids_t)

    # Neighbor books are kinematic in this task: only write pose.
    for asset in (left_book, right_book):
        root_state = asset.data.default_root_state[env_ids_t].clone()
        root_state[:, 1] += y_offsets
        asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)

    command_term = env.command_manager.get_term(command_name)
    command_term.pose_command_b[env_ids_t, 0] = target_pos_x
    command_term.pose_command_b[env_ids_t, 1] = y_offsets
    command_term.pose_command_b[env_ids_t, 2] = target_pos_z

    zero = torch.zeros(num_envs, device=env.device)
    quat = quat_from_euler_xyz(zero, torch.full_like(zero, target_pitch_rad), zero)
    command_term.pose_command_b[env_ids_t, 3:] = quat

    if getattr(command_term.cfg, "use_world_frame", False):
        command_term.pose_command_w[env_ids_t, :3] = command_term.pose_command_b[env_ids_t, :3]
        command_term.pose_command_w[env_ids_t, 3:] = quat


def set_joint_position_limits(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lower: float | None = None,
    upper: float | None = None,
):
    """Set joint position limits for an articulation asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    env_ids_t = resolve_env_ids(env, env_ids)

    limits = asset.data.default_joint_pos_limits.clone()
    if lower is not None:
        limits[env_ids_t, joint_ids, 0] = lower
    if upper is not None:
        limits[env_ids_t, joint_ids, 1] = upper
    # Clamp to PhysX supported range for revolute joints.
    limits[env_ids_t, joint_ids, 0] = torch.clamp(limits[env_ids_t, joint_ids, 0], -2.0 * torch.pi, 2.0 * torch.pi)
    limits[env_ids_t, joint_ids, 1] = torch.clamp(limits[env_ids_t, joint_ids, 1], -2.0 * torch.pi, 2.0 * torch.pi)
    limits_to_set = limits[env_ids_t][:, joint_ids]
    asset.write_joint_position_limit_to_sim(
        limits_to_set, joint_ids=joint_ids, env_ids=env_ids_t, warn_limit_violation=False
    )


def debug_reset_reasons(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prefix: str = "Env reset",
):
    """Print reset reasons for each env_id based on active termination terms."""
    env_ids_t = resolve_env_ids(env, env_ids)

    term_names = env.termination_manager.active_terms
    if not term_names:
        for env_id in env_ids_t.tolist():
            print(f"{prefix} [{env_id}]: no termination terms active")
        return

    term_buffers = {name: env.termination_manager.get_term(name) for name in term_names}
    for env_id in env_ids_t.tolist():
        reasons = [name for name in term_names if bool(term_buffers[name][env_id])]
        if reasons:
            print(f"{prefix} [{env_id}]: {', '.join(reasons)}")
        else:
            print(f"{prefix} [{env_id}]: unknown")


def update_articulation_root_from_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_cfg: SceneEntityCfg,
    source_cfg: SceneEntityCfg,
):
    """Update an articulation root pose to follow a rigid object's root pose."""
    target: Articulation = env.scene[target_cfg.name]
    source = env.scene[source_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)

    pos = source.data.root_pos_w[env_ids_t]
    quat = source.data.root_quat_w[env_ids_t]
    root_pose = torch.cat([pos, quat], dim=-1)
    target.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)


def update_marker_from_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    marker_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Update a marker pose to follow a given articulation body with a local offset."""
    marker: RigidObject = env.scene[marker_cfg.name]
    asset: Articulation = env.scene[body_cfg.name]

    if isinstance(body_cfg.body_ids, slice) and body_cfg.body_ids == slice(None) and body_cfg.body_names is not None:
        body_cfg.resolve(env.scene)
    body_ids = body_cfg.body_ids
    if isinstance(body_ids, slice):
        body_id = 0
    else:
        body_id = body_ids[0]

    env_ids_t = resolve_env_ids(env, env_ids)

    body_pose = asset.data.body_pose_w[env_ids_t, body_id]
    offset_t = torch.tensor(offset, device=env.device, dtype=body_pose.dtype).unsqueeze(0).repeat(env_ids_t.shape[0], 1)
    pos = body_pose[:, 0:3] + quat_apply(body_pose[:, 3:7], offset_t)
    root_pose = torch.cat([pos, body_pose[:, 3:7]], dim=-1)
    marker.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
