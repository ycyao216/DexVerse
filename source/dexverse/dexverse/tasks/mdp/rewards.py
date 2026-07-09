# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.observations import last_action
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .stage_machine import stage_success as stage_success_term
from .utils import (
    asset_axis_w,
    axis_alignment_dot,
    axis_tilt_angle,
    command_axis_w,
    factory_insert_success_mask,
    get_init_joint_pos,
    insert_peg_success_mask,
    normalize_axis,
    plug_charger_insertion_mask,
    push_t_overlap_ratio,
    resolve_joint_ids,
    root_height_delta,
    tracked_axis_rotation_step,
    vector_projection_on_axis,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_axis_alignment_dot(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed cosine between object axis and commanded target axis in world frame."""
    current_axis_w = asset_axis_w(env, asset_cfg=object_cfg, axis_local=axis_local)
    target_axis_w = command_axis_w(env, command_name=command_name, axis_local=axis_local, robot_cfg=robot_cfg)
    return axis_alignment_dot(current_axis_w, target_axis_w)


def object_axis_ang_vel_projection(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object angular velocity projected onto commanded target axis."""
    object: RigidObject = env.scene[object_cfg.name]
    target_axis_w = command_axis_w(env, command_name=command_name, axis_local=axis_local, robot_cfg=robot_cfg)
    return vector_projection_on_axis(object.data.root_ang_vel_w, target_axis_w)


def root_lin_vel_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    squared: bool = False,
) -> torch.Tensor:
    """Generic root linear-velocity magnitude penalty for rigid/articulated assets."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_w
    return torch.sum(torch.square(vel), dim=1) if squared else torch.norm(vel, dim=1)


def cumulative_axis_rotation(
    env: ManagerBasedRLEnv,
    tracker_name: str = "default",
    command_name: str | None = None,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    world_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cumulative signed rotation (rad) around a selected axis."""
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w
    if command_name:
        axis_w = command_axis_w(env, command_name=command_name, axis_local=axis_local, robot_cfg=robot_cfg)
    else:
        axis = normalize_axis(world_axis, device=env.device, dtype=object_quat_w.dtype)
        axis_w = axis.unsqueeze(0).expand(env.num_envs, -1)
    accumulated, _ = tracked_axis_rotation_step(env, object_quat_w, axis_w, tracker_name=tracker_name)
    return accumulated


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    distance_gain: float = 10.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using exponential distance decay.

    The internal decay slope is controlled by ``distance_gain`` while the global
    reward scale should be tuned with ``RewTerm.weight`` in task config.
    ``std`` is kept for backward compatibility with existing configs.
    """
    _ = std
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance_sum = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).sum(dim=-1)

    return torch.exp(-distance_gain * object_ee_distance_sum)


def lift_when_grasping_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.07,
) -> torch.Tensor:
    """Reward lifting the object when grasping."""
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    asset_to_obj_distances = torch.norm(asset_pos - object_pos[:, None, :], dim=-1)
    thumb_to_obj_distance = asset_to_obj_distances[:, 0]
    index_to_obj_distance = asset_to_obj_distances[:, 1]
    middle_to_obj_distance = asset_to_obj_distances[:, 2]
    ring_to_obj_distance = asset_to_obj_distances[:, 3]

    good_contact_cond1 = (thumb_to_obj_distance <= threshold) & (
        (index_to_obj_distance <= threshold)
        | (middle_to_obj_distance <= threshold)
        | (ring_to_obj_distance <= threshold)
    )
    latest_action: torch.Tensor = last_action(env)
    z_lift_action = latest_action[..., 2]
    # clip to 0
    z_lift_action = z_lift_action.clamp(0, 1)
    reward = good_contact_cond1 * z_lift_action
    return reward


def object_lift_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.05,
) -> torch.Tensor:
    """Reward lifting an asset above its default height.

    Returns a [0,1] reward based on how much the object is lifted.
    """
    asset = env.scene[asset_cfg.name]
    lift = root_height_delta(asset)
    return torch.clamp(lift / max(min_height, 1e-6), 0.0, 1.0)


def position_command_error(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
) -> torch.Tensor:
    """Reward tracking of commanded position using command metrics (world-frame distance)."""
    command_term = env.command_manager.get_term(command_name)
    distance = command_term.metrics["position_error"]
    return torch.exp(-distance / max(std, 1e-6))


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success using command metrics (world-frame errors)."""
    command_term = env.command_manager.get_term(command_name)
    pos_dist = command_term.metrics["position_error"]
    if not rot_std:
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = command_term.metrics["orientation_error"]
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def stage_success_reward(
    env: ManagerBasedRLEnv,
    task_key: str,
    terminal_stage: str | None = None,
    persistent: bool = False,
    success_mode: Literal["substage", "all"] | None = None,
    ordering_mode: Literal["strict", "free"] | None = None,
) -> torch.Tensor:
    """Sparse reward from stage-machine terminal success."""
    return stage_success_term(
        env,
        task_key=task_key,
        terminal_stage=terminal_stage,
        persistent=persistent,
        success_mode=success_mode,
        ordering_mode=ordering_mode,
    ).float()


def tilt_angle_reward(
    env: ManagerBasedRLEnv,
    threshold_rad: float = 2.35619449,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tilt_ge: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward tilt progress relative to a threshold angle (in radians).

    When ``tilt_ge`` is True, larger tilt is better and the reward rises from
    0 at ``threshold_rad`` to 1 at ``pi``. When ``tilt_ge`` is False, smaller
    tilt is better and the reward rises from 0 at ``threshold_rad`` to 1 at 0.
    """
    object: RigidObject = env.scene[object_cfg.name]
    angle = axis_tilt_angle(object.data.root_quat_w, axis_local=axis_local, world_axis=world_axis)
    if tilt_ge:
        return torch.clamp((angle - threshold_rad) / (math.pi - threshold_rad), 0.0, 1.0)
    return torch.clamp((threshold_rad - angle) / max(threshold_rad, 1e-6), 0.0, 1.0)


def joint_open_reward(
    env: ManagerBasedRLEnv,
    threshold_rad: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening a joint beyond a threshold (uses max abs joint position)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    max_abs = torch.abs(joint_pos).max(dim=1).values
    return torch.clamp(max_abs / threshold_rad, 0.0, 1.0)


def joint_displacement_reward(
    env: ManagerBasedRLEnv,
    threshold_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward joint displacement magnitude from init beyond a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)

    joint_pos = asset.data.joint_pos[:, joint_ids]
    init_joint_pos = get_init_joint_pos(asset, joint_ids)

    max_abs = torch.abs(joint_pos - init_joint_pos).max(dim=1).values
    return torch.clamp(max_abs / threshold_m, 0.0, 1.0)


def joint_open_fraction_reward(
    env: ManagerBasedRLEnv,
    close_angle_rad: float,
    open_angle_rad: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward joint opening as a normalized fraction between close and open angles."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    max_pos = joint_pos.max(dim=1).values
    denom = open_angle_rad - close_angle_rad
    if abs(denom) < 1e-6:
        return torch.zeros(env.num_envs, device=env.device)
    fraction = (max_pos - close_angle_rad) / denom
    return torch.clamp(fraction, 0.0, 1.0)


def joint_range_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward progress of joint position from lower to upper limits (uses max joint pos)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    lower = asset.data.joint_pos_limits[:, joint_ids, 0]
    upper = asset.data.joint_pos_limits[:, joint_ids, 1]
    denom = torch.clamp(upper - lower, min=1e-6)
    progress = (joint_pos - lower) / denom
    max_progress = progress.max(dim=1).values
    return torch.clamp(max_progress, 0.0, 1.0)


def joint_range_progress_from_init(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward progress of joint motion based on absolute delta from init over reachable range."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)

    joint_pos = asset.data.joint_pos[:, joint_ids]
    init_joint_pos = get_init_joint_pos(asset, joint_ids)

    lower = asset.data.joint_pos_limits[:, joint_ids, 0]
    upper = asset.data.joint_pos_limits[:, joint_ids, 1]
    reachable = torch.maximum(init_joint_pos - lower, upper - init_joint_pos)
    denom = torch.clamp(reachable, min=1e-6)
    progress = torch.abs(joint_pos - init_joint_pos) / denom
    max_progress = progress.max(dim=1).values
    return torch.clamp(max_progress, 0.0, 1.0)


def factory_insert_engaged_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    held_base_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_dist_thresh: float = 0.0025,
    z_threshold: float = 0.01,
    held_base_local_offset_2: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Sparse engaged bonus for peg-insert / gear-mesh style tasks.

    Pass ``held_base_local_offset_2`` to count *either* held reference point
    (e.g. either end of a pen) as engaged.
    """
    return factory_insert_success_mask(
        env=env,
        held_cfg=held_cfg,
        fixed_cfg=fixed_cfg,
        held_base_local_offset=held_base_local_offset,
        target_local_offset=target_local_offset,
        center_dist_thresh=center_dist_thresh,
        z_threshold=z_threshold,
        held_base_local_offset_2=held_base_local_offset_2,
    ).float()


def factory_insert_success_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    held_base_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_dist_thresh: float = 0.0025,
    z_threshold: float = 0.001,
    held_base_local_offset_2: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Sparse success bonus for peg-insert / gear-mesh style tasks.

    Pass ``held_base_local_offset_2`` to count *either* held reference point
    (e.g. either end of a pen) as success.
    """
    return factory_insert_success_mask(
        env=env,
        held_cfg=held_cfg,
        fixed_cfg=fixed_cfg,
        held_base_local_offset=held_base_local_offset,
        target_local_offset=target_local_offset,
        center_dist_thresh=center_dist_thresh,
        z_threshold=z_threshold,
        held_base_local_offset_2=held_base_local_offset_2,
    ).float()


def plug_charger_pose_success_reward(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    insertion_x_threshold: float = 0.02,
    insertion_y_threshold: float = 0.01,
    insertion_z_threshold: float = 0.01,
    held_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Sparse success bonus using insertion geometry criteria in receptacle frame."""
    return plug_charger_insertion_mask(
        env=env,
        held_cfg=held_cfg,
        fixed_cfg=fixed_cfg,
        held_local_offset=held_local_offset,
        fixed_local_offset=fixed_local_offset,
        insertion_x_threshold=insertion_x_threshold,
        lateral_y_threshold=insertion_y_threshold,
        vertical_z_threshold=insertion_z_threshold,
    ).float()


def insert_peg_success_reward(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    peg_head_local_offset: tuple[float, float, float] = (0.10, 0.0, 0.0),
    hole_center_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hole_radius: float = 0.023,
    insertion_x_threshold: float = 0.015,
) -> torch.Tensor:
    """Sparse reward using ManiSkill PegInsertionSide success condition."""
    return insert_peg_success_mask(
        env=env,
        peg_cfg=peg_cfg,
        hole_cfg=hole_cfg,
        peg_head_local_offset=peg_head_local_offset,
        hole_center_local_offset=hole_center_local_offset,
        hole_radius=hole_radius,
        insertion_x_threshold=insertion_x_threshold,
    ).float()


def push_t_pose_dense_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_tee"),
    rotation_weight: float = 0.5,
    translation_weight: float = 0.5,
    translation_distance_gain: float = 5.0,
    overlap_success_threshold: float = 0.90,
    success_bonus: float = 3.0,
    overlap_point_step: float = 0.005,
) -> torch.Tensor:
    """Pose-based dense reward for PushT, aligned with ManiSkill PushT-v1 behavior."""
    tee: Articulation | RigidObject = env.scene[object_cfg.name]
    goal_tee: Articulation | RigidObject = env.scene[goal_cfg.name]

    _, _, tee_yaw = euler_xyz_from_quat(tee.data.root_quat_w)
    _, _, goal_yaw = euler_xyz_from_quat(goal_tee.data.root_quat_w)
    yaw_error = torch.atan2(torch.sin(tee_yaw - goal_yaw), torch.cos(tee_yaw - goal_yaw))

    rot_score = ((torch.cos(yaw_error) + 1.0) * 0.5) ** 2

    tee_to_goal_xy = tee.data.root_pos_w[:, :2] - goal_tee.data.root_pos_w[:, :2]
    tee_to_goal_dist = torch.linalg.vector_norm(tee_to_goal_xy, dim=1)
    trans_score = (1.0 - torch.tanh(translation_distance_gain * tee_to_goal_dist)) ** 2

    reward = rotation_weight * rot_score + translation_weight * trans_score

    overlap = push_t_overlap_ratio(
        env=env,
        object_cfg=object_cfg,
        goal_cfg=goal_cfg,
        point_step=overlap_point_step,
    )
    success = overlap >= float(overlap_success_threshold)
    if success_bonus > 0.0:
        reward = torch.where(success, torch.full_like(reward, float(success_bonus)), reward)
    return reward
