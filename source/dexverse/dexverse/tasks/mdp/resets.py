# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import pickle

import numpy as np
import torch
from dexverse.assets import DEBUG_HDR_PATH as DEBUG_HDR_ASSET_PATH
from dexverse.visual_purpose import hide_marker_from_cameras
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    subtract_frame_transforms,
)

from .utils import get_init_joint_pos, resolve_env_ids, resolve_joint_ids


def _get_or_create_frame_marker(env, key: str, frame_scale: tuple[float, float, float], prim_path: str):
    if not hasattr(env, "_reset_vis_markers"):
        env._reset_vis_markers = {}
    markers = env._reset_vis_markers
    if key not in markers:
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = frame_scale
        markers[key] = VisualizationMarkers(frame_marker_cfg.replace(prim_path=prim_path))
        hide_marker_from_cameras(markers[key])
    return markers[key]


def _get_hand_joint_defaults(robot, env_ids_t: torch.Tensor, arm_joint_ids, device: torch.device):
    """Get non-arm joint ids and their default state for selected envs."""
    if isinstance(arm_joint_ids, slice):
        return [], None, None

    num_total_joints = robot.data.joint_pos.shape[1]
    all_indices = torch.arange(num_total_joints, device=device, dtype=torch.long)
    arm_mask = torch.zeros(num_total_joints, dtype=torch.bool, device=device)
    if torch.is_tensor(arm_joint_ids):
        arm_mask[arm_joint_ids.to(device=device, dtype=torch.long)] = True
    else:
        arm_mask[arm_joint_ids] = True
    hand_joint_ids_list = all_indices[~arm_mask].tolist()
    if len(hand_joint_ids_list) == 0:
        return hand_joint_ids_list, None, None

    default_hand_pos = robot.data.default_joint_pos[env_ids_t][:, hand_joint_ids_list].clone()
    default_hand_vel = torch.zeros_like(default_hand_pos)
    return hand_joint_ids_list, default_hand_pos, default_hand_vel


def reset_root_pose_uniform(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    velocity_range: dict | None = None,
):
    """Reset root pose and root velocity from uniform ranges.

    ``pose_range`` values are treated as offsets from ``default_root_state``.
    ``velocity_range`` is absolute world-frame velocity; when omitted, all
    linear/angular components are reset to zero.
    """
    asset = env.scene[asset_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)

    root_state = asset.data.default_root_state[env_ids_t].clone()  # (N, 13)
    num = env_ids_t.shape[0]

    # Translation offsets
    offsets = torch.zeros((num, 3), device=env.device)
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    z_range = pose_range.get("z", (0.0, 0.0))
    offsets[:, 0].uniform_(x_range[0], x_range[1])
    offsets[:, 1].uniform_(y_range[0], y_range[1])
    offsets[:, 2].uniform_(z_range[0], z_range[1])
    root_state[:, 0:3] = root_state[:, 0:3] + env.scene.env_origins[env_ids_t] + offsets

    # Orientation offsets (roll/pitch/yaw)
    roll_range = pose_range.get("roll", (0.0, 0.0))
    pitch_range = pose_range.get("pitch", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    euler = torch.zeros((num, 3), device=env.device)
    euler[:, 0].uniform_(roll_range[0], roll_range[1])
    euler[:, 1].uniform_(pitch_range[0], pitch_range[1])
    euler[:, 2].uniform_(yaw_range[0], yaw_range[1])
    delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7], delta_quat)

    # Root linear / angular velocity. Keep deterministic zero defaults.
    if velocity_range is None:
        velocity_range = {}

    lin_x_range = velocity_range.get("x", (0.0, 0.0))
    lin_y_range = velocity_range.get("y", (0.0, 0.0))
    lin_z_range = velocity_range.get("z", (0.0, 0.0))
    ang_x_range = velocity_range.get("roll", velocity_range.get("wx", (0.0, 0.0)))
    ang_y_range = velocity_range.get("pitch", velocity_range.get("wy", (0.0, 0.0)))
    ang_z_range = velocity_range.get("yaw", velocity_range.get("wz", (0.0, 0.0)))

    root_state[:, 7].uniform_(lin_x_range[0], lin_x_range[1])
    root_state[:, 8].uniform_(lin_y_range[0], lin_y_range[1])
    root_state[:, 9].uniform_(lin_z_range[0], lin_z_range[1])
    root_state[:, 10].uniform_(ang_x_range[0], ang_x_range[1])
    root_state[:, 11].uniform_(ang_y_range[0], ang_y_range[1])
    root_state[:, 12].uniform_(ang_z_range[0], ang_z_range[1])

    # Kinematic rigid bodies cannot accept velocity writes in PhysX.
    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
    spawn_cfg = getattr(asset.cfg, "spawn", None)
    rigid_props = getattr(spawn_cfg, "rigid_props", None)
    is_kinematic = bool(getattr(rigid_props, "kinematic_enabled", False))
    if not is_kinematic:
        asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)


def reset_root_pose_uniform_excluding(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    *,
    reference_asset_cfg: SceneEntityCfg,
    min_xy_distance: float,
    max_attempts: int = 20,
    velocity_range: dict | None = None,
):
    """Like :func:`reset_root_pose_uniform`, but rejection-samples the xy
    position so it stays at least ``min_xy_distance`` (metres, world frame)
    from the xy of ``reference_asset_cfg``'s root.

    The reference asset's pose is read at call time, so for the constraint to
    be meaningful this event must run *after* the reference asset has been
    reset (the existing ``reset_object`` event normally satisfies this).

    If the constraint cannot be satisfied within ``max_attempts``, the last
    sampled pose is kept and a warning is printed -- usually a sign the
    sampling box is too small relative to ``min_xy_distance``.
    """
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]

    env_ids_t = resolve_env_ids(env, env_ids)
    num = env_ids_t.shape[0]
    device = env.device

    root_state = asset.data.default_root_state[env_ids_t].clone()  # (N, 13)
    base_xyz = root_state[:, 0:3] + env.scene.env_origins[env_ids_t]

    # Reference xy in world frame (already includes env origin).
    ref_xy = reference_asset.data.root_pos_w[env_ids_t, :2]

    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    z_range = pose_range.get("z", (0.0, 0.0))

    offsets = torch.zeros((num, 3), device=device)
    offsets[:, 0].uniform_(x_range[0], x_range[1])
    offsets[:, 1].uniform_(y_range[0], y_range[1])
    offsets[:, 2].uniform_(z_range[0], z_range[1])

    # Rejection-resample only the envs whose xy is too close to the reference.
    min_sq = float(min_xy_distance) * float(min_xy_distance)
    for _ in range(max_attempts):
        delta = (base_xyz[:, :2] + offsets[:, :2]) - ref_xy
        too_close = (delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]) < min_sq
        if not torch.any(too_close):
            break
        n_bad = int(too_close.sum().item())
        new_xy = torch.empty((n_bad, 2), device=device)
        new_xy[:, 0].uniform_(x_range[0], x_range[1])
        new_xy[:, 1].uniform_(y_range[0], y_range[1])
        offsets[too_close, 0] = new_xy[:, 0]
        offsets[too_close, 1] = new_xy[:, 1]

    # Re-check after the loop so we don't warn when the last iteration's
    # resample actually fixed the remaining envs.
    delta = (base_xyz[:, :2] + offsets[:, :2]) - ref_xy
    n_remaining = int(((delta[:, 0] ** 2 + delta[:, 1] ** 2) < min_sq).sum().item())
    if n_remaining > 0:
        print(
            f"[reset_root_pose_uniform_excluding] WARNING: {n_remaining}/{num} envs "
            f"could not satisfy min_xy_distance={min_xy_distance} from "
            f"'{reference_asset_cfg.name}' after {max_attempts} attempts; "
            "falling back to last draw. Consider widening the sampling range "
            "or shrinking min_xy_distance."
        )

    root_state[:, 0:3] = base_xyz + offsets

    # Orientation offsets (roll/pitch/yaw) -- mirrors reset_root_pose_uniform.
    roll_range = pose_range.get("roll", (0.0, 0.0))
    pitch_range = pose_range.get("pitch", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    euler = torch.zeros((num, 3), device=device)
    euler[:, 0].uniform_(roll_range[0], roll_range[1])
    euler[:, 1].uniform_(pitch_range[0], pitch_range[1])
    euler[:, 2].uniform_(yaw_range[0], yaw_range[1])
    delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7], delta_quat)

    if velocity_range is None:
        velocity_range = {}
    lin_x_range = velocity_range.get("x", (0.0, 0.0))
    lin_y_range = velocity_range.get("y", (0.0, 0.0))
    lin_z_range = velocity_range.get("z", (0.0, 0.0))
    ang_x_range = velocity_range.get("roll", velocity_range.get("wx", (0.0, 0.0)))
    ang_y_range = velocity_range.get("pitch", velocity_range.get("wy", (0.0, 0.0)))
    ang_z_range = velocity_range.get("yaw", velocity_range.get("wz", (0.0, 0.0)))

    root_state[:, 7].uniform_(lin_x_range[0], lin_x_range[1])
    root_state[:, 8].uniform_(lin_y_range[0], lin_y_range[1])
    root_state[:, 9].uniform_(lin_z_range[0], lin_z_range[1])
    root_state[:, 10].uniform_(ang_x_range[0], ang_x_range[1])
    root_state[:, 11].uniform_(ang_y_range[0], ang_y_range[1])
    root_state[:, 12].uniform_(ang_z_range[0], ang_z_range[1])

    # Kinematic rigid bodies cannot accept velocity writes in PhysX
    # (mirrors reset_root_pose_uniform).
    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
    spawn_cfg = getattr(asset.cfg, "spawn", None)
    rigid_props = getattr(spawn_cfg, "rigid_props", None)
    is_kinematic = bool(getattr(rigid_props, "kinematic_enabled", False))
    if not is_kinematic:
        asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)


def reset_articulation_with_supports_uniform(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    support_cfgs: list[SceneEntityCfg] | None,
    pose_range: dict,
    zero_support_velocity: bool = False,
):
    """Reset an articulation and its (kinematic) support bodies as one rigid rig.

    A single ``(x, y, z, roll, pitch, yaw)`` offset is sampled per env from
    ``pose_range`` (offsets from each asset's ``default_root_state``) and applied
    as one shared rigid transform pivoted at the articulation's default
    position: the articulation is translated + rotated, and every support is
    rotated about that same pivot and translated by the same offset.

    This keeps supports (e.g. stands that a thin object rests on) in their
    authored placement *under* the articulation at any sampled pose, instead of
    sliding out from under it when each asset is randomized independently.

    Supports are assumed kinematic, so only their pose is written (PhysX rejects
    velocity writes on kinematic bodies); the articulation root velocity is
    zeroed. Set ``zero_support_velocity=True`` when the supports are *dynamic*
    (e.g. a movable tray carried by the robot) so their carried-over velocity is
    cleared at reset too.
    """
    env_ids_t = resolve_env_ids(env, env_ids)
    num = env_ids_t.shape[0]
    device = env.device

    asset = env.scene[asset_cfg.name]

    # One shared translation offset + rotation per env.
    offsets = torch.zeros((num, 3), device=device)
    for i, key in enumerate(("x", "y", "z")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        offsets[:, i].uniform_(lo, hi)
    euler = torch.zeros((num, 3), device=device)
    for i, key in enumerate(("roll", "pitch", "yaw")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        euler[:, i].uniform_(lo, hi)
    delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    env_origins = env.scene.env_origins[env_ids_t]

    # Pivot of the shared transform = articulation default position (env-local).
    asset_root = asset.data.default_root_state[env_ids_t].clone()
    pivot_local = asset_root[:, 0:3]

    new_asset_pos = pivot_local + env_origins + offsets
    new_asset_quat = quat_mul(delta_quat, asset_root[:, 3:7])
    asset.write_root_pose_to_sim(torch.cat([new_asset_pos, new_asset_quat], dim=-1), env_ids=env_ids_t)
    asset.write_root_velocity_to_sim(torch.zeros((num, 6), device=device), env_ids=env_ids_t)

    for support_cfg in support_cfgs or []:
        support = env.scene[support_cfg.name]
        s_root = support.data.default_root_state[env_ids_t].clone()
        # Offset from the pivot (env-local), co-rotated with the rig.
        rel = s_root[:, 0:3] - pivot_local
        new_s_pos = pivot_local + quat_apply(delta_quat, rel) + env_origins + offsets
        new_s_quat = quat_mul(delta_quat, s_root[:, 3:7])
        support.write_root_pose_to_sim(torch.cat([new_s_pos, new_s_quat], dim=-1), env_ids=env_ids_t)
        if zero_support_velocity:
            support.write_root_velocity_to_sim(torch.zeros((num, 6), device=device), env_ids=env_ids_t)


def reset_object_from_place_annotations(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    per_env_positions,
    per_env_quats,
    support_z: float = 0.0,
    pose_range: dict | None = None,
):
    """Reset a rigid object to a per-env upright placement.

    The per-env placement is baked into ``per_env_positions`` / ``per_env_quats``
    at config construction time (see
    :func:`dexverse.tasks.object_annotations.collect_object_pool`). With
    ``MultiAssetSpawnerCfg(random_choice=False)`` environment ``i`` spawns
    ``usd_paths[i]`` so entry ``i`` of these lists corresponds to env ``i``.

    Args:
        env: Manager-based env.
        env_ids: Environments to reset.
        asset_cfg: Scene entity to reset (defaults to ``object``).
        per_env_positions: Iterable of ``(dx, dy, dz)`` offsets (one per env).
            ``dz`` is added to ``support_z`` so the annotated supporting point
            lands on the support surface.
        per_env_quats: Iterable of canonical upright quaternions as
            ``(w, x, y, z)`` (one per env).
        support_z: World-frame z-height of the supporting surface (e.g. table
            top). Added to every ``dz``.
        pose_range: Optional per-call uniform jitter applied on top of the
            canonical placement. Supported keys: ``x``, ``y``, ``z`` (meters)
            and ``roll``, ``pitch``, ``yaw`` (radians). All are offsets; yaw is
            applied around the world +z axis after the canonical quaternion.
    """
    asset = env.scene[asset_cfg.name]
    env_ids_t = resolve_env_ids(env, env_ids)
    num = env_ids_t.shape[0]
    device = env.device

    cache = getattr(env, "_object_placement_cache", None)
    if cache is None:
        cache = {}
        env._object_placement_cache = cache

    cache_key = (asset_cfg.name, id(per_env_positions), id(per_env_quats))
    cached = cache.get(cache_key)
    if cached is None:
        pos_t = torch.as_tensor(per_env_positions, device=device, dtype=torch.float32)
        quat_t = torch.as_tensor(per_env_quats, device=device, dtype=torch.float32)
        if pos_t.ndim != 2 or pos_t.shape[1] != 3:
            raise ValueError(f"per_env_positions must be (N, 3); got shape {tuple(pos_t.shape)}")
        if quat_t.ndim != 2 or quat_t.shape[1] != 4:
            raise ValueError(f"per_env_quats must be (N, 4); got shape {tuple(quat_t.shape)}")
        if pos_t.shape[0] == 0 or quat_t.shape[0] == 0:
            raise ValueError("per_env_positions / per_env_quats must be non-empty")
        if pos_t.shape[0] < env.num_envs or quat_t.shape[0] < env.num_envs:
            reps = int(math.ceil(env.num_envs / max(pos_t.shape[0], 1)))
            pos_t = pos_t.repeat(reps, 1)[: env.num_envs]
            quat_t = quat_t.repeat(reps, 1)[: env.num_envs]
        else:
            pos_t = pos_t[: env.num_envs]
            quat_t = quat_t[: env.num_envs]
        # Normalize quats defensively.
        quat_norm = torch.clamp(torch.linalg.norm(quat_t, dim=1, keepdim=True), min=1e-6)
        quat_t = quat_t / quat_norm
        cached = (pos_t.contiguous(), quat_t.contiguous())
        cache[cache_key] = cached

    pos_t, quat_t = cached

    env_origins = env.scene.env_origins[env_ids_t]
    base_pos = pos_t[env_ids_t].clone()
    base_pos[:, 2] = base_pos[:, 2] + float(support_z)
    target_pos = env_origins + base_pos

    base_quat = quat_t[env_ids_t].clone()

    if pose_range:
        offsets = torch.zeros((num, 3), device=device, dtype=target_pos.dtype)
        x_range = pose_range.get("x", (0.0, 0.0))
        y_range = pose_range.get("y", (0.0, 0.0))
        z_range = pose_range.get("z", (0.0, 0.0))
        offsets[:, 0].uniform_(float(x_range[0]), float(x_range[1]))
        offsets[:, 1].uniform_(float(y_range[0]), float(y_range[1]))
        offsets[:, 2].uniform_(float(z_range[0]), float(z_range[1]))
        target_pos = target_pos + offsets

        has_rot = any(k in pose_range for k in ("roll", "pitch", "yaw"))
        if has_rot:
            roll_range = pose_range.get("roll", (0.0, 0.0))
            pitch_range = pose_range.get("pitch", (0.0, 0.0))
            yaw_range = pose_range.get("yaw", (0.0, 0.0))
            euler = torch.zeros((num, 3), device=device, dtype=target_pos.dtype)
            euler[:, 0].uniform_(float(roll_range[0]), float(roll_range[1]))
            euler[:, 1].uniform_(float(pitch_range[0]), float(pitch_range[1]))
            euler[:, 2].uniform_(float(yaw_range[0]), float(yaw_range[1]))
            delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            # Apply delta in world frame: q_final = q_delta * q_base
            base_quat = quat_mul(delta_quat, base_quat)

    root_pose = torch.cat([target_pos, base_quat], dim=1)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
    zero_vel = torch.zeros((num, 6), device=device, dtype=target_pos.dtype)
    asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_t)


def reset_clutter_objects(
    env,
    env_ids: torch.Tensor,
    asset_names: list[str] | tuple[str, ...],
    slot_offsets: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    base_height: float,
    slot_offsets_by_asset: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    base_height_by_asset: list[float] | tuple[float, ...] | None = None,
    position_jitter_x: float = 0.0,
    position_jitter_y: float = 0.0,
    position_jitter_z: tuple[float, float] = (0.0, 0.0),
    roll_range: tuple[float, float] = (0.0, 0.0),
    pitch_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: dict | None = None,
    unique_slots: bool = True,
):
    """Reset clutter objects with optional per-asset XY slots and per-asset base heights.

    Backward compatible:
    - If ``slot_offsets_by_asset`` is None, XY slots are sampled from ``slot_offsets`` as before.
    - If ``base_height_by_asset`` is None, ``base_height`` is used for all assets as before.
    """
    env_ids_t = resolve_env_ids(env, env_ids)
    if len(asset_names) == 0:
        return
    if slot_offsets_by_asset is None and unique_slots and len(slot_offsets) < len(asset_names):
        raise ValueError(
            f"Need at least as many slot_offsets as asset_names, got {len(slot_offsets)} < {len(asset_names)}"
        )
    if slot_offsets_by_asset is not None and len(slot_offsets_by_asset) != len(asset_names):
        raise ValueError(
            f"slot_offsets_by_asset must match asset_names length, got {len(slot_offsets_by_asset)} !="
            f" {len(asset_names)}"
        )
    if base_height_by_asset is not None and len(base_height_by_asset) != len(asset_names):
        raise ValueError(
            f"base_height_by_asset must match asset_names length, got {len(base_height_by_asset)} != {len(asset_names)}"
        )

    num_envs = env_ids_t.shape[0]
    num_objects = len(asset_names)
    device = env.device

    if slot_offsets_by_asset is not None:
        sampled_xy = (
            torch.tensor(slot_offsets_by_asset, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
        )
    else:
        slot_xy = torch.tensor(slot_offsets, device=device, dtype=torch.float32)
        if unique_slots:
            slot_indices = torch.argsort(torch.rand((num_envs, len(slot_offsets)), device=device), dim=1)[
                :, :num_objects
            ]
        else:
            slot_indices = torch.randint(0, len(slot_offsets), (num_envs, num_objects), device=device)
        sampled_xy = slot_xy[slot_indices]

    sampled_xy[..., 0] += torch.empty((num_envs, num_objects), device=device).uniform_(
        -position_jitter_x, position_jitter_x
    )
    sampled_xy[..., 1] += torch.empty((num_envs, num_objects), device=device).uniform_(
        -position_jitter_y, position_jitter_y
    )
    sampled_z = torch.empty((num_envs, num_objects), device=device).uniform_(position_jitter_z[0], position_jitter_z[1])
    sampled_roll = torch.empty((num_envs, num_objects), device=device).uniform_(roll_range[0], roll_range[1])
    sampled_pitch = torch.empty((num_envs, num_objects), device=device).uniform_(pitch_range[0], pitch_range[1])
    sampled_yaw = torch.empty((num_envs, num_objects), device=device).uniform_(yaw_range[0], yaw_range[1])

    linear_velocity_range = velocity_range or {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    env_origins = env.scene.env_origins[env_ids_t]
    if base_height_by_asset is not None:
        base_heights = (
            torch.tensor(base_height_by_asset, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        )
    else:
        base_heights = torch.full((num_envs, num_objects), float(base_height), device=device, dtype=torch.float32)

    for object_idx, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        root_state = asset.data.default_root_state[env_ids_t].clone()

        root_state[:, 0] = env_origins[:, 0] + sampled_xy[:, object_idx, 0]
        root_state[:, 1] = env_origins[:, 1] + sampled_xy[:, object_idx, 1]
        root_state[:, 2] = env_origins[:, 2] + base_heights[:, object_idx] + sampled_z[:, object_idx]

        delta_quat = quat_from_euler_xyz(
            sampled_roll[:, object_idx],
            sampled_pitch[:, object_idx],
            sampled_yaw[:, object_idx],
        )
        root_state[:, 3:7] = quat_mul(root_state[:, 3:7], delta_quat)

        root_state[:, 7].uniform_(
            linear_velocity_range.get("x", (0.0, 0.0))[0], linear_velocity_range.get("x", (0.0, 0.0))[1]
        )
        root_state[:, 8].uniform_(
            linear_velocity_range.get("y", (0.0, 0.0))[0], linear_velocity_range.get("y", (0.0, 0.0))[1]
        )
        root_state[:, 9].uniform_(
            linear_velocity_range.get("z", (0.0, 0.0))[0], linear_velocity_range.get("z", (0.0, 0.0))[1]
        )
        root_state[:, 10:13] = 0.0

        asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
        asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)


def reset_random_inverted_v_obstacles(
    env,
    env_ids: torch.Tensor,
    obstacle_asset_cfgs: tuple[tuple[SceneEntityCfg, SceneEntityCfg], ...],
    slope_center: tuple[float, float, float],
    slope_quat: tuple[float, float, float, float],
    active_count_range: tuple[int, int],
    apex_tangent_range: tuple[float, float],
    apex_tangent_jitter: float,
    min_apex_tangent_spacing: float,
    apex_lateral_range: tuple[float, float],
    apex_normal_offset: float,
    leg_length: float,
    leg_yaw_angle_rad: float,
    inactive_local_position: tuple[float, float, float] = (0.0, 0.0, -2.0),
):
    """Reset multiple small inverted-V obstacles, parking inactive ones below the ramp."""
    env_ids_t = resolve_env_ids(env, env_ids)
    num_envs = env_ids_t.shape[0]
    device = env.device
    num_obstacles = len(obstacle_asset_cfgs)
    if num_obstacles == 0:
        return

    min_active, max_active = active_count_range
    if min_active < 0 or min_active > max_active or max_active > num_obstacles:
        raise ValueError(
            "active_count_range must satisfy 0 <= min <= max <= number of obstacle asset pairs; "
            f"got {active_count_range} for {num_obstacles} pairs"
        )
    if num_obstacles > 1:
        slot_spacing = (apex_tangent_range[1] - apex_tangent_range[0]) / (num_obstacles - 1)
        guaranteed_spacing = slot_spacing - 2.0 * apex_tangent_jitter
        if guaranteed_spacing < min_apex_tangent_spacing:
            raise ValueError(
                "Obstacle tangent slots are too close for the requested clearance; "
                f"got {guaranteed_spacing:.3f} m, need {min_apex_tangent_spacing:.3f} m"
            )

    env_origins = env.scene.env_origins[env_ids_t]
    slope_quat_t = torch.tensor(slope_quat, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    slope_center_t = torch.tensor(slope_center, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    inactive_local = (
        torch.tensor(inactive_local_position, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    )
    inactive_pos = env_origins + slope_center_t + quat_apply(slope_quat_t, inactive_local)

    active_counts = torch.randint(min_active, max_active + 1, (num_envs,), device=device)
    slot_tangents = torch.linspace(
        apex_tangent_range[0],
        apex_tangent_range[1],
        num_obstacles,
        device=device,
        dtype=torch.float32,
    )
    slot_order = torch.argsort(torch.rand((num_envs, num_obstacles), device=device), dim=1)

    left_dir_local = torch.tensor(
        [math.cos(leg_yaw_angle_rad), math.sin(leg_yaw_angle_rad), 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0)
    right_dir_local = torch.tensor(
        [math.cos(-leg_yaw_angle_rad), math.sin(-leg_yaw_angle_rad), 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0)
    half_leg = 0.5 * leg_length

    left_yaw = torch.full((num_envs,), leg_yaw_angle_rad, device=device, dtype=torch.float32)
    right_yaw = torch.full((num_envs,), -leg_yaw_angle_rad, device=device, dtype=torch.float32)
    zero = torch.zeros_like(left_yaw)
    left_local_quat = quat_from_euler_xyz(zero, zero, left_yaw)
    right_local_quat = quat_from_euler_xyz(zero, zero, right_yaw)
    left_quat = quat_mul(slope_quat_t, left_local_quat)
    right_quat = quat_mul(slope_quat_t, right_local_quat)

    for obstacle_idx, (left_asset_cfg, right_asset_cfg) in enumerate(obstacle_asset_cfgs):
        active_mask = (obstacle_idx < active_counts).unsqueeze(1)
        apex_local = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        apex_local[:, 0] = slot_tangents[slot_order[:, obstacle_idx]]
        if apex_tangent_jitter > 0.0:
            apex_local[:, 0] += torch.empty(num_envs, device=device, dtype=torch.float32).uniform_(
                -apex_tangent_jitter, apex_tangent_jitter
            )
        apex_local[:, 1].uniform_(apex_lateral_range[0], apex_lateral_range[1])
        apex_local[:, 2] = apex_normal_offset

        # The apex is the downhill tip, so each leg extends uphill from it.
        left_center_local = apex_local + half_leg * left_dir_local
        right_center_local = apex_local + half_leg * right_dir_local
        left_pos = env_origins + slope_center_t + quat_apply(slope_quat_t, left_center_local)
        right_pos = env_origins + slope_center_t + quat_apply(slope_quat_t, right_center_local)
        left_pos = torch.where(active_mask, left_pos, inactive_pos)
        right_pos = torch.where(active_mask, right_pos, inactive_pos)

        left_asset = env.scene[left_asset_cfg.name]
        right_asset = env.scene[right_asset_cfg.name]
        left_root_state = left_asset.data.default_root_state[env_ids_t].clone()
        right_root_state = right_asset.data.default_root_state[env_ids_t].clone()
        left_root_state[:, 0:3] = left_pos
        left_root_state[:, 3:7] = left_quat
        left_root_state[:, 7:13] = 0.0
        right_root_state[:, 0:3] = right_pos
        right_root_state[:, 3:7] = right_quat
        right_root_state[:, 7:13] = 0.0

        left_asset.write_root_pose_to_sim(left_root_state[:, 0:7], env_ids=env_ids_t)
        left_asset.write_root_velocity_to_sim(left_root_state[:, 7:13], env_ids=env_ids_t)
        right_asset.write_root_pose_to_sim(right_root_state[:, 0:7], env_ids=env_ids_t)
        right_asset.write_root_velocity_to_sim(right_root_state[:, 7:13], env_ids=env_ids_t)


def sync_object(
    env,
    env_ids: torch.Tensor,
    target_cfg: SceneEntityCfg,
    source_cfg: SceneEntityCfg,
    z_offset: float = 0.0,
    quat: tuple[float, float, float, float] | None = None,
    source_local_offset: tuple[float, float, float] | None = None,
    quat_local: tuple[float, float, float, float] | None = None,
):
    """Align target to source's pose with optional offsets.

    The target's world position is computed as
    ``source.root_pos + R(source.root_quat) * source_local_offset``, then a
    flat ``z_offset`` is added to z. ``source_local_offset`` defaults to
    ``None`` (treated as zero) so the legacy "copy xy, offset z" behaviour
    is preserved for existing call sites.

    Orientation (pick at most one of ``quat`` / ``quat_local``):
      - ``quat`` (default ``None``): absolute world-frame orientation. When
        given it is written verbatim, so the target does *not* follow the
        source's rotation. When ``None`` the target keeps its current quat.
      - ``quat_local``: a quaternion expressed in the *source's* frame. The
        target's world orientation becomes ``R(source.root_quat) * quat_local``
        so the target rotates rigidly with the source. Use this when the
        source is re-randomized (e.g. yaw jitter) and the target must follow.
    """
    target = env.scene[target_cfg.name]
    source = env.scene[source_cfg.name]

    if quat is not None and quat_local is not None:
        raise ValueError("sync_object: pass at most one of `quat` / `quat_local`.")

    env_ids_t = resolve_env_ids(env, env_ids)

    pos = source.data.root_pos_w[env_ids_t].clone()
    if source_local_offset is not None:
        offset_local = torch.tensor(source_local_offset, device=env.device, dtype=pos.dtype)
        offset_world = quat_apply(
            source.data.root_quat_w[env_ids_t],
            offset_local.unsqueeze(0).expand(env_ids_t.shape[0], -1),
        )
        pos = pos + offset_world
    pos[:, 2] = pos[:, 2] + z_offset

    if quat_local is not None:
        quat_local_t = (
            torch.tensor(quat_local, device=env.device, dtype=pos.dtype).unsqueeze(0).repeat(env_ids_t.shape[0], 1)
        )
        quat_t = quat_mul(source.data.root_quat_w[env_ids_t], quat_local_t)
    elif quat is None:
        quat_t = target.data.root_quat_w[env_ids_t]
    else:
        quat_t = torch.tensor(quat, device=env.device, dtype=pos.dtype).unsqueeze(0).repeat(env_ids_t.shape[0], 1)

    target.write_root_pose_to_sim(torch.cat([pos, quat_t], dim=1), env_ids=env_ids_t)


def reset_joints_to_init(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints to init_state positions (fallback to default if not specified)."""
    asset = env.scene[asset_cfg.name]

    joint_ids = resolve_joint_ids(env, asset_cfg)
    env_ids_t = resolve_env_ids(env, env_ids)

    joint_pos = get_init_joint_pos(asset, joint_ids)[env_ids_t]
    joint_vel = torch.zeros_like(joint_pos)
    # Two-step indexing: advanced-indexing on dim 0 (env_ids), then basic
    # slicing on dim 1 (joint_ids). The original ``[env_ids, joint_ids]``
    # form requires PyTorch to broadcast the two index tensors, which fails
    # when both are multi-element and their lengths differ — e.g. 50 envs
    # × 2 articulation joints. This form returns ``[N_envs, N_joints, 2]``
    # uniformly.
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids_t][:, joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids_t)


def reset_robot_to_pose_ik(
    env,
    env_ids: torch.Tensor,
    resolved_scene_cfg: SceneEntityCfg,
    ik_method: str = "dls",
    max_time: float = 1.0,
    num_ik_iters: int = 10,
    visualize: bool = True,
    frame_scale: tuple[float, float, float] = (0.05, 0.05, 0.05),
    instant_set: bool = False,
    z_offset: float = 0.02,
    override_xyz: tuple[float, float, float] | None = None,
    override_quat: tuple[float, float, float, float] | None = None,
):
    # TODO: make this more efficient
    robot = env.scene[resolved_scene_cfg.name]

    # Ensure SceneEntityCfg is resolved (convert names to indices)
    # Check if resolution is needed by seeing if joint_ids/body_ids are still slices
    if (
        isinstance(resolved_scene_cfg.joint_ids, slice)
        and resolved_scene_cfg.joint_ids == slice(None)
        and resolved_scene_cfg.joint_names is not None
    ):
        resolved_scene_cfg.resolve(env.scene)
    if (
        isinstance(resolved_scene_cfg.body_ids, slice)
        and resolved_scene_cfg.body_ids == slice(None)
        and resolved_scene_cfg.body_names is not None
    ):
        resolved_scene_cfg.resolve(env.scene)

    # Handle body_ids which can be a list or slice
    body_ids = resolved_scene_cfg.body_ids
    if isinstance(body_ids, slice):
        # Convert slice to list (slice(None) means all bodies, so use first body)
        if body_ids == slice(None):
            body_id = 0  # Use first body
        else:
            # For other slices, convert to list and take first
            body_id = list(range(robot.num_bodies))[body_ids][0]
    else:
        # It's a list, take first element
        body_id = body_ids[0]

    # Resolve jacobian body index (see tutorial: fixed-base has one less body index)
    # For fixed-base robots, the base body (index 0) is not included in the jacobian computation
    # So we need to subtract 1 from the body index to get the correct jacobian index
    if robot.is_fixed_base:
        ee_jacobi_idx = body_id - 1
    else:
        ee_jacobi_idx = body_id

    # Joints to control
    joint_ids = resolved_scene_cfg.joint_ids

    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
        ),
        num_envs=env.num_envs,
        device=env.device,
    )
    target_pos_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    target_quat_w = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
    wrist_pos_rel: torch.Tensor = env._cached_wrist_pose_npz_simple["wrist_pos_rel"]  # (3,)
    wrist_rot_rel: torch.Tensor = env._cached_wrist_pose_npz_simple["wrist_rot_rel"]  # (3,3)
    obj = env.scene["object"]
    obj_pos_w = obj.data.root_pos_w  # (N,3)
    obj_quat_w = obj.data.root_quat_w  # (N,4)
    wrist_pos_rel_expand = wrist_pos_rel.unsqueeze(0).repeat(env.num_envs, 1)  # (N,3)
    wrist_quat_rel = wrist_rot_rel.unsqueeze(0).repeat(env.num_envs, 1)  # (N,4)
    # Map demo wrist orientation to robot wrist orientation (orientation only)
    demo2robot_R = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]], device=env.device, dtype=torch.float32
    )  # (1,3,3)
    q_map = quat_from_matrix(demo2robot_R).repeat(env.num_envs, 1)  # (N,4)
    wrist_quat_rel = quat_mul(wrist_quat_rel, q_map)
    wrist_pos_w = quat_apply(obj_quat_w, wrist_pos_rel_expand) + obj_pos_w
    # apply a small lift along world z-axis
    wrist_pos_w[:, 2] += float(z_offset)
    wrist_quat_w = quat_mul(obj_quat_w, wrist_quat_rel)
    target_pos_w[env_ids] = wrist_pos_w[env_ids]
    target_quat_w[env_ids] = wrist_quat_w[env_ids]

    root_pose_w = robot.data.root_pose_w  # [N, 7]
    root_pos_w = root_pose_w[:, 0:3]
    root_quat_w = root_pose_w[:, 3:7]
    target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)

    # Optional visualization markers (world-frame). Cache to avoid creating new prims every reset.
    ee_marker = None
    goal_marker = None
    if visualize:
        ee_marker = _get_or_create_frame_marker(env, "ee", frame_scale, "/Visuals/Reset/EE")
        goal_marker = _get_or_create_frame_marker(env, "goal", frame_scale, "/Visuals/Reset/Goal")

    # Prepare IK command buffer (7 for absolute pose: x y z qw qx qy qz)
    ik_cmds = torch.zeros(env.num_envs, ik_controller.action_dim, device=env.device, dtype=torch.float)
    # Fill only for env_ids; others remain zeros and will be set to no-op below each step
    # target_pos_b[:, 0] = -0.5
    # target_pos_b[:, 1] = 0.1
    # target_pos_b[:, 2] = 0.5
    ik_cmds[env_ids, 0:3] = target_pos_b[env_ids]
    ik_cmds[env_ids, 3:7] = target_quat_b[env_ids]

    # Normalize env ids tensor for consistent indexing and save object state before IK
    env_ids_t = resolve_env_ids(env, env_ids)
    saved_obj_pose = torch.cat(
        [obj.data.root_pos_w[env_ids_t], obj.data.root_quat_w[env_ids_t]], dim=-1
    ).clone()  # (K,7)
    saved_obj_vel = torch.cat(
        [obj.data.root_lin_vel_w[env_ids_t], obj.data.root_ang_vel_w[env_ids_t]], dim=-1
    ).clone()  # (K,6)

    # before IK, reset the robot joints to default state
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    if instant_set:
        # Current EE pose and jacobian
        # Optimized: reduce iterations (now 3 instead of 10) - most IK convergence happens in first few iterations
        # NOTE: sim.step() steps ALL environments, not just env_ids. This is a simulator limitation.
        # Non-resetting environments will advance in time, but their joint states are not modified.
        for i in range(num_ik_iters):
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, body_id]  # [N, 7]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, joint_ids]
            # EE pose in base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # Compute single-shot IK and write state
            ik_controller.set_command(ik_cmds)
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            joint_vel_full = torch.zeros_like(joint_pos_des)
            # Only modify joint states for environments being reset
            joint_pos_des = joint_pos_des[env_ids, ...]
            joint_vel_full = joint_vel_full[env_ids, ...]
            robot.write_joint_state_to_sim(joint_pos_des, joint_vel_full, env_ids=env_ids, joint_ids=joint_ids)
            env.scene.write_data_to_sim()
            # WARNING: This steps ALL environments, not just env_ids. Non-resetting envs advance in time.
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            if visualize and ee_marker is not None and goal_marker is not None:
                ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                goal_marker.visualize(target_pos_w, target_quat_w)

        # Restore object to its saved pose/velocity (preserve randomized placement)
        obj.write_root_pose_to_sim(saved_obj_pose, env_ids=env_ids_t)
        obj.write_root_velocity_to_sim(saved_obj_vel, env_ids=env_ids_t)

        # Get final joint positions after IK (from last iteration)
        final_joint_pos = robot.data.joint_pos[env_ids_t].clone()
        final_joint_vel = torch.zeros_like(final_joint_pos)

        # Reset finger joints to their default positions to remove IK disturbances
        hand_joint_ids_list, default_hand_pos, default_hand_vel = _get_hand_joint_defaults(
            robot, env_ids_t, joint_ids, env.device
        )
        if len(hand_joint_ids_list) > 0:
            final_joint_pos[:, hand_joint_ids_list] = default_hand_pos
            final_joint_vel[:, hand_joint_ids_list] = default_hand_vel

        # Write final joint state (arm + hand) to ensure everything is set correctly
        robot.write_joint_state_to_sim(final_joint_pos, final_joint_vel, env_ids=env_ids_t)

        # Clear any residual joint targets and forces to prevent unwanted motion
        # Reset actuators and clear external forces/torques
        robot.reset(env_ids=env_ids_t)
        # Set position targets to current positions (so PD controller holds them)
        robot.set_joint_position_target(final_joint_pos, env_ids=env_ids_t)
        # Set velocity targets to zero
        robot.set_joint_velocity_target(final_joint_vel, env_ids=env_ids_t)
        # Zero out any effort/torque targets
        zero_efforts = torch.zeros_like(final_joint_pos)
        robot.set_joint_effort_target(zero_efforts, env_ids=env_ids_t)

        # Write all data to sim and do a final step to ensure physics is settled
        # WARNING: This steps ALL environments, not just env_ids. Non-resetting envs advance in time.
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
    else:
        # Drive over time using joint position targets
        physics_dt = env.physics_dt
        sim_time = 0.0
        while sim_time < max_time:
            # Current EE pose and jacobian
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, body_id]  # [N, 7]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, joint_ids]
            # EE pose in base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # For non-target envs, set command to current so they no-op
            not_env_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            not_env_mask[env_ids] = False
            if not_env_mask.any():
                ik_cmds[not_env_mask, 0:3] = ee_pos_b[not_env_mask]
                ik_cmds[not_env_mask, 3:7] = ee_quat_b[not_env_mask]
            # Compute IK and set joint position targets for selected envs
            ik_controller.set_command(ik_cmds)
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(joint_pos_des[env_ids], joint_ids=joint_ids, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=physics_dt)
            sim_time += physics_dt
            # visualize frames (world)
            if visualize and ee_marker is not None and goal_marker is not None:
                ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                goal_marker.visualize(target_pos_w, target_quat_w)
        # Zero velocities for affected envs to avoid residual motion
        joint_pos_cur = robot.data.joint_pos[env_ids_t].clone()
        joint_vel_cur = robot.data.joint_vel[env_ids_t].clone()
        joint_vel_cur[:, :] = 0.0
        robot.write_joint_state_to_sim(joint_pos_cur, joint_vel_cur, env_ids=env_ids_t)
        # Restore object to its saved pose/velocity (preserve randomized placement)
        obj.write_root_pose_to_sim(saved_obj_pose, env_ids=env_ids_t)
        obj.write_root_velocity_to_sim(saved_obj_vel, env_ids=env_ids_t)
        # Reset finger joints to their default positions to remove IK disturbances
        hand_joint_ids_list, default_hand_pos, default_hand_vel = _get_hand_joint_defaults(
            robot, env_ids_t, joint_ids, env.device
        )
        if len(hand_joint_ids_list) > 0:
            robot.write_joint_state_to_sim(
                default_hand_pos, default_hand_vel, env_ids=env_ids_t, joint_ids=hand_joint_ids_list
            )


def visualize_wrist_pose_from_file_simple(
    env,
    env_ids: torch.Tensor,
    resolved_scene_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    file_path: str = "",
    points_key: str = "relative_points",
    wrist_rot_key: str = "relative_wrist_rot",
    visualize: bool = True,
    frame_scale: tuple[float, float, float] = (0.1, 0.1, 0.1),
):
    """
    Simple reset event: load wrist pose in object frame, transform to world via current object pose,
    compute wrist pose in robot base frame, and visualize frame(s).
    """
    # load and cache file
    if (
        not hasattr(env, "_cached_wrist_pose_npz_simple")
        or env._cached_wrist_pose_npz_simple.get("file", "") != file_path
    ):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.endswith(".npz"):
            data = np.load(file_path)
        elif file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        if points_key not in data or wrist_rot_key not in data:
            raise KeyError(
                f"Required keys not in npz. Have: {list(data.keys())}, need: '{points_key}', '{wrist_rot_key}'"
            )
        env._cached_wrist_pose_npz_simple = {
            "file": file_path,
            "wrist_pos_rel": torch.tensor(data[points_key], dtype=torch.float32, device=env.device),  # (3,)
            "wrist_rot_rel": torch.tensor(data[wrist_rot_key], dtype=torch.float32, device=env.device),  # (3,3)
        }
    wrist_pos_rel: torch.Tensor = env._cached_wrist_pose_npz_simple["wrist_pos_rel"]
    wrist_rot_rel: torch.Tensor = env._cached_wrist_pose_npz_simple["wrist_rot_rel"]

    # scene references
    obj = env.scene[object_cfg.name]
    robot = env.scene[resolved_scene_cfg.name]

    # expand to all envs and compose to world
    wrist_pos_rel_expand = wrist_pos_rel.unsqueeze(0).repeat(env.num_envs, 1)  # (N,3)
    # Convert rotation matrix (3x3) to quaternion (4,)
    # wrist_rot_rel is already a tensor from the cache, but check its shape
    if isinstance(wrist_rot_rel, torch.Tensor):
        wrist_rot_rel_shape = wrist_rot_rel.shape
    else:
        wrist_rot_rel_shape = np.array(wrist_rot_rel).shape
        wrist_rot_rel = torch.tensor(wrist_rot_rel, dtype=torch.float32, device=env.device)

    if wrist_rot_rel_shape == (3, 3):
        # It's a rotation matrix, convert to quaternion
        wrist_rot_rel_tensor = (
            wrist_rot_rel.unsqueeze(0) if wrist_rot_rel.dim() == 2 else wrist_rot_rel
        )  # (1, 3, 3) or (3, 3)
        if wrist_rot_rel_tensor.dim() == 2:
            wrist_rot_rel_tensor = wrist_rot_rel_tensor.unsqueeze(0)  # (1, 3, 3)
        wrist_quat_rel = quat_from_matrix(wrist_rot_rel_tensor).repeat(env.num_envs, 1)  # (N, 4)
    else:
        # Already a quaternion or different format
        if wrist_rot_rel.dim() == 1 or wrist_rot_rel.shape[0] != env.num_envs:
            wrist_quat_rel = wrist_rot_rel.unsqueeze(0).repeat(env.num_envs, 1)  # (N, 4)
        else:
            wrist_quat_rel = wrist_rot_rel
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    wrist_pos_w = quat_apply(obj_quat_w, wrist_pos_rel_expand) + obj_pos_w
    wrist_quat_w = quat_mul(obj_quat_w, wrist_quat_rel)
    # convert to base frame (for logging / debug)
    root_pose_w = robot.data.root_pose_w
    root_pos_w = root_pose_w[:, 0:3]
    root_quat_w = root_pose_w[:, 3:7]
    wrist_pos_b, wrist_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, wrist_pos_w, wrist_quat_w)

    # Get actual robot end-effector link pose for visualization
    # Try common end-effector link names (order matters - try most specific first)
    ee_link_body_ids = []
    ee_body_candidates = ["palm_link", ".*ee.*", ".*wrist.*", ".*end.*effector.*"]

    for candidate in ee_body_candidates:
        try:
            candidate_ids, _ = robot.find_bodies(candidate)
            if len(candidate_ids) > 0:
                ee_link_body_ids = candidate_ids
                break
        except ValueError:
            # Pattern didn't match, try next candidate
            continue

    if len(ee_link_body_ids) > 0:
        ee_body_id = ee_link_body_ids[0]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]  # [N, 7]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]
        # Transform to base frame for visualization
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    else:
        # Fallback: use root pose if no end-effector link found
        ee_pos_b = root_pos_w
        ee_quat_b = root_quat_w

    # visualize one frame per env at the world pose and the object frame
    if visualize:
        wrist_marker = _get_or_create_frame_marker(env, "wrist_simple", frame_scale, "/Visuals/Reset/WristSimple")
        wrist_marker.visualize(translations=wrist_pos_b, orientations=wrist_quat_b)
        # Visualize actual robot ee_link pose
        ee_marker = _get_or_create_frame_marker(env, "ee_link_actual", frame_scale, "/Visuals/Reset/EEActual")
        ee_marker.visualize(translations=ee_pos_b, orientations=ee_quat_b)

    # Store per-env wrist targets for later IK reset usage
    env_ids_t = resolve_env_ids(env, env_ids)
    if not hasattr(env, "_reset_targets"):
        env._reset_targets = {}
        env._reset_targets["wrist_pos_w"] = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        env._reset_targets["wrist_quat_w"] = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
    if "wrist_pos_w" not in env._reset_targets or env._reset_targets["wrist_pos_w"].shape[0] != env.num_envs:
        env._reset_targets["wrist_pos_w"] = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    if "wrist_quat_w" not in env._reset_targets or env._reset_targets["wrist_quat_w"].shape[0] != env.num_envs:
        env._reset_targets["wrist_quat_w"] = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
    env._reset_targets["wrist_pos_w"][env_ids_t] = wrist_pos_w[env_ids_t]
    env._reset_targets["wrist_quat_w"][env_ids_t] = wrist_quat_w[env_ids_t]


def reset_environment_background(
    env,
    env_ids: torch.Tensor,
    hdr_paths: list[str] | None = None,
    indoor_intensity_range: tuple[float, float] = (800.0, 3000.0),
    outdoor_intensity_range: tuple[float, float] = (1500.0, 4000.0),
    exposure_range: tuple[float, float] = (-1.0, 1.0),
    color_temperature_mean_std: tuple[float, float] = (6500.0, 500.0),
    light_prim_path: str = "/World/skyLight",
):
    """Randomize the dome light HDR texture and lighting parameters on reset.

    The dome light is a single global prim shared across all envs, so this
    samples one HDR per reset call and applies it scene-wide. Indoor vs
    outdoor intensity is keyed off the substrings ``indoor`` / ``outdoor``
    appearing (case-insensitively) in the HDR filename.
    """
    if hdr_paths is None or len(hdr_paths) == 0:
        hdr_paths = [str(DEBUG_HDR_ASSET_PATH)]

    hdr_path = str(np.random.choice(hdr_paths))

    name_lower = os.path.basename(hdr_path).lower()
    if "outdoor" in name_lower:
        intensity = float(np.random.uniform(*outdoor_intensity_range))
    else:
        intensity = float(np.random.uniform(*indoor_intensity_range))

    exposure = float(np.random.uniform(*exposure_range))
    color_temp = float(np.random.normal(color_temperature_mean_std[0], color_temperature_mean_std[1]))

    # Route writes through Kit's ChangeProperty command. Raw
    # ``prim.GetAttribute(...).Set(...)`` does not always propagate through
    # Fabric to the RTX renderer, so the dome light prim would update in USD
    # but the camera kept rendering with the old intensity/texture.
    # ChangeProperty fires the notifications Fabric listens to and RTX picks
    # the change up on the next render.
    import omni.kit.commands  # local import: Kit is only available post-AppLauncher.

    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{light_prim_path}.inputs:texture:file",
        value=hdr_path,
        prev=None,
    )
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{light_prim_path}.inputs:intensity",
        value=intensity,
        prev=None,
    )
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{light_prim_path}.inputs:exposure",
        value=exposure,
        prev=None,
    )
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{light_prim_path}.inputs:colorTemperature",
        value=color_temp,
        prev=None,
    )
