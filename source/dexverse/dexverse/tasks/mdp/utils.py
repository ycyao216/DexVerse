# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import logging
import math

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch
import trimesh
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import get_all_matching_child_prims
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from pxr import UsdGeom
from trimesh.sample import sample_surface

# ---- module-scope caches ----
_PRIM_SAMPLE_CACHE: dict[tuple[str, int], np.ndarray] = {}  # (prim_hash, num_points) -> (N,3) in root frame
_FINAL_SAMPLE_CACHE: dict[str, np.ndarray] = {}  # env_hash -> (num_points,3) in root frame


def normalize_axis(
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized 3D axis tensor on the requested device/dtype."""
    axis = torch.as_tensor(axis_local, device=device, dtype=dtype).flatten()
    if axis.numel() != 3:
        raise ValueError(f"Expected axis_local with 3 elements, got shape {tuple(axis.shape)}.")
    return axis / torch.clamp(torch.linalg.norm(axis), min=1e-6)


def axis_in_world_from_quat(
    quat_w: torch.Tensor,
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Rotate a local axis by world-frame quaternion(s), returning a world-frame axis per env."""
    axis = normalize_axis(axis_local, device=quat_w.device, dtype=quat_w.dtype)
    axis_batch = axis.unsqueeze(0).expand(quat_w.shape[0], -1)
    return quat_apply(quat_w, axis_batch)


def root_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Read root quaternion in world frame for a rigid/articulated asset."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def command_quat_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get command orientation in world frame regardless of command frame setting."""
    command = env.command_manager.get_command(command_name)
    command_quat = normalize(command[:, 3:7])
    command_term = env.command_manager.get_term(command_name)
    if getattr(command_term.cfg, "use_world_frame", False):
        return command_quat
    return quat_mul(root_quat_w(env, robot_cfg), command_quat)


def command_axis_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get a commanded local axis expressed in world frame."""
    return axis_in_world_from_quat(command_quat_w(env, command_name, robot_cfg), axis_local)


def asset_axis_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Get an asset-local axis expressed in world frame."""
    quat_w = root_quat_w(env, asset_cfg)
    return axis_in_world_from_quat(quat_w, axis_local)


def world_points_from_local(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    local_points: torch.Tensor,
) -> torch.Tensor:
    """Transform ``(N, 3)`` asset-local points to world frame, returning ``(E, N, 3)``."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w  # (E, 3)
    quat_w = asset.data.root_quat_w  # (E, 4)
    E = pos_w.shape[0]
    N = local_points.shape[0]
    if N == 0:
        return torch.zeros(E, 0, 3, device=pos_w.device, dtype=pos_w.dtype)
    pts = local_points.unsqueeze(0).expand(E, -1, -1).reshape(-1, 3)
    quat = quat_w.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
    rotated = quat_apply(quat, pts).reshape(E, N, 3)
    return pos_w.unsqueeze(1) + rotated


def axis_in_frame_from_quat(
    axis_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame axis vectors in a target frame defined by quaternion(s)."""
    return quat_apply_inverse(frame_quat_w, axis_w)


def axis_alignment_dot(axis_a_w: torch.Tensor, axis_b_w: torch.Tensor) -> torch.Tensor:
    """Signed cosine similarity between two world-frame axis tensors."""
    return torch.sum(axis_a_w * axis_b_w, dim=1).clamp(-1.0, 1.0)


def vector_projection_on_axis(vector_w: torch.Tensor, axis_w: torch.Tensor) -> torch.Tensor:
    """Projection magnitude of vectors onto axes in world frame."""
    return torch.sum(vector_w * axis_w, dim=1)


def resolve_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | list[int] | tuple[int, ...] | None,
) -> torch.Tensor:
    """Normalize env ids to a contiguous tensor on env.device."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def resolve_joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Resolve SceneEntityCfg.joint_ids from joint_names when needed and return joint_ids."""
    if (
        isinstance(asset_cfg.joint_ids, slice)
        and asset_cfg.joint_ids == slice(None)
        and asset_cfg.joint_names is not None
    ):
        asset_cfg.resolve(env.scene)
    return asset_cfg.joint_ids


def get_init_joint_pos(asset: Articulation, joint_ids) -> torch.Tensor:
    """Initial joint positions honoring cfg.init_state.joint_pos overrides."""
    init_joint_pos = asset.data.default_joint_pos.clone()
    init_cfg = getattr(asset.cfg.init_state, "joint_pos", None) or {}
    if init_cfg:
        name_to_id = {name: idx for idx, name in enumerate(asset.joint_names)}
        if isinstance(joint_ids, slice):
            target_ids = None
        else:
            if torch.is_tensor(joint_ids):
                target_ids = set(joint_ids.tolist())
            else:
                target_ids = set(joint_ids)
        for name, value in init_cfg.items():
            jid = name_to_id.get(name)
            if jid is None:
                continue
            if target_ids is not None and jid not in target_ids:
                continue
            init_joint_pos[:, jid] = float(value)
    return init_joint_pos[:, joint_ids]


def root_height_delta(asset: Articulation | RigidObject) -> torch.Tensor:
    """Root z-height delta relative to default root state."""
    return asset.data.root_pos_w[:, 2] - asset.data.default_root_state[:, 2]


def axis_tilt_angle(
    quat_w: torch.Tensor,
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
    world_axis: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Angle (rad) between rotated local axis and world axis for each env."""
    axis_w = axis_in_world_from_quat(quat_w, axis_local=axis_local)
    world_axis_t = normalize_axis(world_axis, device=quat_w.device, dtype=quat_w.dtype)
    world_axis_w = world_axis_t.unsqueeze(0).expand_as(axis_w)
    cos_angle = torch.sum(axis_w * world_axis_w, dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle)


def axis_to_plane_angle(
    quat_w: torch.Tensor,
    axis_local: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 1.0, 0.0),
    plane_normal: tuple[float, float, float] | list[float] | torch.Tensor = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Angle (rad) between the rotated local axis and the plane with normal ``plane_normal``.

    Returns values in ``[0, π/2]``: ``0`` when the axis lies in the plane,
    ``π/2`` when the axis is perpendicular to it. The sign of the dot product
    is discarded, so the metric is symmetric about the plane (an axis tilted
    30° above and 30° below the plane both return 30°). Useful as an
    alternative tilt metric for "has the object's chosen axis tipped enough
    out of the ground plane to count" success criteria, where the choice of
    ``axis_local`` decides which physical axis you're tracking.
    """
    axis_w = axis_in_world_from_quat(quat_w, axis_local=axis_local)
    n = normalize_axis(plane_normal, device=quat_w.device, dtype=quat_w.dtype)
    n_w = n.unsqueeze(0).expand_as(axis_w)
    sin_angle = torch.sum(axis_w * n_w, dim=1).abs().clamp(0.0, 1.0)
    return torch.asin(sin_angle)


def tracked_axis_rotation_step(
    env: ManagerBasedRLEnv,
    quat_w: torch.Tensor,
    axis_w: torch.Tensor,
    tracker_name: str = "default",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track cumulative signed rotation around an axis and return (accumulated, delta).

    The tracker is idempotent within one env step: repeated calls at the same
    ``progress_buf`` value return cached results without double-counting.
    """
    if quat_w.ndim != 2 or quat_w.shape[1] != 4:
        raise ValueError(f"Expected quat_w with shape (N, 4), got {tuple(quat_w.shape)}.")

    num_envs = quat_w.shape[0]
    if axis_w.ndim == 1:
        axis_w = axis_w.unsqueeze(0).expand(num_envs, -1)
    if axis_w.ndim != 2 or axis_w.shape[1] != 3 or axis_w.shape[0] != num_envs:
        raise ValueError(f"Expected axis_w with shape (N, 3), got {tuple(axis_w.shape)}.")
    axis_w = axis_w / torch.clamp(torch.linalg.norm(axis_w, dim=1, keepdim=True), min=1e-6)

    trackers = getattr(env, "_axis_rotation_trackers", None)
    if trackers is None:
        trackers = {}
        setattr(env, "_axis_rotation_trackers", trackers)

    tracker = trackers.get(tracker_name)
    needs_init = (
        tracker is None
        or tracker["accumulated_angle"].shape[0] != num_envs
        or tracker["accumulated_angle"].device != quat_w.device
        or tracker["accumulated_angle"].dtype != quat_w.dtype
    )
    if needs_init:
        tracker = {
            "accumulated_angle": torch.zeros(num_envs, device=quat_w.device, dtype=quat_w.dtype),
            "delta_angle": torch.zeros(num_envs, device=quat_w.device, dtype=quat_w.dtype),
            "prev_quat_w": quat_w.clone(),
            "last_progress": torch.full((num_envs,), -1, device=quat_w.device, dtype=torch.long),
        }
        trackers[tracker_name] = tracker

    progress_buf = getattr(env, "progress_buf", None)
    if progress_buf is None:
        progress = torch.zeros(num_envs, device=quat_w.device, dtype=torch.long)
    else:
        progress = progress_buf.to(device=quat_w.device, dtype=torch.long)

    new_step = progress != tracker["last_progress"]
    if not torch.any(new_step):
        return tracker["accumulated_angle"], tracker["delta_angle"]

    tracker["delta_angle"][new_step] = 0.0

    reset_mask = new_step & (progress == 0)
    if torch.any(reset_mask):
        tracker["accumulated_angle"][reset_mask] = 0.0
        tracker["prev_quat_w"][reset_mask] = quat_w[reset_mask]

    step_mask = new_step & (~reset_mask)
    if torch.any(step_mask):
        prev_quat = tracker["prev_quat_w"][step_mask]
        curr_quat = quat_w[step_mask]
        delta_quat = quat_mul(curr_quat, quat_inv(prev_quat))

        # Enforce shortest-path quaternion before extracting angle.
        neg_w_mask = delta_quat[:, 0] < 0.0
        delta_quat[neg_w_mask] = -delta_quat[neg_w_mask]

        vec = delta_quat[:, 1:4]
        vec_norm = torch.linalg.norm(vec, dim=1)
        scalar = torch.clamp(delta_quat[:, 0], min=1e-6)
        angle = 2.0 * torch.atan2(vec_norm, scalar)
        angle = torch.clamp(angle, 0.0, math.pi)

        axis_delta = vec / torch.clamp(vec_norm.unsqueeze(1), min=1e-6)
        sign = torch.sign(torch.sum(axis_delta * axis_w[step_mask], dim=1))
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        signed_delta = angle * sign

        tracker["delta_angle"][step_mask] = signed_delta
        tracker["accumulated_angle"][step_mask] += signed_delta
        tracker["prev_quat_w"][step_mask] = curr_quat

    tracker["last_progress"][new_step] = progress[new_step]
    return tracker["accumulated_angle"], tracker["delta_angle"]


def get_or_update_joint_reference(
    env: ManagerBasedRLEnv,
    current_joint_pos: torch.Tensor,
    reference_name: str = "default",
) -> torch.Tensor:
    """Return a per-env joint reference buffer and refresh it automatically at reset."""
    if current_joint_pos.ndim != 2:
        raise ValueError(f"Expected current_joint_pos with shape (N, D), got {tuple(current_joint_pos.shape)}.")

    refs = getattr(env, "_joint_position_references", None)
    if refs is None:
        refs = {}
        setattr(env, "_joint_position_references", refs)

    ref = refs.get(reference_name)
    needs_init = (
        ref is None
        or ref.shape != current_joint_pos.shape
        or ref.device != current_joint_pos.device
        or ref.dtype != current_joint_pos.dtype
    )
    if needs_init:
        ref = current_joint_pos.clone()
        refs[reference_name] = ref

    progress_buf = getattr(env, "progress_buf", None)
    if progress_buf is None:
        reset_mask = torch.ones(current_joint_pos.shape[0], device=current_joint_pos.device, dtype=torch.bool)
    else:
        reset_mask = progress_buf.to(device=current_joint_pos.device, dtype=torch.long) == 0
    if torch.any(reset_mask):
        ref[reset_mask] = current_joint_pos[reset_mask]
    return ref


def _compute_body_components_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-body pose + velocity in the base asset's root frame (flattened per body).

    Returns ``(body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b, body_pos_w_flat)``.
    Shared by :func:`compute_body_state_b` (13/body) and
    :func:`compute_body_pose_vel_b` (pose 7/body + vel 6/body).
    """
    body_asset: Articulation | RigidObject = env.scene[body_asset_cfg.name]
    base_asset: Articulation | RigidObject = env.scene[base_asset_cfg.name]

    # Articulation path: observe requested bodies.
    if hasattr(body_asset.data, "body_pos_w"):
        body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids]
        body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids]
        body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids]
        body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids]
    # Rigid-object path: treat root as a single "body".
    else:
        body_pos_w = body_asset.data.root_pos_w.unsqueeze(1)
        body_quat_w = body_asset.data.root_quat_w.unsqueeze(1)
        body_lin_vel_w = body_asset.data.root_lin_vel_w.unsqueeze(1)
        body_ang_vel_w = body_asset.data.root_ang_vel_w.unsqueeze(1)
    num_bodies = body_pos_w.shape[1]

    if hasattr(base_asset.data, "root_link_pos_w"):
        base_pos_w = base_asset.data.root_link_pos_w
        base_quat_w = base_asset.data.root_link_quat_w
    else:
        base_pos_w = base_asset.data.root_pos_w
        base_quat_w = base_asset.data.root_quat_w

    root_pos_w = base_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1)
    root_quat_w = base_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1)

    body_pos_w_flat = body_pos_w.view(-1, 3)
    body_quat_w_flat = body_quat_w.view(-1, 4)
    body_lin_vel_w_flat = body_lin_vel_w.view(-1, 3)
    body_ang_vel_w_flat = body_ang_vel_w.view(-1, 3)
    root_pos_w_flat = root_pos_w.view(-1, 3)
    root_quat_w_flat = root_quat_w.view(-1, 4)

    body_pos_b, body_quat_b = subtract_frame_transforms(
        root_pos_w_flat, root_quat_w_flat, body_pos_w_flat, body_quat_w_flat
    )
    body_lin_vel_b = quat_apply_inverse(root_quat_w_flat, body_lin_vel_w_flat)
    body_ang_vel_b = quat_apply_inverse(root_quat_w_flat, body_ang_vel_w_flat)
    return body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b, body_pos_w_flat


def compute_body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute body states in base frame and return flattened world positions for optional visualization.

    Per-body layout is ``[pos(3), quat(4), linvel(3), angvel(3)]`` (13), concatenated over bodies.
    """
    body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b, body_pos_w_flat = _compute_body_components_b(
        env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg
    )
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1).view(env.num_envs, -1)
    return out, body_pos_w_flat


def compute_body_pose_vel_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split body state into pose and velocity in the base asset's root frame.

    Returns ``(pose_b, vel_b, body_pos_w_flat)``. Per-body layout is
    ``pose_b = [pos(3), quat(4)]`` (7) and ``vel_b = [linvel(3), angvel(3)]`` (6),
    each concatenated over bodies. Lets observations route observable pose into
    ``state`` and sim-only velocity into ``privileged``.
    """
    body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b, body_pos_w_flat = _compute_body_components_b(
        env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg
    )
    pose_b = torch.cat((body_pos_b, body_quat_b), dim=1).view(env.num_envs, -1)
    vel_b = torch.cat((body_lin_vel_b, body_ang_vel_b), dim=1).view(env.num_envs, -1)
    return pose_b, vel_b, body_pos_w_flat


def factory_insert_success_mask(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    held_base_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_dist_thresh: float = 0.0025,
    z_threshold: float = 0.001,
    held_base_local_offset_2: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Factory-style success test: XY centered and sufficiently inserted in Z.

    When ``held_base_local_offset_2`` is given, the held asset is tested at
    *both* reference points (e.g. the two ends of a pen) and the result passes
    if *either* end is centered over and inserted below the target — so the
    object counts as inserted whichever way round it goes in.
    """
    held: Articulation | RigidObject = env.scene[held_cfg.name]
    fixed: Articulation | RigidObject = env.scene[fixed_cfg.name]

    held_pos = held.data.root_pos_w
    held_quat = held.data.root_quat_w
    fixed_pos = fixed.data.root_pos_w
    fixed_quat = fixed.data.root_quat_w

    target_offset = (
        torch.tensor(target_local_offset, device=env.device, dtype=fixed_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    )
    target_pos = fixed_pos + quat_apply(fixed_quat, target_offset)

    held_local_offsets = [held_base_local_offset]
    if held_base_local_offset_2 is not None:
        held_local_offsets.append(held_base_local_offset_2)

    mask: torch.Tensor | None = None
    for local_offset in held_local_offsets:
        held_offset = (
            torch.tensor(local_offset, device=env.device, dtype=held_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
        )
        held_base_pos = held_pos + quat_apply(held_quat, held_offset)

        xy_dist = torch.linalg.vector_norm(target_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_pos[:, 2]
        centered = xy_dist < float(center_dist_thresh)
        close_or_below = z_disp < float(z_threshold)
        end_mask = torch.logical_and(centered, close_or_below)
        mask = end_mask if mask is None else torch.logical_or(mask, end_mask)

    return mask


def plug_charger_insertion_mask(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    held_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    insertion_x_threshold: float = 0.02,
    lateral_y_threshold: float = 0.01,
    vertical_z_threshold: float = 0.01,
) -> torch.Tensor:
    """Insertion-like success mask in receptacle frame.

    In receptacle frame:
    - x >= insertion_x_threshold
    - |y| <= lateral_y_threshold
    - |z| <= vertical_z_threshold
    """
    held: Articulation | RigidObject = env.scene[held_cfg.name]
    fixed: Articulation | RigidObject = env.scene[fixed_cfg.name]

    held_pos = held.data.root_pos_w
    held_quat = held.data.root_quat_w
    fixed_pos = fixed.data.root_pos_w
    fixed_quat = fixed.data.root_quat_w

    held_offset = torch.tensor(held_local_offset, device=env.device, dtype=held_pos.dtype).unsqueeze(0)
    fixed_offset = torch.tensor(fixed_local_offset, device=env.device, dtype=fixed_pos.dtype).unsqueeze(0)

    held_point_w = held_pos + quat_apply(held_quat, held_offset.repeat(held_pos.shape[0], 1))
    fixed_point_w = fixed_pos + quat_apply(fixed_quat, fixed_offset.repeat(fixed_pos.shape[0], 1))
    rel_pos_w = held_point_w - fixed_point_w
    held_pos_at_fixed = quat_apply_inverse(fixed_quat, rel_pos_w)

    # Success requires the sampled charger point's x in receptacle frame to be
    # larger than insertion_x_threshold (e.g., threshold can be negative).
    x_flag = held_pos_at_fixed[:, 0] >= float(insertion_x_threshold)
    y_flag = torch.abs(held_pos_at_fixed[:, 1]) <= float(lateral_y_threshold)
    z_flag = torch.abs(held_pos_at_fixed[:, 2]) <= float(vertical_z_threshold)
    return x_flag & y_flag & z_flag


def insert_peg_head_in_hole_frame(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    peg_head_local_offset: tuple[float, float, float] = (0.10, 0.0, 0.0),
    hole_center_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return peg-head position expressed in hole local frame.

    This mirrors ManiSkill `PegInsertionSide-v1`:
    `peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p`
    """
    peg: Articulation | RigidObject = env.scene[peg_cfg.name]
    hole: Articulation | RigidObject = env.scene[hole_cfg.name]

    peg_pos = peg.data.root_pos_w
    peg_quat = peg.data.root_quat_w
    hole_pos = hole.data.root_pos_w
    hole_quat = hole.data.root_quat_w

    peg_head_local = torch.tensor(peg_head_local_offset, device=env.device, dtype=peg_pos.dtype).unsqueeze(0)
    hole_center_local = torch.tensor(hole_center_local_offset, device=env.device, dtype=hole_pos.dtype).unsqueeze(0)

    peg_head_pos_w = peg_pos + quat_apply(peg_quat, peg_head_local.repeat(peg_pos.shape[0], 1))
    hole_center_pos_w = hole_pos + quat_apply(hole_quat, hole_center_local.repeat(hole_pos.shape[0], 1))

    rel_pos_w = peg_head_pos_w - hole_center_pos_w
    return quat_apply_inverse(hole_quat, rel_pos_w)


def insert_peg_success_mask(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    peg_head_local_offset: tuple[float, float, float] = (0.10, 0.0, 0.0),
    hole_center_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hole_radius: float = 0.023,
    insertion_x_threshold: float = 0.015,
) -> torch.Tensor:
    """ManiSkill-style success mask for side peg insertion.

    In hole frame:
    - x >= -insertion_x_threshold
    - |y| <= hole_radius
    - |z| <= hole_radius
    """
    peg_head_pos_at_hole = insert_peg_head_in_hole_frame(
        env=env,
        peg_cfg=peg_cfg,
        hole_cfg=hole_cfg,
        peg_head_local_offset=peg_head_local_offset,
        hole_center_local_offset=hole_center_local_offset,
    )
    x_flag = peg_head_pos_at_hole[:, 0] >= -float(insertion_x_threshold)
    y_flag = torch.abs(peg_head_pos_at_hole[:, 1]) <= float(hole_radius)
    z_flag = torch.abs(peg_head_pos_at_hole[:, 2]) <= float(hole_radius)
    return x_flag & y_flag & z_flag


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Samples point clouds for each environment instance by collecting points
    from all matching USD prims under `prim_path`, then downsamples to
    exactly `num_points` per env using farthest-point sampling.

    Caching is in-memory within this module:
      - per-prim raw samples:   _PRIM_SAMPLE_CACHE[(prim_hash, num_points)]
      - final downsampled env:  _FINAL_SAMPLE_CACHE[env_hash]

    Returns:
        torch.Tensor: Shape (num_envs, num_points, 3) on `device`.
    """

    points = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    xform_cache = UsdGeom.XformCache()

    for i in range(num_envs):
        # Resolve prim path
        obj_path = prim_path.replace(".*", str(i))

        # Gather prims
        prims = get_all_matching_child_prims(
            obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        )
        if not prims:
            raise KeyError(f"No valid prims under {obj_path}")

        object_prim = prim_utils.get_prim_at_path(obj_path)
        world_root = xform_cache.GetLocalToWorldTransform(object_prim)

        # hash each child prim by its rel transform + geometry
        prim_hashes = []
        for prim in prims:
            prim_type = prim.GetTypeName()
            hasher = hashlib.sha256()

            rel = world_root.GetInverse() * xform_cache.GetLocalToWorldTransform(prim)  # prim -> root
            mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
            hasher.update(mat_np.tobytes())

            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                hasher.update(verts.tobytes())
            else:
                if prim_type == "Cube":
                    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                    hasher.update(np.float32(size).tobytes())
                elif prim_type == "Sphere":
                    r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    hasher.update(np.float32(r).tobytes())
                elif prim_type == "Cylinder":
                    c = UsdGeom.Cylinder(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Capsule":
                    c = UsdGeom.Capsule(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Cone":
                    c = UsdGeom.Cone(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())

            prim_hashes.append(hasher.hexdigest())

        # scale on root (default to 1 if missing)
        attr = object_prim.GetAttribute("xformOp:scale")
        scale_val = attr.Get() if attr else None
        if scale_val is None:
            base_scale = torch.ones(3, dtype=torch.float32, device=device)
        else:
            base_scale = torch.tensor(scale_val, dtype=torch.float32, device=device)

        # env-level cache key (includes num_points)
        env_key = "_".join(sorted(prim_hashes)) + f"_{num_points}"
        env_hash = hashlib.sha256(env_key.encode()).hexdigest()

        # load from env-level in-memory cache
        if env_hash in _FINAL_SAMPLE_CACHE:
            arr = _FINAL_SAMPLE_CACHE[env_hash]  # (num_points,3) in root frame
            points[i] = torch.from_numpy(arr).to(device) * base_scale.unsqueeze(0)
            continue

        # otherwise build per-prim samples (with per-prim cache)
        all_samples_np: list[np.ndarray] = []
        for prim, ph in zip(prims, prim_hashes):
            key = (ph, num_points)
            if key in _PRIM_SAMPLE_CACHE:
                samples = _PRIM_SAMPLE_CACHE[key]
            else:
                prim_type = prim.GetTypeName()
                if prim_type == "Mesh":
                    mesh = UsdGeom.Mesh(prim)
                    verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    faces = _triangulate_faces(prim)
                    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                else:
                    mesh_tm = create_primitive_mesh(prim)

                face_weights = mesh_tm.area_faces
                samples_np, _ = sample_surface(mesh_tm, num_points * 2, face_weight=face_weights)

                # FPS to num_points on chosen device
                tensor_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)
                prim_idxs = farthest_point_sampling(tensor_pts, num_points)
                local_pts = tensor_pts[prim_idxs]

                # prim -> root transform
                rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
                mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
                mat_t = torch.from_numpy(mat_np).to(device)

                ones = torch.ones((num_points, 1), device=device)
                pts_h = torch.cat([local_pts, ones], dim=1)
                root_h = pts_h @ mat_t
                samples = root_h[:, :3].detach().cpu().numpy()

                if prim_type == "Cone":
                    samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2

                _PRIM_SAMPLE_CACHE[key] = samples  # cache in root frame @ num_points

            all_samples_np.append(samples)

        # combine & env-level FPS (if needed)
        if len(all_samples_np) == 1:
            samples_final = torch.from_numpy(all_samples_np[0]).to(device)
        else:
            combined = torch.from_numpy(np.concatenate(all_samples_np, axis=0)).to(device)
            idxs = farthest_point_sampling(combined, num_points)
            samples_final = combined[idxs]

        # store env-level cache in root frame (CPU)
        _FINAL_SAMPLE_CACHE[env_hash] = samples_final.detach().cpu().numpy()

        # apply root scale and write out
        points[i] = samples_final * base_scale.unsqueeze(0)

    return points


def _triangulate_faces(prim) -> np.ndarray:
    """Convert a USD Mesh prim into triangulated face indices (N, 3)."""

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim):
    """Create a trimesh mesh from a USD primitive (Cube, Sphere, Cylinder, etc.)."""
    import trimesh
    from pxr import UsdGeom

    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def farthest_point_sampling(
    points: torch.Tensor, n_samples: int, memory_threashold=2 * 1024**3
) -> torch.Tensor:  # 2 GiB
    """
    Farthest Point Sampling (FPS) for point sets.

    Selects `n_samples` points such that each new point is farthest from the
    already chosen ones. Uses a full pairwise distance matrix if memory allows,
    otherwise falls back to an iterative version.

    Args:
        points (torch.Tensor): Input points of shape (N, D).
        n_samples (int): Number of samples to select.
        memory_threashold (int): Max allowed bytes for distance matrix. Default 2 GiB.

    Returns:
        torch.Tensor: Indices of sampled points (n_samples,).
    """
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float("inf"), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def push_t_overlap_ratio(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_tee"),
    point_step: float = 0.005,
    horizontal_size: tuple[float, float] = (0.20, 0.05),
    vertical_size: tuple[float, float] = (0.05, 0.15),
    com_y: float = 0.0375,
) -> torch.Tensor:
    """Approximate PushT overlap ratio by sampling the T area in goal frame.

    The ratio is the fraction of sampled points from the movable T-shape that falls
    inside the goal T-shape after projecting both to XY plane.
    """
    moving_t: Articulation | RigidObject = env.scene[object_cfg.name]
    goal_t: Articulation | RigidObject = env.scene[goal_cfg.name]

    obj_pos_xy = moving_t.data.root_pos_w[:, :2]
    goal_pos_xy = goal_t.data.root_pos_w[:, :2]
    _, _, obj_yaw = euler_xyz_from_quat(moving_t.data.root_quat_w)
    _, _, goal_yaw = euler_xyz_from_quat(goal_t.data.root_quat_w)

    def _sample_rect_points(center_y: float, size_x: float, size_y: float) -> torch.Tensor:
        nx = max(1, int(math.ceil(float(size_x) / float(point_step))))
        ny = max(1, int(math.ceil(float(size_y) / float(point_step))))
        dx = float(size_x) / float(nx)
        dy = float(size_y) / float(ny)
        xs = torch.linspace(
            -0.5 * float(size_x) + 0.5 * dx,
            0.5 * float(size_x) - 0.5 * dx,
            nx,
            device=obj_pos_xy.device,
            dtype=obj_pos_xy.dtype,
        )
        ys = torch.linspace(
            center_y - 0.5 * float(size_y) + 0.5 * dy,
            center_y + 0.5 * float(size_y) - 0.5 * dy,
            ny,
            device=obj_pos_xy.device,
            dtype=obj_pos_xy.dtype,
        )
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    h_size_x, h_size_y = float(horizontal_size[0]), float(horizontal_size[1])
    v_size_x, v_size_y = float(vertical_size[0]), float(vertical_size[1])
    h_center_y = -float(com_y)
    v_center_y = h_center_y + 0.5 * h_size_y + 0.5 * v_size_y
    horizontal_pts = _sample_rect_points(h_center_y, h_size_x, h_size_y)
    vertical_pts = _sample_rect_points(v_center_y, v_size_x, v_size_y)
    local_points = torch.cat([horizontal_pts, vertical_pts], dim=0)

    local_x = local_points[:, 0].unsqueeze(0)
    local_y = local_points[:, 1].unsqueeze(0)
    cos_obj = torch.cos(obj_yaw).unsqueeze(1)
    sin_obj = torch.sin(obj_yaw).unsqueeze(1)
    pts_world_x = local_x * cos_obj - local_y * sin_obj + obj_pos_xy[:, 0:1]
    pts_world_y = local_x * sin_obj + local_y * cos_obj + obj_pos_xy[:, 1:2]

    delta_x = pts_world_x - goal_pos_xy[:, 0:1]
    delta_y = pts_world_y - goal_pos_xy[:, 1:2]
    cos_goal = torch.cos(goal_yaw).unsqueeze(1)
    sin_goal = torch.sin(goal_yaw).unsqueeze(1)
    pts_goal_x = delta_x * cos_goal + delta_y * sin_goal
    pts_goal_y = -delta_x * sin_goal + delta_y * cos_goal

    h_size_x, h_size_y = float(horizontal_size[0]), float(horizontal_size[1])
    v_size_x, v_size_y = float(vertical_size[0]), float(vertical_size[1])
    h_half_x = 0.5 * h_size_x
    h_half_y = 0.5 * h_size_y
    v_half_x = 0.5 * v_size_x
    v_half_y = 0.5 * v_size_y
    h_center_y = -float(com_y)
    v_center_y = h_center_y + h_half_y + v_half_y

    in_horizontal = (
        (pts_goal_x >= -h_half_x)
        & (pts_goal_x <= h_half_x)
        & (pts_goal_y >= h_center_y - h_half_y)
        & (pts_goal_y <= h_center_y + h_half_y)
    )
    in_vertical = (
        (pts_goal_x >= -v_half_x)
        & (pts_goal_x <= v_half_x)
        & (pts_goal_y >= v_center_y - v_half_y)
        & (pts_goal_y <= v_center_y + v_half_y)
    )
    inside_goal_t = in_horizontal | in_vertical
    return inside_goal_t.float().mean(dim=1)
