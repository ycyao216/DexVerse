# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .utils import (
    asset_axis_w,
    axis_alignment_dot,
    axis_tilt_angle,
    axis_to_plane_angle,
    factory_insert_success_mask,
    get_init_joint_pos,
    insert_peg_success_mask,
    plug_charger_insertion_mask,
    push_t_overlap_ratio,
    resolve_joint_ids,
    root_height_delta,
)


def _normalize_in_bound_range(
    in_bound_range: dict[str, tuple[float, float]] | None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Convert optional range dict into canonical (x, y, z) bounds tuple."""
    in_bound_range = in_bound_range or {}
    x_bounds = tuple(float(v) for v in in_bound_range.get("x", (0.0, 0.0)))
    y_bounds = tuple(float(v) for v in in_bound_range.get("y", (0.0, 0.0)))
    z_bounds = tuple(float(v) for v in in_bound_range.get("z", (0.0, 0.0)))
    return x_bounds, y_bounds, z_bounds


def _get_cached_bounds_tensor(
    env: ManagerBasedRLEnv,
    cache_key: tuple[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return cached bounds tensor on env.device/dtype to avoid per-step allocations."""
    cache = getattr(env, "_out_of_bound_ranges_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_out_of_bound_ranges_cache", cache)

    ranges = cache.get(cache_key)
    if ranges is None or ranges.device != env.device or ranges.dtype != dtype:
        ranges = torch.tensor(cache_key[1], device=env.device, dtype=dtype)
        cache[cache_key] = ranges
    return ranges


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    bounds = _normalize_in_bound_range(in_bound_range)
    cache_key = (asset_cfg.name, bounds)
    ranges = _get_cached_bounds_tensor(env, cache_key, dtype=object_pos_local.dtype)

    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def object_at_goal_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.02,
) -> torch.Tensor:
    """Termination condition for when the object is close enough to the goal position.

    This checks if the object's position is within the specified threshold of the goal position.
    Orientation is ignored - only position matters.

    Args:
        env: The environment.
        command_name: The name of the command term that contains the goal position.
        threshold: Distance threshold in meters. Defaults to 0.05.

    Returns:
        A boolean tensor indicating which environments have succeeded (object at goal position).
    """
    # Get the command term which has position_error metric
    command_term = env.command_manager.get_term(command_name)

    # Check if position error is below threshold
    position_error = command_term.metrics["position_error"]
    success = position_error < threshold

    return success


def object_at_goal_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float = 0.02,
    orientation_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when the commanded object pose is reached in position and orientation."""
    command_term = env.command_manager.get_term(command_name)
    position_error = command_term.metrics["position_error"]
    orientation_error = command_term.metrics["orientation_error"]
    return (position_error < position_threshold) & (orientation_error < orientation_threshold)


def object_upright_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float = 0.05,
    max_tilt_rad: float = 0.2617993878,  # 15 degrees
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object reaches the goal *position* and is upright.

    Position is read from the ``command_name`` term's ``position_error`` metric
    (so the goal command can stay ``position_only``). Uprightness is the angle
    between the object's local +Z and world +Z, which makes the check
    yaw-agnostic: a cup set down at any heading still counts as upright, while a
    tipped-over cup does not.
    """
    command_term = env.command_manager.get_term(command_name)
    at_goal = command_term.metrics["position_error"] < position_threshold
    obj: RigidObject = env.scene[object_cfg.name]
    angle = axis_tilt_angle(obj.data.root_quat_w, axis_local=(0.0, 0.0, 1.0), world_axis=(0.0, 0.0, 1.0))
    upright = angle <= max_tilt_rad
    return at_goal & upright


def lift_and_tilt(
    env: ManagerBasedRLEnv,
    min_height: float,
    threshold_rad: float,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tilt_ge: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object is lifted above min_height and tilt angle meets threshold."""
    object: RigidObject = env.scene[object_cfg.name]
    angle = axis_tilt_angle(object.data.root_quat_w, axis_local=axis_local, world_axis=world_axis)
    tilt_ok = angle >= threshold_rad if tilt_ge else angle <= threshold_rad
    lifted = object_lifted(env, asset_cfg=object_cfg, min_height=min_height)
    return lifted & tilt_ok


def object_upright_and_lifted(
    env: ManagerBasedRLEnv,
    min_height: float = 0.10,
    max_tilt_rad: float = 0.174532925,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object is upright (tilt <= max_tilt_rad) and lifted above init height."""
    object: RigidObject = env.scene[object_cfg.name]
    angle = axis_tilt_angle(object.data.root_quat_w, axis_local=(0.0, 0.0, 1.0), world_axis=(0.0, 0.0, 1.0))
    upright = angle <= max_tilt_rad

    lifted = root_height_delta(object) >= min_height
    return upright & lifted


def _compare(value: torch.Tensor, target: torch.Tensor | float, op: str, tol: float) -> torch.Tensor:
    if op == ">=":
        return value >= (target - tol)
    if op == ">":
        return value > (target - tol)
    if op == "<=":
        return value <= (target + tol)
    if op == "<":
        return value < (target + tol)
    raise ValueError(f"Unsupported op: {op}")


def joint_reach_threshold(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    op: str = ">=",
    ref: str = "value",
    tol: float = 0.0,
    reduce: str = "any",
) -> torch.Tensor:
    """Terminate when joint values cross a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]

    if ref == "value":
        target = threshold
    elif ref == "upper_limit":
        target = asset.data.joint_pos_limits[:, joint_ids, 1] + threshold
    elif ref == "lower_limit":
        target = asset.data.joint_pos_limits[:, joint_ids, 0] + threshold
    else:
        raise ValueError(f"Unsupported ref: {ref}")

    cmp = _compare(joint_pos, target, op, tol)
    if reduce == "any":
        return cmp.any(dim=1)
    if reduce == "all":
        return cmp.all(dim=1)
    raise ValueError(f"Unsupported reduce: {reduce}")


def joint_relative_move(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    mode: str = "displacement",
    op: str = ">=",
    reduce: str = "any",
) -> torch.Tensor:
    """Terminate when joint displacement/progress crosses a threshold.

    Supported modes:
        - "displacement": abs(q - q_init)
        - "progress": abs(q - q_init) / reachable_from_init
        - "absolute_progress" (aliases: "range_progress", "ratio"): (q - lower) / (upper - lower)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]

    init_joint_pos = get_init_joint_pos(asset, joint_ids)
    delta = torch.abs(joint_pos - init_joint_pos)

    if mode == "displacement":
        value = delta
    elif mode == "progress":
        lower = asset.data.joint_pos_limits[:, joint_ids, 0]
        upper = asset.data.joint_pos_limits[:, joint_ids, 1]
        reachable = torch.maximum(init_joint_pos - lower, upper - init_joint_pos)
        denom = torch.clamp(reachable, min=1e-6)
        value = delta / denom
    elif mode == "ratio":
        lower = asset.data.joint_pos_limits[:, joint_ids, 0]
        upper = asset.data.joint_pos_limits[:, joint_ids, 1]
        span = torch.clamp(upper - lower, min=1e-6)
        value = (joint_pos - lower) / span
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cmp = _compare(value, threshold, op, 0.0)
    if reduce == "any":
        return cmp.any(dim=1)
    if reduce == "all":
        return cmp.all(dim=1)
    raise ValueError(f"Unsupported reduce: {reduce}")


def object_lifted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_cfg: SceneEntityCfg | None = None,
    min_height: float = 0.05,
) -> torch.Tensor:
    """Terminate when an asset is lifted above its default height by min_height."""
    # Backward-compatible alias: some task configs still pass `object_cfg`.
    if object_cfg is not None:
        asset_cfg = object_cfg
    asset = env.scene[asset_cfg.name]
    return root_height_delta(asset) >= min_height


def joint_moved_and_object_lifted(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_height: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    mode: str = "progress",
    op: str = ">=",
    reduce: str = "any",
) -> torch.Tensor:
    """Success when a joint has moved past ``threshold`` AND the asset is lifted.

    Combines :func:`joint_relative_move` (the manipulation goal -- e.g. sliding a
    blade out) with :func:`object_lifted` (the asset root must be at least
    ``min_height`` above its spawn height). Requiring both prevents the policy
    from "succeeding" while the object is still pinned to the table / its
    supports; it must hold the object off the surface and drive the joint.

    ``asset_cfg`` is used for both checks: its ``joint_names`` select the joint
    for the move test, while the lift test only reads the asset root pose (joint
    ids are ignored there).
    """
    moved = joint_relative_move(env, threshold=threshold, asset_cfg=asset_cfg, mode=mode, op=op, reduce=reduce)
    lifted = object_lifted(env, asset_cfg=asset_cfg, min_height=min_height)
    return moved & lifted


class joint_co_completion(ManagerTermBase):
    """Success when all selected joints reach the target within a shared time window.

    Each step, the term records — per env and per selected joint — the first
    episode step at which the joint entered the satisfied region. If a joint
    drops back out, its first-entry step is cleared, so the window only counts
    contiguous dwell.

    ``mode="value"`` uses the same ``op``/``ref``/``tol`` semantics as
    :func:`joint_reach_threshold`. ``mode="ratio"`` compares the absolute joint
    range fraction ``(q - lower) / (upper - lower)`` against ``threshold``.

    Success fires when every selected joint is currently satisfied AND the
    spread between first-entry steps (``max - min``) is at most
    ``window_steps``. Releasing one joint long before the others will not
    trigger success.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._first_entry: torch.Tensor | None = None
        self._last_step: torch.Tensor | None = None

    def _ensure_buffer(self, shape: tuple[int, int], device) -> None:
        if self._first_entry is None or tuple(self._first_entry.shape) != shape or self._first_entry.device != device:
            self._first_entry = torch.full(shape, -1, dtype=torch.long, device=device)
            self._last_step = torch.zeros(shape[0], dtype=torch.long, device=device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("articulation"),
        op: str = ">=",
        ref: str = "value",
        tol: float = 0.0,
        window_steps: int = 30,
        mode: str = "value",
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = resolve_joint_ids(env, asset_cfg)
        joint_pos = asset.data.joint_pos[:, joint_ids]

        if mode == "value":
            if ref == "value":
                target = threshold
            elif ref == "upper_limit":
                target = asset.data.joint_pos_limits[:, joint_ids, 1] + threshold
            elif ref == "lower_limit":
                target = asset.data.joint_pos_limits[:, joint_ids, 0] + threshold
            else:
                raise ValueError(f"Unsupported ref: {ref}")
            value = joint_pos
        elif mode == "ratio":
            lower = asset.data.joint_pos_limits[:, joint_ids, 0]
            upper = asset.data.joint_pos_limits[:, joint_ids, 1]
            span = torch.clamp(upper - lower, min=1e-6)
            value = (joint_pos - lower) / span
            target = threshold
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        satisfied = _compare(value, target, op, tol)

        self._ensure_buffer((env.num_envs, joint_pos.shape[1]), env.device)

        # Detect per-env resets: episode_length_buf is monotone within an
        # episode, so a drop means the env was reset since the last call.
        current = env.episode_length_buf.to(dtype=self._first_entry.dtype)
        reset_rows = current < self._last_step
        if reset_rows.any():
            self._first_entry[reset_rows] = -1
        self._last_step = current.clone()

        step = current
        step = step.unsqueeze(1).expand_as(self._first_entry)
        newly_entered = satisfied & (self._first_entry < 0)
        self._first_entry = torch.where(newly_entered, step, self._first_entry)
        self._first_entry = torch.where(satisfied, self._first_entry, torch.full_like(self._first_entry, -1))

        all_satisfied = satisfied.all(dim=1)
        entries = self._first_entry.clamp_min(0)
        spread = entries.amax(dim=1) - entries.amin(dim=1)
        return all_satisfied & (spread <= window_steps)


def _build_zone_tensors(
    sphere_zones: list | None,
    box_zones: list | None,
    cylinder_zones: list | None,
    device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert flat zone lists to cached tensor parameters."""
    sphere_zones = sphere_zones or []
    box_zones = box_zones or []
    cylinder_zones = cylinder_zones or []
    if sphere_zones:
        arr = torch.tensor(sphere_zones, device=device, dtype=torch.float32)
        sphere_centers = arr[:, :3].contiguous()
        sphere_r2 = (arr[:, 3] ** 2).contiguous()
    else:
        sphere_centers = torch.zeros(0, 3, device=device)
        sphere_r2 = torch.zeros(0, device=device)
    if box_zones:
        arr = torch.tensor(box_zones, device=device, dtype=torch.float32)
        box_centers = arr[:, :3].contiguous()
        box_half = arr[:, 3:6].contiguous()
    else:
        box_centers = torch.zeros(0, 3, device=device)
        box_half = torch.zeros(0, 3, device=device)
    if cylinder_zones:
        normalized_cylinder_zones = []
        for zone in cylinder_zones:
            if len(zone) == 5:
                normalized_cylinder_zones.append([*zone, 1.0, 0.0, 0.0, 0.0])
            elif len(zone) == 9:
                normalized_cylinder_zones.append(zone)
            else:
                raise ValueError(
                    "Cylinder zones must be (cx, cy, cz, radius, half_height) "
                    "or (cx, cy, cz, radius, half_height, qw, qx, qy, qz)."
                )
        arr = torch.tensor(normalized_cylinder_zones, device=device, dtype=torch.float32)
        cylinder_centers = arr[:, :3].contiguous()
        cylinder_radius2_height = torch.stack((arr[:, 3] ** 2, arr[:, 4]), dim=-1).contiguous()
        cylinder_quats = arr[:, 5:9]
        quat_norm = torch.linalg.norm(cylinder_quats, dim=-1, keepdim=True)
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32)
        cylinder_quats = torch.where(
            quat_norm > 1.0e-8,
            cylinder_quats / quat_norm.clamp_min(1.0e-8),
            identity.expand_as(cylinder_quats),
        ).contiguous()
    else:
        cylinder_centers = torch.zeros(0, 3, device=device)
        cylinder_radius2_height = torch.zeros(0, 2, device=device)
        cylinder_quats = torch.zeros(0, 4, device=device)
    return (
        sphere_centers,
        sphere_r2,
        box_centers,
        box_half,
        cylinder_centers,
        cylinder_radius2_height,
        cylinder_quats,
    )


def _eval_zone_violation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    sphere_centers: torch.Tensor,
    sphere_r2: torch.Tensor,
    box_centers: torch.Tensor,
    box_half: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radius2_height: torch.Tensor,
    cylinder_quats: torch.Tensor,
) -> torch.Tensor:
    """Per-env bool: any body listed in ``asset_cfg`` is inside any forbidden zone."""
    n_spheres = sphere_centers.shape[0]
    n_boxes = box_centers.shape[0]
    n_cylinders = cylinder_centers.shape[0]
    if n_spheres == 0 and n_boxes == 0 and n_cylinders == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    body_pos_w = _asset_body_positions_w(env, asset_cfg)
    body_pos_obj = _positions_in_object_frame(env, body_pos_w, object_cfg)

    violated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if n_spheres > 0:
        d2 = ((body_pos_obj.unsqueeze(2) - sphere_centers) ** 2).sum(dim=-1)
        violated |= (d2 < sphere_r2).any(dim=-1).any(dim=-1)
    if n_boxes > 0:
        d = (body_pos_obj.unsqueeze(2) - box_centers).abs()
        inside = (d < box_half).all(dim=-1)
        violated |= inside.any(dim=-1).any(dim=-1)
    if n_cylinders > 0:
        d = body_pos_obj.unsqueeze(2) - cylinder_centers
        cylinder_quats = cylinder_quats.unsqueeze(0).unsqueeze(0).expand(*d.shape[:2], -1, -1)
        d = quat_apply_inverse(cylinder_quats.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)
        radial2 = (d[..., :2] ** 2).sum(dim=-1)
        inside_radial = radial2 < cylinder_radius2_height[:, 0]
        inside_height = d[..., 2].abs() < cylinder_radius2_height[:, 1]
        violated |= (inside_radial & inside_height).any(dim=-1).any(dim=-1)
    return violated


def _asset_body_positions_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return world positions for configured articulation bodies or a rigid root."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset, Articulation):
        body_ids = asset_cfg.body_ids
        if body_ids is None:
            return asset.data.body_pos_w
        return asset.data.body_pos_w[:, body_ids]
    if isinstance(asset, RigidObject):
        return asset.data.root_pos_w.unsqueeze(1)
    raise TypeError(f"Unsupported asset type for contact-zone evaluation: {type(asset)!r}")


def _positions_in_object_frame(
    env: ManagerBasedRLEnv,
    positions_w: torch.Tensor,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Transform ``(E, B, 3)`` world positions into the object's root frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    E, B, _ = positions_w.shape

    rel_w = positions_w - obj.data.root_pos_w[:, None, :]
    obj_quat = obj.data.root_quat_w[:, None, :].expand(-1, B, -1).reshape(-1, 4)
    return quat_apply_inverse(obj_quat, rel_w.reshape(-1, 3)).reshape(E, B, 3)


def _eval_zone_coverage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    sphere_centers: torch.Tensor,
    sphere_r2: torch.Tensor,
    box_centers: torch.Tensor,
    box_half: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radius2_height: torch.Tensor,
    cylinder_quats: torch.Tensor,
) -> torch.Tensor:
    """Per-env bool: every required zone contains at least one configured body."""
    n_spheres = sphere_centers.shape[0]
    n_boxes = box_centers.shape[0]
    n_cylinders = cylinder_centers.shape[0]
    if n_spheres == 0 and n_boxes == 0 and n_cylinders == 0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    body_pos_w = _asset_body_positions_w(env, asset_cfg)
    body_pos_obj = _positions_in_object_frame(env, body_pos_w, object_cfg)

    satisfied = []
    if n_spheres > 0:
        d2 = ((body_pos_obj.unsqueeze(2) - sphere_centers) ** 2).sum(dim=-1)
        satisfied.append((d2 < sphere_r2).any(dim=1))
    if n_boxes > 0:
        d = (body_pos_obj.unsqueeze(2) - box_centers).abs()
        satisfied.append((d < box_half).all(dim=-1).any(dim=1))
    if n_cylinders > 0:
        d = body_pos_obj.unsqueeze(2) - cylinder_centers
        cylinder_quats = cylinder_quats.unsqueeze(0).unsqueeze(0).expand(*d.shape[:2], -1, -1)
        d = quat_apply_inverse(cylinder_quats.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)
        radial2 = (d[..., :2] ** 2).sum(dim=-1)
        inside_radial = radial2 < cylinder_radius2_height[:, 0]
        inside_height = d[..., 2].abs() < cylinder_radius2_height[:, 1]
        satisfied.append((inside_radial & inside_height).any(dim=1))
    return torch.cat(satisfied, dim=1).all(dim=1)


class success_no_forbidden_contact(ManagerTermBase):
    """Goal-position success gated by forbidden-zone clearance.

    Fires only when the object is within ``threshold`` of the commanded goal
    AND no body listed in ``asset_cfg.body_ids`` (e.g. fingertips) lies inside
    any forbidden zone defined in the object's local frame.

    ``sphere_zones``: list of ``(cx, cy, cz, radius)`` in object-local frame.
    ``box_zones``: list of ``(cx, cy, cz, hx, hy, hz)`` (axis-aligned half-sizes
    in object-local frame).
    ``cylinder_zones``: list of ``(cx, cy, cz, radius, half_height)`` or
    ``(cx, cy, cz, radius, half_height, qw, qx, qy, qz)`` in object-local frame.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        (
            self._sphere_centers,
            self._sphere_r2,
            self._box_centers,
            self._box_half,
            self._cylinder_centers,
            self._cylinder_radius2_height,
            self._cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("sphere_zones"),
            cfg.params.get("box_zones"),
            cfg.params.get("cylinder_zones"),
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float = 0.03,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        at_goal = command_term.metrics["position_error"] < threshold

        if (
            self._sphere_centers.shape[0] == 0
            and self._box_centers.shape[0] == 0
            and self._cylinder_centers.shape[0] == 0
        ):
            return at_goal

        violated = _eval_zone_violation(
            env,
            asset_cfg,
            object_cfg,
            self._sphere_centers,
            self._sphere_r2,
            self._box_centers,
            self._box_half,
            self._cylinder_centers,
            self._cylinder_radius2_height,
            self._cylinder_quats,
        )
        return at_goal & ~violated


class success_with_contact_zones(ManagerTermBase):
    """Goal-position success gated by forbidden and designated contact zones.

    In addition to the forbidden-zone clearance supported by
    :class:`success_no_forbidden_contact`, this term requires every designated
    contact zone to contain at least one body from ``contact_asset_cfg``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        (
            self._forbidden_sphere_centers,
            self._forbidden_sphere_r2,
            self._forbidden_box_centers,
            self._forbidden_box_half,
            self._forbidden_cylinder_centers,
            self._forbidden_cylinder_radius2_height,
            self._forbidden_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("sphere_zones"),
            cfg.params.get("box_zones"),
            cfg.params.get("cylinder_zones"),
            env.device,
        )
        (
            self._contact_sphere_centers,
            self._contact_sphere_r2,
            self._contact_box_centers,
            self._contact_box_half,
            self._contact_cylinder_centers,
            self._contact_cylinder_radius2_height,
            self._contact_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("contact_sphere_zones"),
            cfg.params.get("contact_box_zones"),
            cfg.params.get("contact_cylinder_zones"),
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float = 0.03,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        contact_sphere_zones: list | None = None,
        contact_box_zones: list | None = None,
        contact_cylinder_zones: list | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_asset_cfg: SceneEntityCfg | None = None,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        contact_object_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        success = command_term.metrics["position_error"] < threshold

        if (
            self._forbidden_sphere_centers.shape[0] > 0
            or self._forbidden_box_centers.shape[0] > 0
            or self._forbidden_cylinder_centers.shape[0] > 0
        ):
            violated = _eval_zone_violation(
                env,
                asset_cfg,
                object_cfg,
                self._forbidden_sphere_centers,
                self._forbidden_sphere_r2,
                self._forbidden_box_centers,
                self._forbidden_box_half,
                self._forbidden_cylinder_centers,
                self._forbidden_cylinder_radius2_height,
                self._forbidden_cylinder_quats,
            )
            success &= ~violated

        if (
            self._contact_sphere_centers.shape[0] > 0
            or self._contact_box_centers.shape[0] > 0
            or self._contact_cylinder_centers.shape[0] > 0
        ):
            covered = _eval_zone_coverage(
                env,
                contact_asset_cfg or asset_cfg,
                contact_object_cfg or object_cfg,
                self._contact_sphere_centers,
                self._contact_sphere_r2,
                self._contact_box_centers,
                self._contact_box_half,
                self._contact_cylinder_centers,
                self._contact_cylinder_radius2_height,
                self._contact_cylinder_quats,
            )
            success &= covered

        return success


class lift_and_tilt_with_contact_zones(ManagerTermBase):
    """Lift-and-tilt success gated by forbidden and designated contact zones."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        (
            self._forbidden_sphere_centers,
            self._forbidden_sphere_r2,
            self._forbidden_box_centers,
            self._forbidden_box_half,
            self._forbidden_cylinder_centers,
            self._forbidden_cylinder_radius2_height,
            self._forbidden_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("sphere_zones"),
            cfg.params.get("box_zones"),
            cfg.params.get("cylinder_zones"),
            env.device,
        )
        (
            self._contact_sphere_centers,
            self._contact_sphere_r2,
            self._contact_box_centers,
            self._contact_box_half,
            self._contact_cylinder_centers,
            self._contact_cylinder_radius2_height,
            self._contact_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("contact_sphere_zones"),
            cfg.params.get("contact_box_zones"),
            cfg.params.get("contact_cylinder_zones"),
            env.device,
        )

        # Optional object-local offset for the xy-distance check. When set,
        # the xy distance is computed between
        # ``obj.root_pos + R(obj.root_quat) * offset`` and the goal asset's
        # root xy, which lets tasks anchor the gate at e.g. the bottle's
        # spout instead of the bottle's mass centre.
        offset = cfg.params.get("goal_object_local_offset")
        if offset is None:
            self._goal_object_local_offset = None
        else:
            self._goal_object_local_offset = torch.tensor(offset, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        threshold_rad: float,
        axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
        world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        tilt_ge: bool = True,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        contact_sphere_zones: list | None = None,
        contact_box_zones: list | None = None,
        contact_cylinder_zones: list | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_asset_cfg: SceneEntityCfg | None = None,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        contact_object_cfg: SceneEntityCfg | None = None,
        goal_asset_cfg: SceneEntityCfg | None = None,
        goal_xy_threshold: float | None = None,
        goal_object_local_offset: tuple[float, float, float] | None = None,
        plane_axis_local: tuple[float, float, float] | None = None,
        plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        plane_angle_threshold_rad: float | None = None,
    ) -> torch.Tensor:
        # Lift gate (always required).
        obj: RigidObject = env.scene[object_cfg.name]
        lifted = object_lifted(env, asset_cfg=object_cfg, min_height=min_height)

        # Primary tilt gate: angle between a local axis and a world axis.
        primary_angle = axis_tilt_angle(obj.data.root_quat_w, axis_local=axis_local, world_axis=world_axis)
        primary_tilt_ok = primary_angle >= threshold_rad if tilt_ge else primary_angle <= threshold_rad

        # Optional secondary tilt gate: angle between a (possibly different)
        # local axis and the ground plane (normal = ``plane_normal``). When
        # supplied, satisfying *either* tilt criterion counts as enough tilt.
        if plane_axis_local is not None and plane_angle_threshold_rad is not None:
            plane_angle = axis_to_plane_angle(
                obj.data.root_quat_w,
                axis_local=plane_axis_local,
                plane_normal=plane_normal,
            )
            plane_tilt_ok = plane_angle >= float(plane_angle_threshold_rad)
            tilt_ok = primary_tilt_ok | plane_tilt_ok
        else:
            tilt_ok = primary_tilt_ok

        success = lifted & tilt_ok
        if goal_asset_cfg is not None and goal_xy_threshold is not None:
            goal: RigidObject = env.scene[goal_asset_cfg.name]
            if self._goal_object_local_offset is not None:
                offset_e = self._goal_object_local_offset.unsqueeze(0).expand(env.num_envs, -1)
                offset_w = quat_apply(obj.data.root_quat_w, offset_e)
                zone_w = obj.data.root_pos_w + offset_w
            else:
                zone_w = obj.data.root_pos_w
            dxy = zone_w[:, :2] - goal.data.root_pos_w[:, :2]
            within_goal = dxy.pow(2).sum(dim=-1) <= float(goal_xy_threshold) ** 2
            success &= within_goal

        if (
            self._forbidden_sphere_centers.shape[0] > 0
            or self._forbidden_box_centers.shape[0] > 0
            or self._forbidden_cylinder_centers.shape[0] > 0
        ):
            violated = _eval_zone_violation(
                env,
                asset_cfg,
                object_cfg,
                self._forbidden_sphere_centers,
                self._forbidden_sphere_r2,
                self._forbidden_box_centers,
                self._forbidden_box_half,
                self._forbidden_cylinder_centers,
                self._forbidden_cylinder_radius2_height,
                self._forbidden_cylinder_quats,
            )
            success &= ~violated

        if (
            self._contact_sphere_centers.shape[0] > 0
            or self._contact_box_centers.shape[0] > 0
            or self._contact_cylinder_centers.shape[0] > 0
        ):
            covered = _eval_zone_coverage(
                env,
                contact_asset_cfg or asset_cfg,
                contact_object_cfg or object_cfg,
                self._contact_sphere_centers,
                self._contact_sphere_r2,
                self._contact_box_centers,
                self._contact_box_half,
                self._contact_cylinder_centers,
                self._contact_cylinder_radius2_height,
                self._contact_cylinder_quats,
            )
            success &= covered

        return success


class lifted_no_forbidden_contact(ManagerTermBase):
    """Height-based lift success gated by forbidden-zone clearance.

    Fires when the object's world-frame z-coordinate exceeds ``min_height``
    AND no body listed in ``asset_cfg.body_ids`` lies inside any forbidden
    zone (object-local frame).

    Mirrors :func:`object_lifted` for the success half so existing
    bimanual-lift configs only need to swap the termination ``func``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        (
            self._sphere_centers,
            self._sphere_r2,
            self._box_centers,
            self._box_half,
            self._cylinder_centers,
            self._cylinder_radius2_height,
            self._cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("sphere_zones"),
            cfg.params.get("box_zones"),
            cfg.params.get("cylinder_zones"),
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        lifted = root_height_delta(obj) >= min_height

        if (
            self._sphere_centers.shape[0] == 0
            and self._box_centers.shape[0] == 0
            and self._cylinder_centers.shape[0] == 0
        ):
            return lifted

        violated = _eval_zone_violation(
            env,
            asset_cfg,
            object_cfg,
            self._sphere_centers,
            self._sphere_r2,
            self._box_centers,
            self._box_half,
            self._cylinder_centers,
            self._cylinder_radius2_height,
            self._cylinder_quats,
        )
        return lifted & ~violated


def nutthread_success(
    env: ManagerBasedRLEnv,
    success_threshold_turns: float = 0.375,
    center_dist_thresh: float = 0.0025,
    thread_pitch: float = 0.002,
    nut_cfg: SceneEntityCfg = SceneEntityCfg("nut"),
    bolt_cfg: SceneEntityCfg = SceneEntityCfg("bolt"),
    nut_base_height: float = 0.01,
    bolt_head_height: float = 0.01,
    bolt_shank_height: float = 0.025,
    target_thread_turns: float = 1.5,
) -> torch.Tensor:
    """Factory-style NutThread success: centered in XY and deep enough in Z."""
    nut: Articulation = env.scene[nut_cfg.name]
    bolt: Articulation = env.scene[bolt_cfg.name]

    nut_pos = nut.data.root_pos_w
    nut_quat = nut.data.root_quat_w
    bolt_pos = bolt.data.root_pos_w
    bolt_quat = bolt.data.root_quat_w

    nut_base_local = torch.zeros((env.num_envs, 3), device=env.device, dtype=nut_pos.dtype)
    nut_base_local[:, 2] = float(nut_base_height)
    nut_base_pos = nut_pos + quat_apply(nut_quat, nut_base_local)

    target_local = torch.zeros((env.num_envs, 3), device=env.device, dtype=bolt_pos.dtype)
    target_local[:, 2] = float(bolt_head_height + bolt_shank_height - thread_pitch * target_thread_turns)
    target_pos = bolt_pos + quat_apply(bolt_quat, target_local)

    xy_dist = torch.linalg.vector_norm(target_pos[:, 0:2] - nut_base_pos[:, 0:2], dim=1)
    z_disp = nut_base_pos[:, 2] - target_pos[:, 2]
    centered = xy_dist < float(center_dist_thresh)
    close_or_below = z_disp < float(thread_pitch * success_threshold_turns)
    success = torch.logical_and(centered, close_or_below)
    return success


def factory_insert_success(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    held_base_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_dist_thresh: float = 0.0025,
    z_threshold: float = 0.001,
    held_base_local_offset_2: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Terminate when held asset is centered and inserted below z-threshold.

    Pass ``held_base_local_offset_2`` to terminate on *either* held reference
    point (e.g. either end of a pen) being inserted.
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
    )


def plug_charger_pose_success(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    insertion_x_threshold: float = 0.02,
    insertion_y_threshold: float = 0.01,
    insertion_z_threshold: float = 0.01,
    held_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """PlugCharger success using insertion geometry criteria in receptacle frame."""
    return plug_charger_insertion_mask(
        env=env,
        held_cfg=held_cfg,
        fixed_cfg=fixed_cfg,
        held_local_offset=held_local_offset,
        fixed_local_offset=fixed_local_offset,
        insertion_x_threshold=insertion_x_threshold,
        lateral_y_threshold=insertion_y_threshold,
        vertical_z_threshold=insertion_z_threshold,
    )


def insert_peg_success(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    peg_head_local_offset: tuple[float, float, float] = (0.10, 0.0, 0.0),
    hole_center_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hole_radius: float = 0.023,
    insertion_x_threshold: float = 0.015,
) -> torch.Tensor:
    """Terminate on ManiSkill PegInsertionSide success condition."""
    return insert_peg_success_mask(
        env=env,
        peg_cfg=peg_cfg,
        hole_cfg=hole_cfg,
        peg_head_local_offset=peg_head_local_offset,
        hole_center_local_offset=hole_center_local_offset,
        hole_radius=hole_radius,
        insertion_x_threshold=insertion_x_threshold,
    )


def push_t_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_tee"),
    success_threshold: float = 0.90,
    overlap_point_step: float = 0.005,
) -> torch.Tensor:
    """Terminate when the movable T overlaps the goal T above threshold."""
    overlap = push_t_overlap_ratio(
        env=env,
        object_cfg=object_cfg,
        goal_cfg=goal_cfg,
        point_step=overlap_point_step,
    )
    return overlap >= float(success_threshold)


def assets_axis_aligned(
    env: ManagerBasedRLEnv,
    source_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    source_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    target_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    threshold_cos: float = 0.95,
    use_abs: bool = True,
) -> torch.Tensor:
    """Per-env bool: a chosen local axis of two assets aligns within a cosine threshold.

    Both axes are rotated into world frame using each asset's root quaternion,
    then compared via signed cosine similarity. ``use_abs=True`` treats the axes
    as undirected so anti-parallel orientations also count as aligned.
    """
    src_axis_w = asset_axis_w(env, asset_cfg=source_asset_cfg, axis_local=source_axis_local)
    tgt_axis_w = asset_axis_w(env, asset_cfg=target_asset_cfg, axis_local=target_axis_local)
    cos = axis_alignment_dot(src_axis_w, tgt_axis_w)
    if use_abs:
        cos = torch.abs(cos)
    return cos >= threshold_cos


class hammer_strike_success(ManagerTermBase):
    """Success when grasp is held, forbidden zones are clear, and target compresses.

    Combines, per env:
      - ``forbidden`` zone clearance for the hand on the source object,
      - optional ``target_*_zones`` clearance for the hand on the target asset
        (e.g. to forbid pressing the nail directly with a fingertip),
      - target-articulation joint displacement past ``press_threshold_m``.

    ``*_zones`` are interpreted in the source object's local frame;
    ``target_*_zones`` are interpreted in the target asset's local frame.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        (
            self._forbidden_sphere_centers,
            self._forbidden_sphere_r2,
            self._forbidden_box_centers,
            self._forbidden_box_half,
            self._forbidden_cylinder_centers,
            self._forbidden_cylinder_radius2_height,
            self._forbidden_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("sphere_zones"),
            cfg.params.get("box_zones"),
            cfg.params.get("cylinder_zones"),
            env.device,
        )
        (
            self._target_forbidden_sphere_centers,
            self._target_forbidden_sphere_r2,
            self._target_forbidden_box_centers,
            self._target_forbidden_box_half,
            self._target_forbidden_cylinder_centers,
            self._target_forbidden_cylinder_radius2_height,
            self._target_forbidden_cylinder_quats,
        ) = _build_zone_tensors(
            cfg.params.get("target_sphere_zones"),
            cfg.params.get("target_box_zones"),
            cfg.params.get("target_cylinder_zones"),
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        press_threshold_m: float,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        target_sphere_zones: list | None = None,
        target_box_zones: list | None = None,
        target_cylinder_zones: list | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        target_asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        target_joint_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        press_mode: str = "displacement",
    ) -> torch.Tensor:
        success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if (
            self._forbidden_sphere_centers.shape[0] > 0
            or self._forbidden_box_centers.shape[0] > 0
            or self._forbidden_cylinder_centers.shape[0] > 0
        ):
            violated = _eval_zone_violation(
                env,
                asset_cfg,
                object_cfg,
                self._forbidden_sphere_centers,
                self._forbidden_sphere_r2,
                self._forbidden_box_centers,
                self._forbidden_box_half,
                self._forbidden_cylinder_centers,
                self._forbidden_cylinder_radius2_height,
                self._forbidden_cylinder_quats,
            )
            success &= ~violated
        if (
            self._target_forbidden_sphere_centers.shape[0] > 0
            or self._target_forbidden_box_centers.shape[0] > 0
            or self._target_forbidden_cylinder_centers.shape[0] > 0
        ):
            target_violated = _eval_zone_violation(
                env,
                asset_cfg,
                target_asset_cfg,
                self._target_forbidden_sphere_centers,
                self._target_forbidden_sphere_r2,
                self._target_forbidden_box_centers,
                self._target_forbidden_box_half,
                self._target_forbidden_cylinder_centers,
                self._target_forbidden_cylinder_radius2_height,
                self._target_forbidden_cylinder_quats,
            )
            success &= ~target_violated
        pressed = joint_relative_move(
            env,
            threshold=float(press_threshold_m),
            asset_cfg=target_joint_cfg,
            mode=press_mode,
            op=">=",
            reduce="any",
        )
        return success & pressed
