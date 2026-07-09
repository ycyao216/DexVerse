# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dexverse.visual_purpose import hide_marker_from_cameras
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, transform_points
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from .utils import (
    asset_axis_w,
    axis_in_frame_from_quat,
    axis_tilt_angle,
    axis_to_plane_angle,
    command_axis_w,
    compute_body_pose_vel_b,
    compute_body_state_b,
    root_height_delta,
    sample_object_point_cloud,
)


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def object_height_delta(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root height risen above its spawn pose, in meters.

    This is the same quantity the height-based lift success uses
    (:func:`root_height_delta`): ``root_pos_w.z - default_root_state.z``. It is
    observable / deployable (height above the known reset, equivalently above the
    table), so it belongs in the ``state`` group, not ``privileged``. Named
    distinctly from the ``object_lift_height`` *reward* to avoid an ``mdp``
    namespace collision.

    Returns:
        Tensor of shape ``(num_envs, 1)``.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return root_height_delta(asset).unsqueeze(1)


def scalar_obs(
    env: ManagerBasedRLEnv,
    value: float | bool,
) -> torch.Tensor:
    """Constant scalar observation, repeated once per environment."""
    return torch.full((env.num_envs, 1), float(value), device=env.device)


def asset_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Asset root position in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, asset.data.root_pos_w - robot.data.root_pos_w)


def object_local_point_pos_b(
    env: ManagerBasedRLEnv,
    local_offset: tuple[float, float, float] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of an object-local functional point in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    point_w = object.data.root_pos_w
    if local_offset is not None:
        offset = torch.as_tensor(local_offset, device=env.device, dtype=point_w.dtype)
        offset = offset.unsqueeze(0).expand(env.num_envs, -1)
        point_w = point_w + quat_apply(object.data.root_quat_w, offset)
    return quat_apply_inverse(robot.data.root_quat_w, point_w - robot.data.root_pos_w)


def object_up_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object local +Z axis expressed in the robot's root frame."""
    return object_rot_axis_b(env, axis_local=(0.0, 0.0, 1.0), robot_cfg=robot_cfg, object_cfg=object_cfg)


def object_tilt_angle(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Angle (rad) between object's +Z axis and world +Z (0=up, pi=down)."""
    object: RigidObject = env.scene[object_cfg.name]
    return axis_tilt_angle(object.data.root_quat_w, axis_local=(0.0, 0.0, 1.0), world_axis=(0.0, 0.0, 1.0)).unsqueeze(1)


def object_lin_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_lin_vel_w)


def object_ang_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object angular velocity in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_ang_vel_w)


def object_rot_axis_b(
    env: ManagerBasedRLEnv,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object local axis direction in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    axis_w = asset_axis_w(env, asset_cfg=object_cfg, axis_local=axis_local)
    return axis_in_frame_from_quat(axis_w, robot.data.root_quat_w)


def target_rot_axis_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target local axis direction from pose command in the robot root frame."""
    axis_w = command_axis_w(env, command_name=command_name, axis_local=axis_local, robot_cfg=robot_cfg)
    robot: RigidObject = env.scene[robot_cfg.name]
    return axis_in_frame_from_quat(axis_w, robot.data.root_quat_w)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    out, _ = compute_body_state_b(env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg)
    return out


def body_pose_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body pose (pos, quat) in the base asset's root frame — the velocity-free
    half of :func:`body_state_b`, for the observable ``state`` group.

    Per-body layout ``[position(3), quaternion(4)(wxyz)]`` concatenated over bodies.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 7)``.
    """
    pose_b, _, _ = compute_body_pose_vel_b(env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg)
    return pose_b


def body_vel_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body velocity (lin, ang) in the base asset's root frame — the sim-only
    half of :func:`body_state_b`, for the ``privileged`` group.

    Per-body layout ``[linvel(3), angvel(3)]`` concatenated over bodies.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 6)``.
    """
    _, vel_b, _ = compute_body_pose_vel_b(env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg)
    return vel_b


class body_state_b_vis(ManagerTermBase):
    """Body state with optional viewport visualization of observed body links.

    The red sphere markers are off by default. The PointInstancer that backs
    ``VisualizationMarkers`` isn't reliably hidden from RTX cameras by the
    ``purpose=guide`` tag that ``hide_marker_from_cameras`` sets, so the
    markers were contaminating RGB observations / demos / training images.
    To re-enable the viewport markers, pass ``"draw_markers": True`` in
    ``ObsTerm.params``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.body_asset_cfg: SceneEntityCfg = cfg.params.get("body_asset_cfg", SceneEntityCfg("robot"))
        self.base_asset_cfg: SceneEntityCfg = cfg.params.get("base_asset_cfg", SceneEntityCfg("robot"))
        self.draw_markers: bool = bool(cfg.params.get("draw_markers", False))

        self.visualizer = None
        if self.draw_markers:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/BodyStateMarkers",
                markers={
                    "dot": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self.visualizer = VisualizationMarkers(marker_cfg)
            hide_marker_from_cameras(self.visualizer)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_asset_cfg: SceneEntityCfg = None,
        base_asset_cfg: SceneEntityCfg = None,
        draw_markers: bool = False,
    ) -> torch.Tensor:
        """Compute body state and (optionally) visualize body positions."""
        body_asset_cfg = body_asset_cfg or self.body_asset_cfg
        base_asset_cfg = base_asset_cfg or self.base_asset_cfg

        out, body_pos_w_flat = compute_body_state_b(env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg)
        if self.visualizer is not None:
            self.visualizer.visualize(translations=body_pos_w_flat)
        return out


class robot_fingertips_vis(ManagerTermBase):
    """Visualize robot fingertips with colored spheres matching human fingertips.

    Colors match the retargeter visualization:
    - thumb_fingertip: Red
    - fingertip (index): Green
    - fingertip_2 (middle): Blue
    - fingertip_3 (ring): Yellow
    - fingertip_4 (little/pinky): Magenta
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.body_asset_cfg: SceneEntityCfg = cfg.params.get("body_asset_cfg", SceneEntityCfg("robot"))

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        # Create markers with colors matching human fingertips
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/RobotFingertipMarkers",
            markers={
                "thumb_tip": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red for thumb
                ),
                "index_tip": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green for index
                ),
                "middle_tip": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # Blue for middle
                ),
                "ring_tip": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow for ring
                ),
                "little_tip": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 1.0)
                    ),  # Magenta for little/pinky
                ),
            },
        )
        self.visualizer = VisualizationMarkers(marker_cfg)
        hide_marker_from_cameras(self.visualizer)

        # Map robot fingertip body names to marker indices
        # Using tip_head links: thumb_tip_head=0 (Red), index_tip_head=1 (Green), middle_tip_head=2 (Blue), ring_tip_head=3 (Yellow), little_tip_head=4 (Magenta)
        # Also support standard naming conventions (thdistal, ffdistal, mfdistal, rfdistal, lfdistal for Shadow Hand)
        self.fingertip_marker_map = {
            "thumb_tip_head": 0,  # Red
            "index_tip_head": 1,  # Green (index)
            "middle_tip_head": 2,  # Blue (middle)
            "ring_tip_head": 3,  # Yellow (ring)
            "little_tip_head": 4,  # Magenta (little/pinky)
            # Shadow Hand standard naming (fallback)
            "thdistal": 0,  # Red (thumb)
            "ffdistal": 1,  # Green (index/first finger)
            "mfdistal": 2,  # Blue (middle finger)
            "rfdistal": 3,  # Yellow (ring finger)
            "lfdistal": 4,  # Magenta (little finger)
        }

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_asset_cfg: SceneEntityCfg = None,
    ) -> torch.Tensor:
        """Visualize robot fingertip positions with colored spheres."""
        body_asset_cfg = body_asset_cfg or self.body_asset_cfg

        body_asset: Articulation = env.scene[body_asset_cfg.name]
        fingertip_body_names = body_asset_cfg.body_names

        if fingertip_body_names is None or len(fingertip_body_names) == 0:
            return torch.zeros(env.num_envs, 0, device=env.device)

        # Debug: Print available body names and requested fingertip names (only once)
        if not hasattr(self, "_debug_printed"):
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Robot fingertip visualization - Available body names: {body_asset.body_names}")
            logger.info(f"Robot fingertip visualization - Requested fingertip names: {fingertip_body_names}")
            self._debug_printed = True

        # Get fingertip body indices
        fingertip_body_ids = []
        marker_indices_list = []
        for idx, body_name in enumerate(fingertip_body_names):
            if body_name in body_asset.body_names:
                body_idx = body_asset.body_names.index(body_name)
                fingertip_body_ids.append(body_idx)
                # Get marker index: use mapping if available, otherwise use order (thumb=0, index=1, middle=2, ring=3, little=4)
                marker_idx = self.fingertip_marker_map.get(body_name, min(idx, 4))
                marker_indices_list.append(marker_idx)

        if len(fingertip_body_ids) == 0:
            return torch.zeros(env.num_envs, 0, device=env.device)

        # Get fingertip positions in world frame
        fingertip_pos_w = body_asset.data.body_pos_w[:, fingertip_body_ids]  # Shape: (num_envs, num_fingertips, 3)

        # Flatten for visualization: (num_envs * num_fingertips, 3)
        fingertip_pos_w_flat = fingertip_pos_w.view(-1, 3)

        # Create marker indices tensor: (num_envs * num_fingertips,)
        marker_indices = torch.tensor(marker_indices_list * env.num_envs, dtype=torch.int32, device=env.device)

        # Visualize fingertip positions with colored markers
        self.visualizer.visualize(translations=fingertip_pos_w_flat, marker_indices=marker_indices)

        # Return empty tensor (this is just for visualization, not used as observation)
        return torch.zeros(env.num_envs, 0, device=env.device)


class object_world_frame_vis(ManagerTermBase):
    """Visualize object and world frames with axis markers."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        frame_scale = cfg.params.get("frame_scale", (0.08, 0.08, 0.08))

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.markers["frame"].scale = frame_scale
        self.object_frame_vis = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ObjectFrame"))
        hide_marker_from_cameras(self.object_frame_vis)
        self.world_frame_vis = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/WorldFrame"))
        hide_marker_from_cameras(self.world_frame_vis)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        object_cfg = object_cfg or self.object_cfg
        obj: RigidObject = env.scene[object_cfg.name]

        marker_indices = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.object_frame_vis.visualize(
            translations=obj.data.root_pos_w,
            orientations=obj.data.root_quat_w,
            marker_indices=marker_indices,
        )

        world_pos = env.scene.env_origins
        world_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=world_pos.dtype)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )
        self.world_frame_vis.visualize(
            translations=world_pos,
            orientations=world_quat,
            marker_indices=marker_indices,
        )

        return torch.zeros(env.num_envs, 0, device=env.device)


class insertion_reference_points_vis(ManagerTermBase):
    """Visualize held insertion point and fixed target point as debug markers."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._held_cfg: SceneEntityCfg = cfg.params.get("held_cfg", SceneEntityCfg("object"))
        self._fixed_cfg: SceneEntityCfg = cfg.params.get("fixed_cfg", SceneEntityCfg("target"))
        self._held_local_offset = cfg.params.get("held_local_offset", (0.0, 0.0, 0.0))
        self._target_local_offset = cfg.params.get("target_local_offset", (0.0, 0.0, 0.0))
        self._radius = float(cfg.params.get("radius", 0.01))
        self._prim_path = cfg.params.get("prim_path", "/Visuals/InsertionReferencePoints")
        self._held_color = cfg.params.get("held_color", (0.95, 0.15, 0.15))
        self._target_color = cfg.params.get("target_color", (0.10, 0.85, 0.20))
        self._show_frames = bool(cfg.params.get("show_frames", False))
        self._frame_scale = cfg.params.get("frame_scale", (0.04, 0.04, 0.04))
        self._held_frame_prim_path = cfg.params.get("held_frame_prim_path", f"{self._prim_path}/HeldFrame")
        self._target_frame_prim_path = cfg.params.get("target_frame_prim_path", f"{self._prim_path}/TargetFrame")

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        marker_cfg = VisualizationMarkersCfg(
            prim_path=self._prim_path,
            markers={
                "held": sim_utils.SphereCfg(
                    radius=self._radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self._held_color),
                ),
                "target": sim_utils.SphereCfg(
                    radius=self._radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self._target_color),
                ),
            },
        )
        self._visualizer = VisualizationMarkers(marker_cfg)
        hide_marker_from_cameras(self._visualizer)

        self._held_frame_visualizer = None
        self._target_frame_visualizer = None
        if self._show_frames:
            from isaaclab.markers.config import FRAME_MARKER_CFG

            frame_cfg = FRAME_MARKER_CFG.copy()
            frame_cfg.markers["frame"].scale = self._frame_scale
            self._held_frame_visualizer = VisualizationMarkers(frame_cfg.replace(prim_path=self._held_frame_prim_path))
            hide_marker_from_cameras(self._held_frame_visualizer)
            self._target_frame_visualizer = VisualizationMarkers(
                frame_cfg.replace(prim_path=self._target_frame_prim_path)
            )
            hide_marker_from_cameras(self._target_frame_visualizer)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        held_cfg: SceneEntityCfg | None = None,
        fixed_cfg: SceneEntityCfg | None = None,
        held_local_offset: tuple[float, float, float] | None = None,
        target_local_offset: tuple[float, float, float] | None = None,
        radius: float | None = None,
        prim_path: str | None = None,
        held_color: tuple[float, float, float] | None = None,
        target_color: tuple[float, float, float] | None = None,
        show_frames: bool | None = None,
        frame_scale: tuple[float, float, float] | None = None,
        held_frame_prim_path: str | None = None,
        target_frame_prim_path: str | None = None,
    ) -> torch.Tensor:
        _ = (
            radius,
            prim_path,
            held_color,
            target_color,
            show_frames,
            frame_scale,
            held_frame_prim_path,
            target_frame_prim_path,
        )
        held_cfg = held_cfg or self._held_cfg
        fixed_cfg = fixed_cfg or self._fixed_cfg
        held_local_offset = held_local_offset or self._held_local_offset
        target_local_offset = target_local_offset or self._target_local_offset

        held: Articulation | RigidObject = env.scene[held_cfg.name]
        fixed: Articulation | RigidObject = env.scene[fixed_cfg.name]

        held_offset = torch.tensor(held_local_offset, device=env.device, dtype=held.data.root_pos_w.dtype)
        held_offset = held_offset.unsqueeze(0).repeat(env.num_envs, 1)
        target_offset = torch.tensor(target_local_offset, device=env.device, dtype=fixed.data.root_pos_w.dtype)
        target_offset = target_offset.unsqueeze(0).repeat(env.num_envs, 1)

        held_point_w = held.data.root_pos_w + quat_apply(held.data.root_quat_w, held_offset)
        target_point_w = fixed.data.root_pos_w + quat_apply(fixed.data.root_quat_w, target_offset)

        translations = torch.cat([held_point_w, target_point_w], dim=0)
        marker_indices = torch.cat(
            [
                torch.zeros(env.num_envs, dtype=torch.int32, device=env.device),
                torch.ones(env.num_envs, dtype=torch.int32, device=env.device),
            ],
            dim=0,
        )
        self._visualizer.visualize(translations=translations, marker_indices=marker_indices)
        if self._held_frame_visualizer is not None and self._target_frame_visualizer is not None:
            frame_marker_indices = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
            self._held_frame_visualizer.visualize(
                translations=held_point_w,
                orientations=held.data.root_quat_w,
                marker_indices=frame_marker_indices,
            )
            self._target_frame_visualizer.visualize(
                translations=target_point_w,
                orientations=fixed.data.root_quat_w,
                marker_indices=frame_marker_indices,
            )
        return torch.zeros(env.num_envs, 0, device=env.device)


class forbidden_zones_vis(ManagerTermBase):
    """Render forbidden zones (red, semi-transparent) attached to the object frame.

    Each zone is defined in the object's local frame; markers are repositioned
    every step so they track the object as it moves. ``sphere_zones``,
    ``box_zones``, and ``cylinder_zones`` use the same flat encoding as
    :class:`success_no_forbidden_contact`.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sphere_zones = cfg.params.get("sphere_zones") or []
        box_zones = cfg.params.get("box_zones") or []
        cylinder_zones = cfg.params.get("cylinder_zones") or []
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        color = cfg.params.get("color", (0.9, 0.1, 0.1))
        opacity = cfg.params.get("opacity", 0.4)
        prim_path_prefix = cfg.params.get("prim_path_prefix", "/Visuals/ForbiddenZone")

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        device = env.device
        self._visualizers = []
        centers = []
        quats = []
        identity_quat = (1.0, 0.0, 0.0, 0.0)

        for i, zone in enumerate(sphere_zones):
            cx, cy, cz, r = zone
            mcfg = VisualizationMarkersCfg(
                prim_path=f"{prim_path_prefix}/Sphere_{i}",
                markers={
                    "z": sim_utils.SphereCfg(
                        radius=float(r),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), opacity=float(opacity)),
                    )
                },
            )
            vis = VisualizationMarkers(mcfg)
            hide_marker_from_cameras(vis)
            self._visualizers.append(vis)
            centers.append([float(cx), float(cy), float(cz)])
            quats.append(identity_quat)

        for i, zone in enumerate(box_zones):
            cx, cy, cz, hx, hy, hz = zone
            mcfg = VisualizationMarkersCfg(
                prim_path=f"{prim_path_prefix}/Box_{i}",
                markers={
                    "z": sim_utils.CuboidCfg(
                        size=(2.0 * float(hx), 2.0 * float(hy), 2.0 * float(hz)),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), opacity=float(opacity)),
                    )
                },
            )
            vis = VisualizationMarkers(mcfg)
            hide_marker_from_cameras(vis)
            self._visualizers.append(vis)
            centers.append([float(cx), float(cy), float(cz)])
            quats.append(identity_quat)

        for i, zone in enumerate(cylinder_zones):
            if len(zone) == 5:
                cx, cy, cz, r, half_height = zone
                quat = identity_quat
            elif len(zone) == 9:
                cx, cy, cz, r, half_height, qw, qx, qy, qz = zone
                quat = (float(qw), float(qx), float(qy), float(qz))
            else:
                raise ValueError(
                    "Cylinder zones must be (cx, cy, cz, radius, half_height) "
                    "or (cx, cy, cz, radius, half_height, qw, qx, qy, qz)."
                )
            mcfg = VisualizationMarkersCfg(
                prim_path=f"{prim_path_prefix}/Cylinder_{i}",
                markers={
                    "z": sim_utils.CylinderCfg(
                        radius=float(r),
                        height=2.0 * float(half_height),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), opacity=float(opacity)),
                    )
                },
            )
            vis = VisualizationMarkers(mcfg)
            hide_marker_from_cameras(vis)
            self._visualizers.append(vis)
            centers.append([float(cx), float(cy), float(cz)])
            quats.append(quat)

        if centers:
            self._centers_obj = torch.tensor(centers, device=device, dtype=torch.float32)
            self._quats_obj = torch.tensor(quats, device=device, dtype=torch.float32)
            quat_norm = torch.linalg.norm(self._quats_obj, dim=-1, keepdim=True)
            identity = torch.tensor(identity_quat, device=device, dtype=torch.float32)
            self._quats_obj = torch.where(
                quat_norm > 1.0e-8,
                self._quats_obj / quat_norm.clamp_min(1.0e-8),
                identity.expand_as(self._quats_obj),
            )
        else:
            self._centers_obj = torch.zeros(0, 3, device=device)
            self._quats_obj = torch.zeros(0, 4, device=device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sphere_zones: list | None = None,
        box_zones: list | None = None,
        cylinder_zones: list | None = None,
        object_cfg: SceneEntityCfg | None = None,
        color: tuple[float, float, float] | None = None,
        opacity: float | None = None,
        prim_path_prefix: str | None = None,
    ) -> torch.Tensor:
        if not self._visualizers:
            return torch.zeros(env.num_envs, 0, device=env.device)

        object_cfg = object_cfg or self.object_cfg
        obj: RigidObject = env.scene[object_cfg.name]
        obj_pos_w = obj.data.root_pos_w  # (E, 3)
        obj_quat_w = obj.data.root_quat_w  # (E, 4)

        Z = self._centers_obj.shape[0]
        E = env.num_envs
        centers_e = self._centers_obj.unsqueeze(0).expand(E, -1, -1)  # (E, Z, 3)
        quat_e = obj_quat_w.unsqueeze(1).expand(-1, Z, -1).reshape(-1, 4)
        rotated = quat_apply(quat_e, centers_e.reshape(-1, 3)).reshape(E, Z, 3)
        world_centers = obj_pos_w.unsqueeze(1) + rotated  # (E, Z, 3)

        marker_indices = torch.zeros(E, dtype=torch.int32, device=env.device)
        for z_idx, viz in enumerate(self._visualizers):
            zone_quat_obj = self._quats_obj[z_idx].unsqueeze(0).expand(E, -1)
            zone_quat_w = quat_mul(obj_quat_w, zone_quat_obj)
            viz.visualize(
                translations=world_centers[:, z_idx, :],
                orientations=zone_quat_w,
                marker_indices=marker_indices,
            )

        return torch.zeros(env.num_envs, 0, device=env.device)


class contact_zones_vis(forbidden_zones_vis):
    """Render designated contact zones attached to the object frame."""


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
            hide_marker_from_cameras(self.visualizer)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


class camera_point_cloud_w(ManagerTermBase):
    """World-frame point cloud derived from the camera depth image.

    Unlike ``object_point_cloud_b``, this term does not use ground-truth object geometry. It
    back-projects the rendered depth image so the result naturally reflects occlusions and only
    includes surfaces visible to the camera. The output is downsampled or padded to a fixed size.

    Args (from ``cfg.params``):
        sensor_cfg: Scene entity for the camera sensor. Defaults to ``SceneEntityCfg("tiled_camera")``.
        table_cfg: Scene entity for the table used to derive a default crop box. Defaults to ``SceneEntityCfg("table")``.
        data_type: Depth buffer to back-project. Defaults to ``"distance_to_image_plane"``.
        num_points: Number of points returned per environment. Defaults to ``4096``.
        bbox_min_w: Explicit minimum world-frame XYZ crop bounds. If set together with ``bbox_max_w``, these
            override the table-derived crop box.
        bbox_max_w: Explicit maximum world-frame XYZ crop bounds. If set together with ``bbox_min_w``, these
            override the table-derived crop box.
        table_xy_inset: Optional XY inset applied symmetrically to the table bounds. Defaults to ``0.0``.
        table_z_min_offset: Offset above the table top used as the crop-box floor. Defaults to ``0.01``.
        table_z_max_offset: Offset above the table top used as the crop-box ceiling. Defaults to ``0.60``.
        flatten: If ``True``, return ``(num_envs, 3 * num_points)``. Otherwise return
            ``(num_envs, num_points, 3)``.
        visualize: Whether to draw the sampled world-frame points. Defaults to ``False``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.params.get("visualize", False):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            marker_cfg.markers["hit"].radius = 0.002
            self.visualizer = VisualizationMarkers(marker_cfg)
            hide_marker_from_cameras(self.visualizer)

    @staticmethod
    def _sample_or_pad_points(points_w: torch.Tensor, num_points: int) -> torch.Tensor:
        """Return exactly ``num_points`` world-frame points."""
        points_w = points_w.reshape(-1, 3)
        if points_w.shape[0] == 0:
            return torch.zeros((num_points, 3), device=points_w.device, dtype=points_w.dtype)
        if points_w.shape[0] >= num_points:
            indices = torch.randperm(points_w.shape[0], device=points_w.device)[:num_points]
            return points_w[indices]
        pad_indices = torch.randint(points_w.shape[0], (num_points - points_w.shape[0],), device=points_w.device)
        return torch.cat((points_w, points_w[pad_indices]), dim=0)

    @staticmethod
    def _filter_points_in_aabb(
        points_w: torch.Tensor,
        bbox_min_w: tuple[float, float, float] | None,
        bbox_max_w: tuple[float, float, float] | None,
    ) -> torch.Tensor:
        """Keep only points inside the provided world-frame axis-aligned bounding box."""
        if bbox_min_w is None or bbox_max_w is None:
            return points_w

        bbox_min = torch.tensor(bbox_min_w, device=points_w.device, dtype=points_w.dtype)
        bbox_max = torch.tensor(bbox_max_w, device=points_w.device, dtype=points_w.dtype)
        in_bounds = torch.logical_and(points_w >= bbox_min, points_w <= bbox_max).all(dim=1)
        return points_w[in_bounds]

    @staticmethod
    def _compute_table_aabb(
        table: RigidObject,
        env_id: int,
        device: str,
        dtype: torch.dtype,
        table_xy_inset: float,
        table_z_min_offset: float,
        table_z_max_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a world-frame crop box from the table pose and configured table size."""
        table_size = torch.tensor(table.cfg.spawn.size, device=device, dtype=dtype)
        table_pos_w = table.data.root_pos_w[env_id]
        half_extent_xy = table_size[:2] * 0.5 - table_xy_inset
        table_top_z = table_pos_w[2] + table_size[2] * 0.5

        bbox_min = torch.stack((
            table_pos_w[0] - half_extent_xy[0],
            table_pos_w[1] - half_extent_xy[1],
            table_top_z + table_z_min_offset,
        ))
        bbox_max = torch.stack((
            table_pos_w[0] + half_extent_xy[0],
            table_pos_w[1] + half_extent_xy[1],
            table_top_z + table_z_max_offset,
        ))
        return bbox_min, bbox_max

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
        data_type: str = "distance_to_image_plane",
        num_points: int = 4096,
        bbox_min_w: tuple[float, float, float] | None = None,
        bbox_max_w: tuple[float, float, float] | None = None,
        table_xy_inset: float = 0.0,
        table_z_min_offset: float = 0.01,
        table_z_max_offset: float = 0.60,
        flatten: bool = False,
        visualize: bool = False,
    ) -> torch.Tensor:
        """Compute a fixed-size world-frame point cloud from the camera depth image."""
        camera = env.scene[sensor_cfg.name]
        table = env.scene[table_cfg.name]
        depth_images = camera.data.output[data_type]
        point_cloud_w = torch.zeros((env.num_envs, num_points, 3), device=env.device, dtype=depth_images.dtype)

        for env_id in range(env.num_envs):
            depth_image = depth_images[env_id]
            if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
                depth_image = depth_image.squeeze(-1)

            visible_points_w = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[env_id],
                depth=depth_image,
                position=camera.data.pos_w[env_id],
                orientation=camera.data.quat_w_ros[env_id],
                device=env.device,
            ).reshape(-1, 3)
            visible_points_w = visible_points_w[torch.isfinite(visible_points_w).all(dim=1)]
            if bbox_min_w is None or bbox_max_w is None:
                bbox_min_tensor, bbox_max_tensor = self._compute_table_aabb(
                    table=table,
                    env_id=env_id,
                    device=env.device,
                    dtype=visible_points_w.dtype,
                    table_xy_inset=table_xy_inset,
                    table_z_min_offset=table_z_min_offset,
                    table_z_max_offset=table_z_max_offset,
                )
                visible_points_w = visible_points_w[
                    torch.logical_and(visible_points_w >= bbox_min_tensor, visible_points_w <= bbox_max_tensor).all(
                        dim=1
                    )
                ]
            else:
                visible_points_w = self._filter_points_in_aabb(visible_points_w, bbox_min_w, bbox_max_w)
            point_cloud_w[env_id] = self._sample_or_pad_points(visible_points_w, num_points)

        if visualize and hasattr(self, "visualizer"):
            self.visualizer.visualize(translations=point_cloud_w.reshape(-1, 3))

        return point_cloud_w.view(env.num_envs, -1) if flatten else point_cloud_w


class merged_camera_point_cloud_w(camera_point_cloud_w):
    """World-frame point cloud merged from several camera depth images.

    Behaves like :class:`camera_point_cloud_w`, but back-projects the depth images from
    multiple cameras, unions their visible world-frame points, applies one shared crop
    box, and downsamples/pads the union to ``num_points``. Merging complementary views
    fills occlusions that any single camera misses.

    Args (from ``cfg.params``):
        sensor_cfgs: Scene entities for the camera sensors to merge. Defaults to
            ``[SceneEntityCfg("tiled_camera")]``.
        table_cfg, data_type, num_points, bbox_min_w, bbox_max_w, table_xy_inset,
        table_z_min_offset, table_z_max_offset, flatten, visualize: Identical to
        :class:`camera_point_cloud_w`.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        # Skip camera_point_cloud_w.__init__ so the visualizer uses a distinct prim path.
        ManagerTermBase.__init__(self, cfg, env)
        if cfg.params.get("visualize", False):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/MergedCameraPointCloud")
            marker_cfg.markers["hit"].radius = 0.002
            self.visualizer = VisualizationMarkers(marker_cfg)
            hide_marker_from_cameras(self.visualizer)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfgs: list[SceneEntityCfg] | None = None,
        table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
        data_type: str = "distance_to_image_plane",
        num_points: int = 4096,
        bbox_min_w: tuple[float, float, float] | None = None,
        bbox_max_w: tuple[float, float, float] | None = None,
        table_xy_inset: float = 0.0,
        table_z_min_offset: float = 0.01,
        table_z_max_offset: float = 0.60,
        flatten: bool = False,
        visualize: bool = False,
    ) -> torch.Tensor:
        """Compute a fixed-size world-frame point cloud merged across cameras."""
        if sensor_cfgs is None:
            sensor_cfgs = [SceneEntityCfg("tiled_camera")]
        cameras = [env.scene[sensor_cfg.name] for sensor_cfg in sensor_cfgs]
        table = env.scene[table_cfg.name]
        dtype = cameras[0].data.output[data_type].dtype
        point_cloud_w = torch.zeros((env.num_envs, num_points, 3), device=env.device, dtype=dtype)

        for env_id in range(env.num_envs):
            per_camera_points = []
            for camera in cameras:
                depth_image = camera.data.output[data_type][env_id]
                if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
                    depth_image = depth_image.squeeze(-1)

                points_w = create_pointcloud_from_depth(
                    intrinsic_matrix=camera.data.intrinsic_matrices[env_id],
                    depth=depth_image,
                    position=camera.data.pos_w[env_id],
                    orientation=camera.data.quat_w_ros[env_id],
                    device=env.device,
                ).reshape(-1, 3)
                per_camera_points.append(points_w[torch.isfinite(points_w).all(dim=1)])

            visible_points_w = torch.cat(per_camera_points, dim=0)
            if bbox_min_w is None or bbox_max_w is None:
                bbox_min_tensor, bbox_max_tensor = self._compute_table_aabb(
                    table=table,
                    env_id=env_id,
                    device=env.device,
                    dtype=visible_points_w.dtype,
                    table_xy_inset=table_xy_inset,
                    table_z_min_offset=table_z_min_offset,
                    table_z_max_offset=table_z_max_offset,
                )
                visible_points_w = visible_points_w[
                    torch.logical_and(visible_points_w >= bbox_min_tensor, visible_points_w <= bbox_max_tensor).all(
                        dim=1
                    )
                ]
            else:
                visible_points_w = self._filter_points_in_aabb(visible_points_w, bbox_min_w, bbox_max_w)
            point_cloud_w[env_id] = self._sample_or_pad_points(visible_points_w, num_points)

        if visualize and hasattr(self, "visualizer"):
            self.visualizer.visualize(translations=point_cloud_w.reshape(-1, 3))

        return point_cloud_w.view(env.num_envs, -1) if flatten else point_cloud_w


class camera_point_cloud_c(ManagerTermBase):
    """Camera-frame point cloud derived from the camera depth image.

    This term back-projects depth into camera coordinates (ROS camera convention).
    For table/bbox cropping compatibility, points are temporarily transformed to world
    for mask computation, but the returned points remain in camera frame.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.params.get("visualize", False):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloudCameraFrame")
            marker_cfg.markers["hit"].radius = 0.002
            self.visualizer = VisualizationMarkers(marker_cfg)

    @staticmethod
    def _sample_or_pad_points(points_c: torch.Tensor, num_points: int) -> torch.Tensor:
        points_c = points_c.reshape(-1, 3)
        if points_c.shape[0] == 0:
            return torch.zeros((num_points, 3), device=points_c.device, dtype=points_c.dtype)
        if points_c.shape[0] >= num_points:
            indices = torch.randperm(points_c.shape[0], device=points_c.device)[:num_points]
            return points_c[indices]
        pad_indices = torch.randint(points_c.shape[0], (num_points - points_c.shape[0],), device=points_c.device)
        return torch.cat((points_c, points_c[pad_indices]), dim=0)

    @staticmethod
    def _compute_table_aabb(
        table: RigidObject,
        env_id: int,
        device: str,
        dtype: torch.dtype,
        table_xy_inset: float,
        table_z_min_offset: float,
        table_z_max_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table_size = torch.tensor(table.cfg.spawn.size, device=device, dtype=dtype)
        table_pos_w = table.data.root_pos_w[env_id]
        half_extent_xy = table_size[:2] * 0.5 - table_xy_inset
        table_top_z = table_pos_w[2] + table_size[2] * 0.5
        bbox_min = torch.stack((
            table_pos_w[0] - half_extent_xy[0],
            table_pos_w[1] - half_extent_xy[1],
            table_top_z + table_z_min_offset,
        ))
        bbox_max = torch.stack((
            table_pos_w[0] + half_extent_xy[0],
            table_pos_w[1] + half_extent_xy[1],
            table_top_z + table_z_max_offset,
        ))
        return bbox_min, bbox_max

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
        data_type: str = "distance_to_image_plane",
        num_points: int = 4096,
        bbox_min_w: tuple[float, float, float] | None = None,
        bbox_max_w: tuple[float, float, float] | None = None,
        table_xy_inset: float = 0.0,
        table_z_min_offset: float = 0.01,
        table_z_max_offset: float = 0.60,
        flatten: bool = False,
        visualize: bool = False,
    ) -> torch.Tensor:
        camera = env.scene[sensor_cfg.name]
        table = env.scene[table_cfg.name]
        depth_images = camera.data.output[data_type]
        point_cloud_c = torch.zeros((env.num_envs, num_points, 3), device=env.device, dtype=depth_images.dtype)

        for env_id in range(env.num_envs):
            depth_image = depth_images[env_id]
            if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
                depth_image = depth_image.squeeze(-1)

            visible_points_c = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[env_id],
                depth=depth_image,
                device=env.device,
            ).reshape(-1, 3)

            visible_points_c = visible_points_c[torch.isfinite(visible_points_c).all(dim=1)]
            if visible_points_c.shape[0] == 0:
                point_cloud_c[env_id] = self._sample_or_pad_points(visible_points_c, num_points)
                continue

            visible_points_w = transform_points(
                visible_points_c,
                position=camera.data.pos_w[env_id],
                orientation=camera.data.quat_w_ros[env_id],
                device=env.device,
            ).reshape(-1, 3)
            visible_points_w = visible_points_w[torch.isfinite(visible_points_w).all(dim=1)]

            if bbox_min_w is None or bbox_max_w is None:
                bbox_min_tensor, bbox_max_tensor = self._compute_table_aabb(
                    table=table,
                    env_id=env_id,
                    device=env.device,
                    dtype=visible_points_w.dtype,
                    table_xy_inset=table_xy_inset,
                    table_z_min_offset=table_z_min_offset,
                    table_z_max_offset=table_z_max_offset,
                )
            else:
                bbox_min_tensor = torch.tensor(bbox_min_w, device=env.device, dtype=visible_points_w.dtype)
                bbox_max_tensor = torch.tensor(bbox_max_w, device=env.device, dtype=visible_points_w.dtype)

            keep_mask = torch.logical_and(visible_points_w >= bbox_min_tensor, visible_points_w <= bbox_max_tensor).all(
                dim=1
            )
            visible_points_c = visible_points_c[keep_mask]
            point_cloud_c[env_id] = self._sample_or_pad_points(visible_points_c, num_points)

        if visualize and hasattr(self, "visualizer"):
            self.visualizer.visualize(translations=point_cloud_c.reshape(-1, 3))

        return point_cloud_c.view(env.num_envs, -1) if flatten else point_cloud_c


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)  # Shape: (num_envs, num_sensors, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    # Flatten to (num_envs, 3 * num_sensors) for concatenation with other observations
    return forces_b.view(env.num_envs, -1)


def max_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Maximum absolute joint position for the given asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    max_abs = torch.abs(joint_pos).max(dim=1).values
    return max_abs.unsqueeze(1)


def max_joint_pos_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Maximum joint position (signed) for the given asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    max_pos = joint_pos.max(dim=1).values
    return max_pos.unsqueeze(1)


class pour_progress_marker_vis(ManagerTermBase):
    """Sphere marker at the goal that animates color from red → green as a
    configured tilt criterion approaches its threshold.

    Visualization helper for pour-style tasks. The sphere position tracks the
    ``goal_asset_cfg`` root (i.e. the success marker / goal), and its color
    is selected from a precomputed gradient based on the current tilt
    progress — useful for "am I tilted enough yet?" feedback at a glance.

    Progress is the *max* of the two tilt-criteria progresses that
    :class:`mdp.lift_and_tilt_with_contact_zones` evaluates, each clamped to
    ``[0, 1]``:

    * Primary axis-vs-axis: ``angle / threshold_rad`` when ``primary_tilt_ge``
      is True (or its complement against ``π`` when False). With a negative
      ``primary_threshold_rad`` (effectively disabled), this contributes 1.
    * Plane-angle: ``axis_to_plane_angle(...) / plane_threshold_rad``.

    Either criterion may be omitted by leaving its threshold ``None``.

    Args (from ``cfg.params``):
        goal_asset_cfg: Scene asset whose root the marker tracks. Defaults to
            ``SceneEntityCfg("success_marker")``.
        object_cfg: Scene asset whose orientation drives the tilt metrics.
            Defaults to ``SceneEntityCfg("object")``.
        primary_threshold_rad: Threshold for ``axis_tilt_angle`` (rad).
        primary_axis_local / primary_world_axis / primary_tilt_ge: Mirror
            ``lift_and_tilt_with_contact_zones`` parameters.
        plane_threshold_rad: Threshold for ``axis_to_plane_angle`` (rad).
        plane_axis_local / plane_normal: Mirror the same parameters.
        radius: Sphere radius (m). Default 0.04.
        visible: Whether to render the marker. Default True.
        opacity: Material opacity in ``[0, 1]``. Default 0.7.
        num_color_steps: Number of discrete colors in the red→green gradient.
            Default 11 (≈10% steps).
        prim_path_prefix: USD prim path prefix for the markers. Default
            ``"/Visuals/PourProgressMarker"``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._goal_cfg: SceneEntityCfg = cfg.params.get("goal_asset_cfg", SceneEntityCfg("success_marker"))
        self._object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))

        self._primary_threshold = cfg.params.get("primary_threshold_rad")
        self._primary_axis_local = cfg.params.get("primary_axis_local", (0.0, 0.0, 1.0))
        self._primary_world_axis = cfg.params.get("primary_world_axis", (0.0, 0.0, 1.0))
        self._primary_tilt_ge = bool(cfg.params.get("primary_tilt_ge", True))

        self._plane_threshold = cfg.params.get("plane_threshold_rad")
        self._plane_axis_local = cfg.params.get("plane_axis_local")
        self._plane_normal = cfg.params.get("plane_normal", (0.0, 0.0, 1.0))

        radius = float(cfg.params.get("radius", 0.04))
        self._visible = bool(cfg.params.get("visible", True))
        opacity = float(cfg.params.get("opacity", 0.7))
        num_steps = int(cfg.params.get("num_color_steps", 11))
        prim_path_prefix = cfg.params.get("prim_path_prefix", "/Visuals/PourProgressMarker")

        if num_steps < 2:
            num_steps = 2
        self._num_steps = num_steps

        markers = {}
        for i in range(num_steps):
            t = i / (num_steps - 1)
            # Linear red→green; keeps the cue obvious without a yellow
            # intermediate. Tune in cfg if a different ramp is wanted.
            color = (1.0 - t, t, 0.0)
            markers[f"step_{i}"] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
            )

        self._visualizer = VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path_prefix, markers=markers))
        hide_marker_from_cameras(self._visualizer)
        self._visualizer.set_visibility(self._visible)

    def _primary_progress(self, quat_w: torch.Tensor) -> torch.Tensor:
        thr = self._primary_threshold
        if thr is None:
            return torch.zeros(quat_w.shape[0], device=quat_w.device, dtype=quat_w.dtype)
        thr = float(thr)
        angle = axis_tilt_angle(
            quat_w,
            axis_local=self._primary_axis_local,
            world_axis=self._primary_world_axis,
        )
        if self._primary_tilt_ge:
            if thr <= 0.0:
                return torch.ones_like(angle)
            return torch.clamp(angle / thr, 0.0, 1.0)
        # tilt_ge=False: success when angle <= thr; map "shrinking angle" to progress.
        denom = max(float(torch.pi) - thr, 1e-6)
        return torch.clamp((float(torch.pi) - angle) / denom, 0.0, 1.0)

    def _plane_progress(self, quat_w: torch.Tensor) -> torch.Tensor:
        thr = self._plane_threshold
        if thr is None or self._plane_axis_local is None:
            return torch.zeros(quat_w.shape[0], device=quat_w.device, dtype=quat_w.dtype)
        thr = max(float(thr), 1e-6)
        angle = axis_to_plane_angle(
            quat_w,
            axis_local=self._plane_axis_local,
            plane_normal=self._plane_normal,
        )
        return torch.clamp(angle / thr, 0.0, 1.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_asset_cfg: SceneEntityCfg | None = None,
        object_cfg: SceneEntityCfg | None = None,
        primary_threshold_rad: float | None = None,
        primary_axis_local: tuple[float, float, float] | None = None,
        primary_world_axis: tuple[float, float, float] | None = None,
        primary_tilt_ge: bool | None = None,
        plane_threshold_rad: float | None = None,
        plane_axis_local: tuple[float, float, float] | None = None,
        plane_normal: tuple[float, float, float] | None = None,
        radius: float | None = None,
        visible: bool | None = None,
        opacity: float | None = None,
        num_color_steps: int | None = None,
        prim_path_prefix: str | None = None,
    ) -> torch.Tensor:
        if visible is not None and bool(visible) != self._visible:
            self._visible = bool(visible)
            self._visualizer.set_visibility(self._visible)
        if not self._visible:
            return torch.zeros(env.num_envs, 0, device=env.device)

        goal_cfg = goal_asset_cfg or self._goal_cfg
        obj_cfg = object_cfg or self._object_cfg
        goal: RigidObject = env.scene[goal_cfg.name]
        obj: RigidObject = env.scene[obj_cfg.name]

        progress = torch.maximum(
            self._primary_progress(obj.data.root_quat_w),
            self._plane_progress(obj.data.root_quat_w),
        )
        indices = (progress * (self._num_steps - 1)).round().clamp(0, self._num_steps - 1).to(torch.int32)

        self._visualizer.visualize(
            translations=goal.data.root_pos_w,
            marker_indices=indices,
        )
        return torch.zeros(env.num_envs, 0, device=env.device)


def setup_fingertip_contact_observation(env_cfg, target_prim: str = "Object", clip: float = 20.0) -> None:
    """Wire fingertip→object contact sensors and the ``contact`` observation.

    Config-build helper (call from a task's ``__post_init__``). For each robot
    fingertip body, adds a :class:`ContactSensorCfg` filtered against
    ``{ENV_REGEX_NS}/<target_prim>`` and exposes the aggregated contact force as
    ``observations.contact.contact`` (set to ``None`` if the robot has no contact
    sensors). Also points the privileged ``hand_tips_state_b`` observation at the
    robot's hand-tip bodies.
    """
    if env_cfg.robot_config.setup_contact_sensors:
        tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
        finger_tip_body_list = env_cfg.robot_config.fingertip_body_names
        for link_name in finger_tip_body_list:
            setattr(
                env_cfg.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=f"{tip_prim_prefix}{link_name}",
                    filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{target_prim}"],
                ),
            )
        env_cfg.observations.contact.contact = ObsTerm(
            func=fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-clip, clip),
        )
    else:
        env_cfg.observations.contact = None
    env_cfg.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
        env_cfg.robot_config.hand_tips_body_names
    )
