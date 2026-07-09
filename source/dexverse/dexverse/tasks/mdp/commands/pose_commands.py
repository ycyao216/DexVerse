# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from dexverse.visual_purpose import hide_marker_from_cameras
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_from_euler_xyz,
    quat_unique,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import pose_commands_cfg as dex_cmd_cfgs


class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object.

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets can be defined in either the robot's *base frame* or *world frame*
        based on the `use_world_frame` config flag. For metrics/visualization,
        targets are always transformed/computed in the world frame.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.
        If `use_world_frame=True`, this is in world frame. Otherwise, it's in robot base frame.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        whether to use world frame, and optional visualization settings.
    """

    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # Only initialize success visualizer if markers are configured
        # VisualizationMarkers requires at least one marker, so we skip initialization if empty
        if len(self.cfg.success_visualizer_cfg.markers) > 0:
            self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
            hide_marker_from_cameras(self.success_visualizer)
            self.success_visualizer.set_visibility(True)
        else:
            self.success_visualizer = None

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command to world frame if needed
        if self.cfg.use_world_frame:
            # Commands are already in world frame
            self.pose_command_w[:, :3] = self.pose_command_b[:, :3]
            self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:]
        else:
            # transform command from base frame to simulation world frame
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.pose_command_b[:, :3],
                self.pose_command_b[:, 3:],
            )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        success_id = self.metrics["position_error"] < 0.05
        if not self.cfg.position_only:
            success_id &= self.metrics["orientation_error"] < 0.5
        # Only visualize if success visualizer is initialized (i.e., markers are configured)
        if self.success_visualizer is not None:
            self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.use_world_frame:
            # Sample directly in world frame
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
            # When using world frame, also update pose_command_w immediately
            self.pose_command_w[env_ids, :3] = self.pose_command_b[env_ids, :3]
        else:
            # Sample in robot base frame (original behavior)
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        quat_final = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.pose_command_b[env_ids, 3:] = quat_final
        # When using world frame, also update pose_command_w orientation immediately
        if self.cfg.use_world_frame:
            self.pose_command_w[env_ids, 3:] = quat_final

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # -- goal pose
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                hide_marker_from_cameras(self.goal_visualizer)
                # -- current body pose
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
                hide_marker_from_cameras(self.curr_visualizer)
            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # marker indices: 0=frame, 1=position_far (red), 2=position_near (green)
        # Always show frame markers (index 0) for both goal and current poses
        frame_marker_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if not self.cfg.position_only:
            # -- goal pose: show frame at goal position with goal orientation
            self.goal_visualizer.visualize(
                translations=self.pose_command_w[:, :3],
                orientations=self.pose_command_w[:, 3:],
                marker_indices=frame_marker_indices,
            )
            # -- current object pose: show frame at current position with current orientation
            self.curr_visualizer.visualize(
                translations=self.object.data.root_pos_w[:, :3],
                orientations=self.object.data.root_quat_w,
                marker_indices=frame_marker_indices,
            )
        else:
            # -- goal pose: show frame at goal position with goal orientation
            self.goal_visualizer.visualize(
                translations=self.pose_command_w[:, :3],
                orientations=self.pose_command_w[:, 3:],
                marker_indices=frame_marker_indices,
            )
            # -- current object pose: show frame at current position with current orientation
            self.curr_visualizer.visualize(
                translations=self.object.data.root_pos_w[:, :3],
                orientations=self.object.data.root_quat_w,
                marker_indices=frame_marker_indices,
            )


class ObjectAssetTrackingPoseCommand(ObjectUniformPoseCommand):
    """Pose command whose position tracks a target asset's pose plus a local offset.

    Useful when the goal should follow a per-env-randomized prop (e.g. a
    place-target whose location varies each episode). Orientation is still
    drawn from the configured ranges (so visualizers and metrics behave
    identically to :class:`ObjectUniformPoseCommand`); the position is
    overwritten every step from
    ``target.root_pos_w + R(target.root_quat_w) * local_offset``, which keeps
    the goal locked to the target even if it moves at simulation time.
    """

    cfg: dex_cmd_cfgs.ObjectAssetTrackingPoseCommandCfg

    def __init__(self, cfg: dex_cmd_cfgs.ObjectAssetTrackingPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.target = env.scene[cfg.target_asset_name]
        self._local_offset = torch.tensor(cfg.local_offset, device=self.device, dtype=torch.float32)

    def _goal_world_position(self) -> torch.Tensor:
        target_pos = self.target.data.root_pos_w
        target_quat = self.target.data.root_quat_w
        offset_w = quat_apply(
            target_quat,
            self._local_offset.unsqueeze(0).expand(self.num_envs, -1),
        )
        return target_pos + offset_w

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        goal_pos = self._goal_world_position()
        self.pose_command_b[env_ids, :3] = goal_pos[env_ids]
        if self.cfg.use_world_frame:
            self.pose_command_w[env_ids, :3] = goal_pos[env_ids]

    def _update_command(self):
        # Refresh every step so the goal follows the target if it moves.
        goal_pos = self._goal_world_position()
        self.pose_command_b[:, :3] = goal_pos
        if self.cfg.use_world_frame:
            self.pose_command_w[:, :3] = goal_pos
