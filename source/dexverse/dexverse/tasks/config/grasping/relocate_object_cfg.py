# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relocate-object task: move a rigid object to a commanded goal position.

The goal is sampled uniformly within ``target_height_range`` above the table
top and held fixed for the episode.  A semi-transparent sphere visualises the
goal position.

To use a different object, replace ``scene.object`` and set
``object_half_height`` to match the new geometry.
"""


import isaaclab.sim as sim_utils
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import (
    DEFAULT_OBJECT_CFG,
    PickupObjectEnvCfg,
    PickupObjectObservationsCfg,
    PickupObjectRewardsCfg,
    PickupObjectTerminationsCfg,
)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectCommandsCfg(dexverse_base_env.CommandsCfg):
    """Samples a fixed goal position once per episode."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.10, 0.20),  # overwritten from target_height_range in __post_init__
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="object",
        position_only=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectObservationsCfg(PickupObjectObservationsCfg):
    """Adds the commanded goal pose to the ``goal`` group."""

    @configclass
    class GoalObsCfg(ObsGroup):
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    goal: GoalObsCfg = GoalObsCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectRewardsCfg(PickupObjectRewardsCfg):
    """Extends shared rewards with goal-tracking and success bonus."""

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=2.0,
        params={"std": 0.15, "command_name": "object_pose"},
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=8.0,
        params={"pos_std": 0.05, "rot_std": None, "command_name": "object_pose"},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectTerminationsCfg(PickupObjectTerminationsCfg):
    """Adds a goal-proximity success termination."""

    success = DoneTerm(
        func=mdp.object_at_goal_position,
        params={"command_name": "object_pose", "threshold": 0.03},
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectEnvCfg(PickupObjectEnvCfg):
    """Relocate-object task configuration (base, robot-agnostic)."""

    supports_object_pose_command: bool = True
    # Height above the table top within which the goal is sampled (metres).
    target_height_range: tuple = (0.15, 0.25)
    # Half-extent of the xy box (around table center) the goal is sampled from.
    goal_xy_half_extent: float = 0.2
    # Yaw jitter applied on top of each object's upright placement at reset.
    object_init_yaw_range: tuple = (0.0, 0.0)

    scene: dexverse_base_env.SceneCfg = dexverse_base_env.SceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=DEFAULT_OBJECT_CFG,
    )
    observations: RelocateObjectObservationsCfg = RelocateObjectObservationsCfg()
    commands: RelocateObjectCommandsCfg = RelocateObjectCommandsCfg()
    rewards: RelocateObjectRewardsCfg = RelocateObjectRewardsCfg()
    terminations: RelocateObjectTerminationsCfg = RelocateObjectTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # Fix goal to a single position per episode, relative to the table.
        self.commands.object_pose.body_name = self.robot_config.palm_body_name
        self.commands.object_pose.resampling_time_range = (
            self.episode_length_s + 1.0,
            self.episode_length_s + 1.0,
        )
        self.commands.object_pose.use_world_frame = True
        h = self.goal_xy_half_extent
        self.commands.object_pose.ranges.pos_x = (table_pos[0] - h, table_pos[0] + h)
        self.commands.object_pose.ranges.pos_y = (table_pos[1] - h, table_pos[1] + h)
        self.commands.object_pose.ranges.pos_z = (
            table_top_z + self.target_height_range[0],
            table_top_z + self.target_height_range[1],
        )

        # Widen the object reset so x jitters by the same half-side as y, and
        # apply a yaw spin on top of each object's upright placement.
        half_side = self.center_square_size * 0.5
        if self.events.reset_object is not None:
            pose_range = self.events.reset_object.params["pose_range"]
            pose_range["x"] = [-half_side, half_side]
            pose_range["yaw"] = list(self.object_init_yaw_range)

        # Semi-transparent sphere at the goal position.
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.03,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.9),
                        opacity=0.35,
                    ),
                )
            },
        )
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=0.03,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=0.03,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                ),
            },
        )


# ---------------------------------------------------------------------------
# Concrete robot config
# ---------------------------------------------------------------------------


@configclass
class RelocateObjectEnvFloatingDexHandRightCfg(RelocateObjectEnvCfg):
    """Relocate-object config for floating dexterous hands (Shadow / Leap)."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
