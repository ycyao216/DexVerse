# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pivoting and extracting a book from a shelf."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop

# Book dimensions are (shelf-depth, shelf-width/thickness, height).
TARGET_BOOK_SIZE_M = (0.16, 0.03, 0.24)
TARGET_BOOK_MASS_KG = 0.35
NEIGHBOR_BOOK_SIZE_M = (0.14, 0.032, 0.25)

SHELF_FRONT_X_M = 0.02
SHELF_DEPTH_M = 0.24
SHELF_WIDTH_M = 0.42
SHELF_HEIGHT_M = 0.30
SHELF_BASE_THICKNESS_M = 0.02
SHELF_BACK_THICKNESS_M = 0.02
SHELF_SIDE_THICKNESS_M = 0.02

BOOK_BACK_CLEARANCE_M = 0.008
BOOK_SIDE_CLEARANCE_M = 0.006

TARGET_EXTRACTION_DISTANCE_M = 0.18
TARGET_LIFT_M = 0.05
TARGET_PITCH_RAD = -math.radians(65.0)

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6
SUCCESS_POSITION_THRESHOLD_M = 0.05
SUCCESS_ORIENTATION_THRESHOLD_RAD = 0.45
TILT_PROGRESS_THRESHOLD_RAD = math.radians(30.0)


def _make_book_cfg(
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    mass: float | None = None,
    kinematic: bool = False,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                kinematic_enabled=kinematic,
                disable_gravity=kinematic,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.9,
                metallic=0.0,
            ),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_panel_cfg(
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float] = (0.40, 0.30, 0.22),
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.95,
                metallic=0.0,
            ),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


OBJECT_CFG = _make_book_cfg("Object", TARGET_BOOK_SIZE_M, (0.72, 0.18, 0.12), mass=TARGET_BOOK_MASS_KG)
LEFT_BOOK_CFG = _make_book_cfg("NeighborBookLeft", NEIGHBOR_BOOK_SIZE_M, (0.20, 0.38, 0.75), kinematic=True)
RIGHT_BOOK_CFG = _make_book_cfg("NeighborBookRight", NEIGHBOR_BOOK_SIZE_M, (0.82, 0.70, 0.22), kinematic=True)
SHELF_BASE_CFG = _make_panel_cfg("ShelfBase", (SHELF_DEPTH_M, SHELF_WIDTH_M, SHELF_BASE_THICKNESS_M))
SHELF_BACK_CFG = _make_panel_cfg("ShelfBack", (SHELF_BACK_THICKNESS_M, SHELF_WIDTH_M, SHELF_HEIGHT_M))
SHELF_SIDE_LEFT_CFG = _make_panel_cfg("ShelfSideLeft", (SHELF_DEPTH_M, SHELF_SIDE_THICKNESS_M, SHELF_HEIGHT_M))
SHELF_SIDE_RIGHT_CFG = _make_panel_cfg("ShelfSideRight", (SHELF_DEPTH_M, SHELF_SIDE_THICKNESS_M, SHELF_HEIGHT_M))


@configclass
class TakeBookOffShelfCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for the take-book-off-shelf task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="object",
        resampling_time_range=(12.0, 12.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(TARGET_PITCH_RAD, TARGET_PITCH_RAD),
            yaw=(0.0, 0.0),
        ),
        position_only=False,
    )


@configclass
class TakeBookOffShelfObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the take-book-off-shelf task.

    Book pose + up-axis in ``proprio``; velocities in ``privileged``. Goal
    pose + goal up-axis live in ``goal``. ``object_state_b`` is
    kept as a 13-vec composite in proprio because the policy consumes it
    that way.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class GoalObsCfg(ObsGroup):
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )
        target_object_up_b = ObsTerm(
            func=mdp.target_rot_axis_b,
            params={"command_name": "object_pose", "axis_local": (0.0, 0.0, 1.0)},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class TakeBookOffShelfRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pivoting the target book out of the shelf."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.35,
            "distance_gain": 8.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=1.5,
    )

    outward_tilt = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=1.5,
        params={
            "threshold_rad": TILT_PROGRESS_THRESHOLD_RAD,
            "axis_local": (0.0, 0.0, 1.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": True,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=5.0,
        params={
            "std": 0.12,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10.0,
        params={
            "command_name": "object_pose",
            "pos_std": SUCCESS_POSITION_THRESHOLD_M,
            "rot_std": SUCCESS_ORIENTATION_THRESHOLD_RAD,
        },
    )


@configclass
class TakeBookOffShelfTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the take-book-off-shelf task."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.object_at_goal_pose,
        params={
            "command_name": "object_pose",
            "position_threshold": SUCCESS_POSITION_THRESHOLD_M,
            "orientation_threshold": SUCCESS_ORIENTATION_THRESHOLD_RAD,
        },
    )


@configclass
class TakeBookOffShelfSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = OBJECT_CFG
    neighbor_book_left: RigidObjectCfg = LEFT_BOOK_CFG
    neighbor_book_right: RigidObjectCfg = RIGHT_BOOK_CFG
    shelf_base: RigidObjectCfg = SHELF_BASE_CFG
    shelf_back: RigidObjectCfg = SHELF_BACK_CFG
    shelf_side_left: RigidObjectCfg = SHELF_SIDE_LEFT_CFG
    shelf_side_right: RigidObjectCfg = SHELF_SIDE_RIGHT_CFG


@configclass
class TakeBookOffShelfEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for resetting the three-book cluster."""

    reset_object = None
    reset_book_cluster = EventTerm(
        func=mdp.reset_book_cluster_and_command,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "left_book_cfg": SceneEntityCfg("neighbor_book_left"),
            "right_book_cfg": SceneEntityCfg("neighbor_book_right"),
            "y_range": (0.0, 0.0),
            "target_pos_x": 0.0,
            "target_pos_z": 0.0,
            "target_pitch_rad": TARGET_PITCH_RAD,
            "command_name": "object_pose",
        },
    )


@configclass
class TakeBookOffShelfEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pivot a book outward and extract it from a simple tabletop shelf."""

    supports_object_pose_command: bool = True

    commands: TakeBookOffShelfCommandsCfg = TakeBookOffShelfCommandsCfg()
    observations: TakeBookOffShelfObservationsCfg = TakeBookOffShelfObservationsCfg()
    rewards: TakeBookOffShelfRewardsCfg = TakeBookOffShelfRewardsCfg()
    terminations: TakeBookOffShelfTerminationsCfg = TakeBookOffShelfTerminationsCfg()
    events: TakeBookOffShelfEventCfg = TakeBookOffShelfEventCfg()
    scene: TakeBookOffShelfSceneCfg = TakeBookOffShelfSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
        neighbor_book_left=LEFT_BOOK_CFG,
        neighbor_book_right=RIGHT_BOOK_CFG,
        shelf_base=SHELF_BASE_CFG,
        shelf_back=SHELF_BACK_CFG,
        shelf_side_left=SHELF_SIDE_LEFT_CFG,
        shelf_side_right=SHELF_SIDE_RIGHT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.use_world_frame = True
        self.commands.object_pose.position_only = False

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        shelf_base_center_x = SHELF_FRONT_X_M + 0.5 * SHELF_DEPTH_M
        shelf_center_z = table_top_z + SHELF_BASE_THICKNESS_M + 0.5 * SHELF_HEIGHT_M
        shelf_back_center_x = SHELF_FRONT_X_M + SHELF_DEPTH_M - 0.5 * SHELF_BACK_THICKNESS_M
        side_center_y = 0.5 * (SHELF_WIDTH_M - SHELF_SIDE_THICKNESS_M)
        book_center_z = table_top_z + SHELF_BASE_THICKNESS_M + 0.5 * TARGET_BOOK_SIZE_M[2]
        book_init_x = (
            SHELF_FRONT_X_M
            + SHELF_DEPTH_M
            - SHELF_BACK_THICKNESS_M
            - BOOK_BACK_CLEARANCE_M
            - 0.5 * TARGET_BOOK_SIZE_M[0]
        )
        neighbor_center_y = 0.5 * (TARGET_BOOK_SIZE_M[1] + NEIGHBOR_BOOK_SIZE_M[1]) + BOOK_SIDE_CLEARANCE_M

        self.scene.object.init_state.pos = (book_init_x, 0.0, book_center_z)
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.neighbor_book_left.init_state.pos = (book_init_x, neighbor_center_y, book_center_z)
        self.scene.neighbor_book_right.init_state.pos = (book_init_x, -neighbor_center_y, book_center_z)

        self.scene.shelf_base.init_state.pos = (shelf_base_center_x, 0.0, table_top_z + 0.5 * SHELF_BASE_THICKNESS_M)
        self.scene.shelf_back.init_state.pos = (shelf_back_center_x, 0.0, shelf_center_z)
        self.scene.shelf_side_left.init_state.pos = (shelf_base_center_x, side_center_y, shelf_center_z)
        self.scene.shelf_side_right.init_state.pos = (shelf_base_center_x, -side_center_y, shelf_center_z)

        target_x = book_init_x - TARGET_EXTRACTION_DISTANCE_M
        target_z = book_center_z + TARGET_LIFT_M
        self.commands.object_pose.ranges.pos_x = (target_x, target_x)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (target_z, target_z)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (TARGET_PITCH_RAD, TARGET_PITCH_RAD)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        inner_half_width = 0.5 * (SHELF_WIDTH_M - 2.0 * SHELF_SIDE_THICKNESS_M)
        cluster_half_width = neighbor_center_y + 0.5 * NEIGHBOR_BOOK_SIZE_M[1]
        cluster_y_limit = max(inner_half_width - cluster_half_width - BOOK_SIDE_CLEARANCE_M, 0.0)
        self.events.reset_book_cluster.params["y_range"] = (-cluster_y_limit, cluster_y_limit)
        self.events.reset_book_cluster.params["target_pos_x"] = target_x
        self.events.reset_book_cluster.params["target_pos_z"] = target_z

        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            half_x = table_size[0] * 0.5
            half_y = table_size[1] * 0.5
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-half_x, half_x),
                "y": (-half_y, half_y),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class TakeBookOffShelfEnvFloatingDexHandRightCfg(TakeBookOffShelfEnvCfg):
    """Take-book-off-shelf config with unified floating-hand teleop defaults."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
