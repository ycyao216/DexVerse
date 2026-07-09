# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pivoting a large thin cuboid upright against a tabletop wall."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
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

OBJECT_SIZE_M = (0.30, 0.16, 0.02)
OBJECT_MASS_KG = 0.75
OBJECT_STATIC_FRICTION = 1.6
OBJECT_DYNAMIC_FRICTION = 1.4
TABLE_STATIC_FRICTION = 0.05
TABLE_DYNAMIC_FRICTION = 0.03
CONTACT_RESTITUTION = 0.0
SUCCESS_AXIS_LOCAL = (-1.0, 0.0, 0.0)
ALIGNMENT_REWARD_THRESHOLD_RAD = math.radians(90.0)
SUCCESS_MAX_TILT_RAD = math.radians(20.0)
SUCCESS_MIN_HEIGHT_M = 0.2
SUCCESS_GOAL_PITCH_RAD = math.pi * 0.5

WALL_THICKNESS_M = 0.02
WALL_WIDTH_M = 0.40
WALL_HEIGHT_M = 0.36
WALL_CENTER_X_M = 0.22
OBJECT_WALL_GAP_M = 0.04

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6


OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=OBJECT_SIZE_M,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=OBJECT_STATIC_FRICTION,
            dynamic_friction=OBJECT_DYNAMIC_FRICTION,
            restitution=CONTACT_RESTITUTION,
            friction_combine_mode="average",
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.80, 0.55, 0.22),
            roughness=0.85,
            metallic=0.0,
        ),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)

WALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Wall",
    spawn=sim_utils.CuboidCfg(
        size=(WALL_THICKNESS_M, WALL_WIDTH_M, WALL_HEIGHT_M),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.35, 0.38),
            roughness=0.95,
            metallic=0.0,
        ),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)


@configclass
class PivotLargeCuboidAgainstWallObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for pivoting a large cuboid against a wall.

    Object pose / up-axis / long-axis in ``proprio``; velocities in
    ``privileged``. Commanded target pose + long-axis live in ``goal``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_long_axis_b = ObsTerm(
            func=mdp.object_rot_axis_b,
            noise=Unoise(n_min=0.0, n_max=0.0),
            params={"axis_local": SUCCESS_AXIS_LOCAL},
        )
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=0.0, n_max=0.0))

    @configclass
    class GoalObsCfg(ObsGroup):
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )
        target_object_long_axis_b = ObsTerm(
            func=mdp.target_rot_axis_b,
            params={"command_name": "object_pose", "axis_local": SUCCESS_AXIS_LOCAL},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class PivotLargeCuboidAgainstWallCommandsCfg(dexverse_base_env.CommandsCfg):
    """Goal-pose command for visualizing success-aligned target frame."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="object",
        resampling_time_range=(21.0, 21.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        position_only=False,
    )


@configclass
class PivotLargeCuboidAgainstWallRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pivoting a large cuboid upright."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 8.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    vertical_alignment = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=4.0,
        params={
            "threshold_rad": ALIGNMENT_REWARD_THRESHOLD_RAD,
            "axis_local": SUCCESS_AXIS_LOCAL,
            "tilt_ge": False,
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": SUCCESS_MIN_HEIGHT_M,
        },
    )


@configclass
class PivotLargeCuboidAgainstWallTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for pivoting a large cuboid upright."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.lift_and_tilt,
        params={
            "min_height": SUCCESS_MIN_HEIGHT_M,
            "threshold_rad": SUCCESS_MAX_TILT_RAD,
            "axis_local": SUCCESS_AXIS_LOCAL,
            "tilt_ge": False,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PivotLargeCuboidAgainstWallSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = OBJECT_CFG
    wall: RigidObjectCfg = WALL_CFG


@configclass
class PivotLargeCuboidAgainstWallEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pivot a large thin cuboid from flat-on-table to upright against a wall."""

    supports_object_pose_command: bool = True

    commands: PivotLargeCuboidAgainstWallCommandsCfg = PivotLargeCuboidAgainstWallCommandsCfg()
    observations: PivotLargeCuboidAgainstWallObservationsCfg = PivotLargeCuboidAgainstWallObservationsCfg()
    rewards: PivotLargeCuboidAgainstWallRewardsCfg = PivotLargeCuboidAgainstWallRewardsCfg()
    terminations: PivotLargeCuboidAgainstWallTerminationsCfg = PivotLargeCuboidAgainstWallTerminationsCfg()
    scene: PivotLargeCuboidAgainstWallSceneCfg = PivotLargeCuboidAgainstWallSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
        wall=WALL_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        object_init_x = WALL_CENTER_X_M - 0.5 * WALL_THICKNESS_M - 0.5 * OBJECT_SIZE_M[0] - OBJECT_WALL_GAP_M

        self.scene.object.init_state.pos = (object_init_x, 0.0, table_top_z + 0.5 * OBJECT_SIZE_M[2])
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.wall.spawn.size = (WALL_THICKNESS_M, WALL_WIDTH_M, WALL_HEIGHT_M)
        self.scene.wall.init_state.pos = (WALL_CENTER_X_M, 0.0, table_top_z + 0.5 * WALL_HEIGHT_M)
        self.scene.wall.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=TABLE_STATIC_FRICTION,
            dynamic_friction=TABLE_DYNAMIC_FRICTION,
            restitution=CONTACT_RESTITUTION,
            friction_combine_mode="min",
        )

        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-0.06, 0.06],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }

        goal_z = table_top_z + 0.5 * OBJECT_SIZE_M[2] + SUCCESS_MIN_HEIGHT_M
        self.commands.object_pose.use_world_frame = True
        self.commands.object_pose.position_only = False
        self.commands.object_pose.ranges.pos_x = (object_init_x, object_init_x)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (goal_z, goal_z)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (SUCCESS_GOAL_PITCH_RAD, SUCCESS_GOAL_PITCH_RAD)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

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
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class PivotLargeCuboidAgainstWallEnvFloatingDexHandRightCfg(PivotLargeCuboidAgainstWallEnvCfg):
    """Pivot-large-cuboid config for floating dex hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
