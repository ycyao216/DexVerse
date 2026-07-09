# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pushing a large sphere up a tabletop slope."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ..robot_init import set_robot_wrist_init_world_pos

# Matches SMALL_SPHERE_RADIUS in push_small_sphere_obstacle_slope_cfg.py.
# Drives the object collider, its spawn height (table_top_z + radius), the goal
# surface offset, and all goal/success markers.
SPHERE_RADIUS = 0.08
SPHERE_MASS_KG = 2.5
TARGET_OPACITY = 0.30
RESET_Y_RANGE = 0.05
SUCCESS_THRESHOLD_M = 0.08
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6
# Initial palm world position in m. Drives a joint translation on floating
# Shadow/Leap and an arm IK target on UR10e (= floating_shadow base + 0.2).
ROBOT_INIT_PALM_WORLD_X = -0.55

SLOPE_LENGTH = 0.80
SLOPE_WIDTH = 0.90
SLOPE_THICKNESS = 0.04
SLOPE_ANGLE_DEG = 15.0
SLOPE_ANGLE_RAD = math.radians(SLOPE_ANGLE_DEG)
SLOPE_CENTER_X = 0.48
SLOPE_CENTER_Y = 0.0
SLOPE_HEIGHT_OFFSET = -0.04
GOAL_OFFSET_FROM_TOP_M = 0.18
# Per-episode random spread (+/- m) of the goal target across the slope width
# (world y). Sampled uniformly each time the command resamples. Bounded by
# 0.5*SLOPE_WIDTH - SPHERE_RADIUS (= 0.45 - 0.08 = 0.37) so the goal sphere
# stays fully between the side guards.
GOAL_Y_RANDOM_RANGE = 0.30
START_OFFSET_FROM_SLOPE_M = 0.14
SIDE_GUARD_THICKNESS = 0.04
SIDE_GUARD_HEIGHT = 0.25
SIDE_GUARD_COLOR = (0.30, 0.30, 0.35)

SLOPE_TANGENT = (math.cos(SLOPE_ANGLE_RAD), 0.0, math.sin(SLOPE_ANGLE_RAD))
SLOPE_NORMAL = (-math.sin(SLOPE_ANGLE_RAD), 0.0, math.cos(SLOPE_ANGLE_RAD))
SLOPE_QUAT = (math.cos(SLOPE_ANGLE_RAD / 2.0), 0.0, -math.sin(SLOPE_ANGLE_RAD / 2.0), 0.0)


OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.SphereCfg(
        radius=SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=SPHERE_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.30, 0.20)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

SLOPE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Slope",
    spawn=sim_utils.CuboidCfg(
        size=(SLOPE_LENGTH, SLOPE_WIDTH, SLOPE_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25), roughness=0.9),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SLOPE_QUAT),
)


def _make_side_guard_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(SLOPE_LENGTH, SIDE_GUARD_THICKNESS, SIDE_GUARD_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SIDE_GUARD_COLOR, roughness=0.9),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SLOPE_QUAT),
    )


SIDE_GUARD_LEFT_CFG = _make_side_guard_cfg("SideGuardLeft")
SIDE_GUARD_RIGHT_CFG = _make_side_guard_cfg("SideGuardRight")


def _surface_centerline_point(
    slope_center: tuple[float, float, float],
    tangent_scale: float,
    normal_scale: float,
) -> tuple[float, float, float]:
    """Return a point on the slope centerline using tangent/normal offsets."""
    return (
        slope_center[0] + tangent_scale * SLOPE_TANGENT[0] + normal_scale * SLOPE_NORMAL[0],
        slope_center[1] + tangent_scale * SLOPE_TANGENT[1] + normal_scale * SLOPE_NORMAL[1],
        slope_center[2] + tangent_scale * SLOPE_TANGENT[2] + normal_scale * SLOPE_NORMAL[2],
    )


@configclass
class PushSphereUpSlopeCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for the push-sphere-up-slope task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(8.0, 8.0),
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
        success_vis_asset_name="object",
        position_only=True,
    )


@configclass
class PushSphereUpSlopeObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for push-sphere-up-slope.

    Sphere position / orientation in ``proprio``; velocities in
    ``privileged``. Commanded goal position lives in ``goal`` (the quat
    component is irrelevant for a sphere but harmless).
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

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

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class PushSphereUpSlopeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for the push-sphere-up-slope task."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.5,
            "distance_gain": 4.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=1.5,
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=5.0,
        params={
            "std": 0.20,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10.0,
        params={
            "pos_std": SUCCESS_THRESHOLD_M,
            "rot_std": None,
            "command_name": "object_pose",
        },
    )


@configclass
class PushSphereUpSlopeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the push-sphere-up-slope task."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.object_at_goal_position,
        params={
            "command_name": "object_pose",
            "threshold": SUCCESS_THRESHOLD_M,
        },
    )


@configclass
class PushSphereUpSlopeSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = OBJECT_CFG
    slope: RigidObjectCfg = SLOPE_CFG
    side_guard_left: RigidObjectCfg = SIDE_GUARD_LEFT_CFG
    side_guard_right: RigidObjectCfg = SIDE_GUARD_RIGHT_CFG


@configclass
class PushSphereUpSlopeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Push a large sphere from the foot of a slope to its top."""

    supports_object_pose_command: bool = True

    commands: PushSphereUpSlopeCommandsCfg = PushSphereUpSlopeCommandsCfg()
    observations: PushSphereUpSlopeObservationsCfg = PushSphereUpSlopeObservationsCfg()
    rewards: PushSphereUpSlopeRewardsCfg = PushSphereUpSlopeRewardsCfg()
    terminations: PushSphereUpSlopeTerminationsCfg = PushSphereUpSlopeTerminationsCfg()
    scene: PushSphereUpSlopeSceneCfg = PushSphereUpSlopeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
        slope=SLOPE_CFG,
        side_guard_left=SIDE_GUARD_LEFT_CFG,
        side_guard_right=SIDE_GUARD_RIGHT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # Start the hand farther back so it does not intersect the sphere at reset.
        set_robot_wrist_init_world_pos(self, x=ROBOT_INIT_PALM_WORLD_X)

        self.episode_length_s = 20.0
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.position_only = True
        self.commands.object_pose.use_world_frame = True

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        slope_center_z = (
            table_top_z
            + 0.5 * (SLOPE_THICKNESS * math.cos(SLOPE_ANGLE_RAD) + SLOPE_LENGTH * math.sin(SLOPE_ANGLE_RAD))
            + SLOPE_HEIGHT_OFFSET
        )
        slope_center = (SLOPE_CENTER_X, SLOPE_CENTER_Y, slope_center_z)
        self.scene.slope.init_state.pos = slope_center
        self.scene.slope.init_state.rot = SLOPE_QUAT
        side_guard_center_y = 0.5 * SLOPE_WIDTH + 0.5 * SIDE_GUARD_THICKNESS
        side_guard_normal_offset = 0.5 * SLOPE_THICKNESS + 0.5 * SIDE_GUARD_HEIGHT
        for side_guard, center_y in (
            (self.scene.side_guard_left, side_guard_center_y),
            (self.scene.side_guard_right, -side_guard_center_y),
        ):
            side_guard.init_state.pos = _surface_centerline_point(
                slope_center=slope_center,
                tangent_scale=0.0,
                normal_scale=side_guard_normal_offset,
            )
            side_guard.init_state.pos = (
                side_guard.init_state.pos[0],
                slope_center[1] + center_y,
                side_guard.init_state.pos[2],
            )
            side_guard.init_state.rot = SLOPE_QUAT

        lower_surface_point = _surface_centerline_point(
            slope_center=slope_center,
            tangent_scale=-0.5 * SLOPE_LENGTH,
            normal_scale=0.5 * SLOPE_THICKNESS,
        )

        sphere_start = (
            lower_surface_point[0] - START_OFFSET_FROM_SLOPE_M,
            lower_surface_point[1],
            table_top_z + SPHERE_RADIUS,
        )
        self.scene.object.init_state.pos = sphere_start
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        goal_surface_point = _surface_centerline_point(
            slope_center=slope_center,
            tangent_scale=0.5 * SLOPE_LENGTH - GOAL_OFFSET_FROM_TOP_M,
            normal_scale=0.5 * SLOPE_THICKNESS,
        )
        goal_center = (
            goal_surface_point[0] + SPHERE_RADIUS * SLOPE_NORMAL[0],
            goal_surface_point[1] + SPHERE_RADIUS * SLOPE_NORMAL[1],
            goal_surface_point[2] + SPHERE_RADIUS * SLOPE_NORMAL[2],
        )

        self.commands.object_pose.ranges.pos_x = (goal_center[0], goal_center[0])
        # Randomize the goal across the slope width (world y); x/z stay fixed on
        # the slope surface (the slope only tilts in x, so surface z is constant
        # across y).
        self.commands.object_pose.ranges.pos_y = (
            goal_center[1] - GOAL_Y_RANDOM_RANGE,
            goal_center[1] + GOAL_Y_RANDOM_RANGE,
        )
        self.commands.object_pose.ranges.pos_z = (goal_center[2], goal_center[2])

        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.20, 0.60, 0.90),
                        opacity=TARGET_OPACITY,
                    ),
                )
            },
        )
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.30, 0.20)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.80, 0.25)),
                ),
            },
        )

        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-RESET_Y_RANGE, RESET_Y_RANGE],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi, math.pi],
            }

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
class PushSphereUpSlopeEnvFloatingDexHandRightCfg(PushSphereUpSlopeEnvCfg):
    """Push-sphere-up-slope config for floating dex hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
