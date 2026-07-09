# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pushing a small sphere uphill while avoiding obstacles."""

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
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

SMALL_SPHERE_RADIUS = 0.08
SMALL_SPHERE_MASS_KG = 0.50
TARGET_OPACITY = 0.30
RESET_Y_RANGE = 0.04
SUCCESS_THRESHOLD_M = 0.08
TARGET_LINE_TOLERANCE_M = 0.0
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6
# Initial palm world position in m. Drives a joint translation on floating
# Shadow/Leap and an arm IK target on UR10e (= floating_shadow base + 0.12).
ROBOT_INIT_PALM_WORLD_X = -0.63

SLOPE_LENGTH = 0.80
SLOPE_WIDTH = 0.90
SLOPE_THICKNESS = 0.04
SLOPE_ANGLE_DEG = 15.0
SLOPE_ANGLE_RAD = math.radians(SLOPE_ANGLE_DEG)
SLOPE_CENTER_X = 0.18
SLOPE_CENTER_Y = 0.0
SLOPE_HEIGHT_OFFSET = -0.04
GOAL_OFFSET_FROM_TOP_M = 0.12
START_OFFSET_FROM_SLOPE_M = 0.10
SIDE_GUARD_THICKNESS = 0.04
SIDE_GUARD_HEIGHT = 0.16
SIDE_GUARD_COLOR = (0.30, 0.30, 0.35)

MIN_NUM_OBSTACLES = 2
MAX_NUM_OBSTACLES = 5
ACTIVE_NUM_OBSTACLES = 2
OBSTACLE_LEG_SIZE = (0.12, 0.025, 0.07)
OBSTACLE_LEG_YAW_ANGLE_DEG = 38.0
OBSTACLE_LEG_YAW_ANGLE_RAD = math.radians(OBSTACLE_LEG_YAW_ANGLE_DEG)
GOAL_TANGENT = 0.5 * SLOPE_LENGTH - GOAL_OFFSET_FROM_TOP_M
# Keep the obstacle's uphill end below the target point along the slope, so
# obstacle randomization never places it above the goal in x/z.
OBSTACLE_UPHILL_CLEARANCE_FROM_GOAL = 0.10
OBSTACLE_MAX_APEX_TANGENT = (
    GOAL_TANGENT - OBSTACLE_LEG_SIZE[0] * math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD) - OBSTACLE_UPHILL_CLEARANCE_FROM_GOAL
)
OBSTACLE_APEX_TANGENT_RANGE = (-0.30, OBSTACLE_MAX_APEX_TANGENT)
OBSTACLE_APEX_TANGENT_JITTER = 0.005
OBSTACLE_CLEARANCE_M = SMALL_SPHERE_RADIUS + 0.03
OBSTACLE_APEX_MIN_TANGENT_SPACING = min(
    OBSTACLE_CLEARANCE_M,
    (OBSTACLE_APEX_TANGENT_RANGE[1] - OBSTACLE_APEX_TANGENT_RANGE[0]) / (MAX_NUM_OBSTACLES - 1)
    - 2.0 * OBSTACLE_APEX_TANGENT_JITTER,
)
OBSTACLE_LATERAL_HALF_EXTENT = (
    OBSTACLE_LEG_SIZE[0] * abs(math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD)) + 0.5 * OBSTACLE_LEG_SIZE[1]
)
_OBSTACLE_APEX_LATERAL_LIMIT_RAW = 0.5 * SLOPE_WIDTH - OBSTACLE_CLEARANCE_M - OBSTACLE_LATERAL_HALF_EXTENT
if _OBSTACLE_APEX_LATERAL_LIMIT_RAW < 0.0:
    raise ValueError("Slope is too narrow to keep obstacle clearance from the side guards.")
OBSTACLE_APEX_LATERAL_LIMIT = min(0.5, _OBSTACLE_APEX_LATERAL_LIMIT_RAW)
OBSTACLE_APEX_LATERAL_RANGE = (
    -OBSTACLE_APEX_LATERAL_LIMIT,
    OBSTACLE_APEX_LATERAL_LIMIT,
)
OBSTACLE_COLOR = (0.25, 0.25, 0.30)

SLOPE_TANGENT = (math.cos(SLOPE_ANGLE_RAD), 0.0, math.sin(SLOPE_ANGLE_RAD))
SLOPE_NORMAL = (-math.sin(SLOPE_ANGLE_RAD), 0.0, math.cos(SLOPE_ANGLE_RAD))
SLOPE_QUAT = (math.cos(SLOPE_ANGLE_RAD / 2.0), 0.0, -math.sin(SLOPE_ANGLE_RAD / 2.0), 0.0)


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


def _make_obstacle_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_LEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBSTACLE_COLOR, roughness=0.9),
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


SMALL_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.SphereCfg(
        radius=SMALL_SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=SMALL_SPHERE_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.18)),
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

SIDE_GUARD_LEFT_CFG = _make_side_guard_cfg("SideGuardLeft")
SIDE_GUARD_RIGHT_CFG = _make_side_guard_cfg("SideGuardRight")
OBSTACLE_CFG_PAIRS = tuple(
    (_make_obstacle_cfg(f"Obstacle{obstacle_idx}Left"), _make_obstacle_cfg(f"Obstacle{obstacle_idx}Right"))
    for obstacle_idx in range(MAX_NUM_OBSTACLES)
)


def object_passed_target_xz_line(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tolerance: float = TARGET_LINE_TOLERANCE_M,
) -> torch.Tensor:
    """Success once the ball has crossed the target's x and z coordinates."""
    obj = env.scene[object_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    obj_pos = obj.data.root_pos_w
    return (obj_pos[:, 0] >= target_pos[:, 0] - tolerance) & (obj_pos[:, 2] >= target_pos[:, 2] - tolerance)


@configclass
class PushSmallSphereObstacleSlopeCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for the obstacle-slope pushing task."""

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
class PushSmallSphereObstacleSlopeObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for push-sphere-with-obstacles on a slope.

    Sphere position / orientation and the (kinematic) obstacle states in
    ``proprio``; sphere velocities in ``privileged``. Commanded goal pose
    lives in ``goal``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        # Obstacle states are kinematic (velocities = 0) so we keep them as
        # the original body_state_b 13-vec rather than splitting per axis.
        obstacle_0_left_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_0_left"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_0_right_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_0_right"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_1_left_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_1_left"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_1_right_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_1_right"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_2_left_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_2_left"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_2_right_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_2_right"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_3_left_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_3_left"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_3_right_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_3_right"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_4_left_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_4_left"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        obstacle_4_right_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("obstacle_4_right"), "base_asset_cfg": SceneEntityCfg("table")},
        )
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
class PushSmallSphereObstacleSlopeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for the obstacle-slope pushing task."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 5.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=1.5,
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=5.0,
        params={
            "std": 0.14,
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
class PushSmallSphereObstacleSlopeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the obstacle-slope pushing task."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=object_passed_target_xz_line,
        params={
            "command_name": "object_pose",
            "tolerance": TARGET_LINE_TOLERANCE_M,
        },
    )


@configclass
class PushSmallSphereObstacleSlopeEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for the obstacle-slope task."""

    reset_random_inverted_v_obstacles = EventTerm(
        func=mdp.reset_random_inverted_v_obstacles,
        mode="reset",
        params={
            "obstacle_asset_cfgs": tuple(
                (
                    SceneEntityCfg(f"obstacle_{obstacle_idx}_left"),
                    SceneEntityCfg(f"obstacle_{obstacle_idx}_right"),
                )
                for obstacle_idx in range(MAX_NUM_OBSTACLES)
            ),
            "slope_center": (SLOPE_CENTER_X, SLOPE_CENTER_Y, 0.0),
            "slope_quat": SLOPE_QUAT,
            "active_count_range": (ACTIVE_NUM_OBSTACLES, ACTIVE_NUM_OBSTACLES),
            "apex_tangent_range": OBSTACLE_APEX_TANGENT_RANGE,
            "apex_tangent_jitter": OBSTACLE_APEX_TANGENT_JITTER,
            "min_apex_tangent_spacing": OBSTACLE_APEX_MIN_TANGENT_SPACING,
            "apex_lateral_range": OBSTACLE_APEX_LATERAL_RANGE,
            "apex_normal_offset": 0.5 * SLOPE_THICKNESS + 0.5 * OBSTACLE_LEG_SIZE[2],
            "leg_length": OBSTACLE_LEG_SIZE[0],
            "leg_yaw_angle_rad": OBSTACLE_LEG_YAW_ANGLE_RAD,
        },
    )


@configclass
class PushSmallSphereObstacleSlopeSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = SMALL_OBJECT_CFG
    slope: RigidObjectCfg = SLOPE_CFG
    side_guard_left: RigidObjectCfg = SIDE_GUARD_LEFT_CFG
    side_guard_right: RigidObjectCfg = SIDE_GUARD_RIGHT_CFG
    obstacle_0_left: RigidObjectCfg = OBSTACLE_CFG_PAIRS[0][0]
    obstacle_0_right: RigidObjectCfg = OBSTACLE_CFG_PAIRS[0][1]
    obstacle_1_left: RigidObjectCfg = OBSTACLE_CFG_PAIRS[1][0]
    obstacle_1_right: RigidObjectCfg = OBSTACLE_CFG_PAIRS[1][1]
    obstacle_2_left: RigidObjectCfg = OBSTACLE_CFG_PAIRS[2][0]
    obstacle_2_right: RigidObjectCfg = OBSTACLE_CFG_PAIRS[2][1]
    obstacle_3_left: RigidObjectCfg = OBSTACLE_CFG_PAIRS[3][0]
    obstacle_3_right: RigidObjectCfg = OBSTACLE_CFG_PAIRS[3][1]
    obstacle_4_left: RigidObjectCfg = OBSTACLE_CFG_PAIRS[4][0]
    obstacle_4_right: RigidObjectCfg = OBSTACLE_CFG_PAIRS[4][1]


@configclass
class PushSmallSphereObstacleSlopeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Push a smaller sphere uphill while navigating around randomized obstacles."""

    supports_object_pose_command: bool = True

    commands: PushSmallSphereObstacleSlopeCommandsCfg = PushSmallSphereObstacleSlopeCommandsCfg()
    observations: PushSmallSphereObstacleSlopeObservationsCfg = PushSmallSphereObstacleSlopeObservationsCfg()
    rewards: PushSmallSphereObstacleSlopeRewardsCfg = PushSmallSphereObstacleSlopeRewardsCfg()
    terminations: PushSmallSphereObstacleSlopeTerminationsCfg = PushSmallSphereObstacleSlopeTerminationsCfg()
    events: PushSmallSphereObstacleSlopeEventCfg = PushSmallSphereObstacleSlopeEventCfg()
    scene: PushSmallSphereObstacleSlopeSceneCfg = PushSmallSphereObstacleSlopeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=SMALL_OBJECT_CFG,
        slope=SLOPE_CFG,
        side_guard_left=SIDE_GUARD_LEFT_CFG,
        side_guard_right=SIDE_GUARD_RIGHT_CFG,
        obstacle_0_left=OBSTACLE_CFG_PAIRS[0][0],
        obstacle_0_right=OBSTACLE_CFG_PAIRS[0][1],
        obstacle_1_left=OBSTACLE_CFG_PAIRS[1][0],
        obstacle_1_right=OBSTACLE_CFG_PAIRS[1][1],
        obstacle_2_left=OBSTACLE_CFG_PAIRS[2][0],
        obstacle_2_right=OBSTACLE_CFG_PAIRS[2][1],
        obstacle_3_left=OBSTACLE_CFG_PAIRS[3][0],
        obstacle_3_right=OBSTACLE_CFG_PAIRS[3][1],
        obstacle_4_left=OBSTACLE_CFG_PAIRS[4][0],
        obstacle_4_right=OBSTACLE_CFG_PAIRS[4][1],
    )

    def __post_init__(self):
        super().__post_init__()

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
        self.events.reset_random_inverted_v_obstacles.params["slope_center"] = slope_center
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
            table_top_z + SMALL_SPHERE_RADIUS,
        )
        self.scene.object.init_state.pos = sphere_start
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        goal_surface_point = _surface_centerline_point(
            slope_center=slope_center,
            tangent_scale=0.5 * SLOPE_LENGTH - GOAL_OFFSET_FROM_TOP_M,
            normal_scale=0.5 * SLOPE_THICKNESS,
        )
        goal_center = (
            goal_surface_point[0] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[0],
            goal_surface_point[1] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[1],
            goal_surface_point[2] + SMALL_SPHERE_RADIUS * SLOPE_NORMAL[2],
        )
        self.commands.object_pose.ranges.pos_x = (goal_center[0], goal_center[0])
        self.commands.object_pose.ranges.pos_y = (goal_center[1], goal_center[1])
        self.commands.object_pose.ranges.pos_z = (goal_center[2], goal_center[2])

        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=SMALL_SPHERE_RADIUS,
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
                    radius=SMALL_SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.18)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=SMALL_SPHERE_RADIUS,
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

        obstacle_apex_tangents = [
            OBSTACLE_APEX_TANGENT_RANGE[0]
            + obstacle_idx * (OBSTACLE_APEX_TANGENT_RANGE[1] - OBSTACLE_APEX_TANGENT_RANGE[0]) / (MAX_NUM_OBSTACLES - 1)
            for obstacle_idx in range(MAX_NUM_OBSTACLES)
        ]
        left_dir_local = (
            math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD),
            math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD),
            0.0,
        )
        right_dir_local = (
            math.cos(-OBSTACLE_LEG_YAW_ANGLE_RAD),
            math.sin(-OBSTACLE_LEG_YAW_ANGLE_RAD),
            0.0,
        )
        left_obstacle_rot = (
            math.cos(SLOPE_ANGLE_RAD / 2.0) * math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            -math.sin(SLOPE_ANGLE_RAD / 2.0) * math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            -math.sin(SLOPE_ANGLE_RAD / 2.0) * math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            math.cos(SLOPE_ANGLE_RAD / 2.0) * math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
        )
        right_obstacle_rot = (
            math.cos(SLOPE_ANGLE_RAD / 2.0) * math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            math.sin(SLOPE_ANGLE_RAD / 2.0) * math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            -math.sin(SLOPE_ANGLE_RAD / 2.0) * math.cos(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
            -math.cos(SLOPE_ANGLE_RAD / 2.0) * math.sin(OBSTACLE_LEG_YAW_ANGLE_RAD / 2.0),
        )
        obstacle_normal_offset = 0.5 * SLOPE_THICKNESS + 0.5 * OBSTACLE_LEG_SIZE[2]
        for obstacle_idx, apex_tangent in enumerate(obstacle_apex_tangents):
            left_center_local = (
                apex_tangent + 0.5 * OBSTACLE_LEG_SIZE[0] * left_dir_local[0],
                0.5 * OBSTACLE_LEG_SIZE[0] * left_dir_local[1],
                obstacle_normal_offset,
            )
            right_center_local = (
                apex_tangent + 0.5 * OBSTACLE_LEG_SIZE[0] * right_dir_local[0],
                0.5 * OBSTACLE_LEG_SIZE[0] * right_dir_local[1],
                obstacle_normal_offset,
            )
            obstacle_left = getattr(self.scene, f"obstacle_{obstacle_idx}_left")
            obstacle_right = getattr(self.scene, f"obstacle_{obstacle_idx}_right")
            obstacle_left.init_state.pos = _surface_centerline_point(
                slope_center=slope_center,
                tangent_scale=left_center_local[0],
                normal_scale=left_center_local[2],
            )
            obstacle_left.init_state.pos = (
                obstacle_left.init_state.pos[0],
                slope_center[1] + left_center_local[1],
                obstacle_left.init_state.pos[2],
            )
            obstacle_right.init_state.pos = _surface_centerline_point(
                slope_center=slope_center,
                tangent_scale=right_center_local[0],
                normal_scale=right_center_local[2],
            )
            obstacle_right.init_state.pos = (
                obstacle_right.init_state.pos[0],
                slope_center[1] + right_center_local[1],
                obstacle_right.init_state.pos[2],
            )
            obstacle_left.init_state.rot = left_obstacle_rot
            obstacle_right.init_state.rot = right_obstacle_rot

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
class PushSmallSphereObstacleSlopeEnvFloatingDexHandRightCfg(PushSmallSphereObstacleSlopeEnvCfg):
    """Obstacle-slope pushing config for floating dex hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
