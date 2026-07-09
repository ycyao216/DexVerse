# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for picking a thin object out of a small open-top container."""

from __future__ import annotations

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

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6
TARGET_OPACITY = 0.30
SHOW_OBJECT_GOAL_MARKER = False
SHOW_OBJECT_FRAME_VIS = False

OBJECT_SIZE_M = (0.018, 0.085, 0.12)
OBJECT_HALF_SIZE_M = tuple(0.5 * dim for dim in OBJECT_SIZE_M)
OBJECT_MASS_KG = 0.06
OBJECT_TABLE_CLEARANCE_M = 0.004
OBJECT_BACK_CLEARANCE_M = 0.012
OBJECT_SIDE_CLEARANCE_M = 0.004
OBJECT_SIDE_OFFSET_Y_M = 0.0

CONTAINER_FRONT_X_M = 0.16
CONTAINER_INNER_DEPTH_M = 0.12
CONTAINER_INNER_WIDTH_M = 0.12
CONTAINER_WALL_THICKNESS_M = 0.02
CONTAINER_WALL_HEIGHT_M = 0.11

SUCCESS_EXTRACTION_CLEARANCE_M = 0.035
SUCCESS_LIFT_M = CONTAINER_WALL_HEIGHT_M - 0.5 * OBJECT_SIZE_M[2] + SUCCESS_EXTRACTION_CLEARANCE_M
SUCCESS_POSITION_THRESHOLD_M = 0.04
GOAL_FORWARD_OFFSET_M = 0.10
GOAL_LIFT_EXTRA_M = 0.04

# Initial palm world position in m. Drives joint translations on floating
# Shadow/Leap and an arm IK target on UR10e
# (= floating_shadow base (-0.75, 0, 0.5) + (0.54, 0, 0.34)).
ROBOT_INIT_PALM_WORLD_X = -0.21
ROBOT_INIT_PALM_WORLD_Z = 0.84

CONTACT_STATIC_FRICTION = 1.6
CONTACT_DYNAMIC_FRICTION = 1.4
CONTACT_RESTITUTION = 0.0
CONTACT_FRICTION_COMBINE_MODE = "max"


def high_friction_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=CONTACT_STATIC_FRICTION,
        dynamic_friction=CONTACT_DYNAMIC_FRICTION,
        restitution=CONTACT_RESTITUTION,
        friction_combine_mode=CONTACT_FRICTION_COMBINE_MODE,
    )


def _make_panel_cfg(
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float] = (0.40, 0.34, 0.28),
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


OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=OBJECT_SIZE_M,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=high_friction_material_cfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.68, 0.24), roughness=0.75),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)

CONTAINER_FRONT_CFG = _make_panel_cfg(
    "ContainerFront",
    (CONTAINER_WALL_THICKNESS_M, CONTAINER_INNER_WIDTH_M + 2 * CONTAINER_WALL_THICKNESS_M, CONTAINER_WALL_HEIGHT_M),
)
CONTAINER_BACK_CFG = _make_panel_cfg(
    "ContainerBack",
    (CONTAINER_WALL_THICKNESS_M, CONTAINER_INNER_WIDTH_M + 2 * CONTAINER_WALL_THICKNESS_M, CONTAINER_WALL_HEIGHT_M),
)
CONTAINER_LEFT_CFG = _make_panel_cfg(
    "ContainerLeft",
    (CONTAINER_INNER_DEPTH_M, CONTAINER_WALL_THICKNESS_M, CONTAINER_WALL_HEIGHT_M),
)
CONTAINER_RIGHT_CFG = _make_panel_cfg(
    "ContainerRight",
    (CONTAINER_INNER_DEPTH_M, CONTAINER_WALL_THICKNESS_M, CONTAINER_WALL_HEIGHT_M),
)


@configclass
class PickThinObjectFromContainerCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for the small-container extraction task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="object",
        resampling_time_range=(12.0, 12.0),
        debug_vis=SHOW_OBJECT_GOAL_MARKER,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        position_only=True,
    )


@configclass
class PickThinObjectFromContainerObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for extracting a thin object from a small container.

    Object position / orientation / up-axis in ``proprio``; velocities in
    ``privileged``. Commanded target pose lives in ``goal``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=-0.0, n_max=0.0))
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
class PickThinObjectFromContainerRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for extracting the thin object out of the container."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.30,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.05,
        },
    )

    lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": SUCCESS_LIFT_M,
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=4.0,
        params={
            "std": 0.08,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10.0,
        params={
            "command_name": "object_pose",
            "pos_std": SUCCESS_POSITION_THRESHOLD_M,
            "rot_std": None,
        },
    )


@configclass
class PickThinObjectFromContainerTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the small-container extraction task."""

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
            "threshold": SUCCESS_POSITION_THRESHOLD_M,
        },
    )


@configclass
class PickThinObjectFromContainerSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = OBJECT_CFG
    container_front: RigidObjectCfg = CONTAINER_FRONT_CFG
    container_back: RigidObjectCfg = CONTAINER_BACK_CFG
    container_left: RigidObjectCfg = CONTAINER_LEFT_CFG
    container_right: RigidObjectCfg = CONTAINER_RIGHT_CFG


@configclass
class PickThinObjectFromContainerEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pick a thin cuboid out of a small open-top container by using wall contact."""

    supports_object_pose_command: bool = True

    commands: PickThinObjectFromContainerCommandsCfg = PickThinObjectFromContainerCommandsCfg()
    observations: PickThinObjectFromContainerObservationsCfg = PickThinObjectFromContainerObservationsCfg()
    rewards: PickThinObjectFromContainerRewardsCfg = PickThinObjectFromContainerRewardsCfg()
    terminations: PickThinObjectFromContainerTerminationsCfg = PickThinObjectFromContainerTerminationsCfg()
    scene: PickThinObjectFromContainerSceneCfg = PickThinObjectFromContainerSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
        container_front=CONTAINER_FRONT_CFG,
        container_back=CONTAINER_BACK_CFG,
        container_left=CONTAINER_LEFT_CFG,
        container_right=CONTAINER_RIGHT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        set_robot_wrist_init_world_pos(self, x=ROBOT_INIT_PALM_WORLD_X, z=ROBOT_INIT_PALM_WORLD_Z)

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        wall_center_z = table_top_z + 0.5 * CONTAINER_WALL_HEIGHT_M
        # Compute object center from mesh half-size and add a small table clearance
        # to avoid initial overlap with the tabletop contact surface.
        object_center_z = table_top_z + OBJECT_HALF_SIZE_M[2] + OBJECT_TABLE_CLEARANCE_M

        back_wall_center_x = (
            CONTAINER_FRONT_X_M
            + CONTAINER_WALL_THICKNESS_M
            + CONTAINER_INNER_DEPTH_M
            + 0.5 * CONTAINER_WALL_THICKNESS_M
        )
        front_wall_center_x = CONTAINER_FRONT_X_M + 0.5 * CONTAINER_WALL_THICKNESS_M
        side_wall_center_x = CONTAINER_FRONT_X_M + CONTAINER_WALL_THICKNESS_M + 0.5 * CONTAINER_INNER_DEPTH_M
        side_wall_center_y = 0.5 * (CONTAINER_INNER_WIDTH_M + CONTAINER_WALL_THICKNESS_M)

        # Absolute container-inner XY limits for the thin cuboid center.
        # x-range: [front-inner-face + half_obj_x, back-inner-face - half_obj_x]
        # y-range: [-inner_half_width + half_obj_y, inner_half_width - half_obj_y]
        inner_x_min = CONTAINER_FRONT_X_M + CONTAINER_WALL_THICKNESS_M + OBJECT_HALF_SIZE_M[0]
        inner_x_max = CONTAINER_FRONT_X_M + CONTAINER_WALL_THICKNESS_M + CONTAINER_INNER_DEPTH_M - OBJECT_HALF_SIZE_M[0]
        inner_y_min = -0.5 * CONTAINER_INNER_WIDTH_M + OBJECT_HALF_SIZE_M[1] + OBJECT_SIDE_CLEARANCE_M
        inner_y_max = 0.5 * CONTAINER_INNER_WIDTH_M - OBJECT_HALF_SIZE_M[1] - OBJECT_SIDE_CLEARANCE_M

        # Keep object fully inside the container footprint and away from back wall.
        object_center_x_nominal = inner_x_max - OBJECT_BACK_CLEARANCE_M
        object_center_x = min(inner_x_max, max(inner_x_min, object_center_x_nominal))
        object_center_y = min(inner_y_max, max(inner_y_min, OBJECT_SIDE_OFFSET_Y_M))

        self.scene.object.init_state.pos = (object_center_x, object_center_y, object_center_z)
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.container_front.init_state.pos = (front_wall_center_x, 0.0, wall_center_z)
        self.scene.container_back.init_state.pos = (back_wall_center_x, 0.0, wall_center_z)
        self.scene.container_left.init_state.pos = (side_wall_center_x, side_wall_center_y, wall_center_z)
        self.scene.container_right.init_state.pos = (side_wall_center_x, -side_wall_center_y, wall_center_z)

        goal_x = CONTAINER_FRONT_X_M - GOAL_FORWARD_OFFSET_M
        goal_z = object_center_z + SUCCESS_LIFT_M + GOAL_LIFT_EXTRA_M
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.position_only = True
        self.commands.object_pose.use_world_frame = True
        self.commands.object_pose.ranges.pos_x = (goal_x, goal_x)
        self.commands.object_pose.ranges.pos_y = (object_center_y, object_center_y)
        self.commands.object_pose.ranges.pos_z = (goal_z, goal_z)
        if SHOW_OBJECT_GOAL_MARKER:
            self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/goal_pose",
                markers={
                    "target": sim_utils.CuboidCfg(
                        size=OBJECT_SIZE_M,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.6, 0.9),
                            opacity=TARGET_OPACITY,
                        ),
                    )
                },
            )

        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
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
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class PickThinObjectFromContainerEnvFloatingDexHandRightCfg(PickThinObjectFromContainerEnvCfg):
    """Small-container extraction config for floating dex hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
