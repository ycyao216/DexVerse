# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pour-wineglass task with tabletop manipulation."""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop

WINE_GLASS_USD_PATH = str(DEXVERSE_AUTHORED_ASSETS_DIR / "wine_glass" / "wine_glass.usd")

CENTER_SQUARE_SIZE = 0.45
WINE_GLASS_Z_OFFSET = 0.02
WINE_GLASS_SCALE = (1.0, 1.0, 1.0)
SUCCESS_MIN_HEIGHT_M = 0.2
SUCCESS_MARKER_SIZE = (0.01, 0.01, 0.16)
SUCCESS_MARKER_COLOR = (0.1, 0.9, 0.1)
WINE_GLASS_INIT_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)
POUR_ANGLE_RAD = math.radians(100.0)
SUCCESS_MARKER_QUAT = (
    math.cos(POUR_ANGLE_RAD * 0.5),
    -math.sin(POUR_ANGLE_RAD * 0.5),
    0.0,
    0.0,
)  # rotate +Z toward +Y in the YZ plane by POUR_ANGLE_RAD

WINE_GLASS_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=WINE_GLASS_USD_PATH,
        scale=WINE_GLASS_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        # Collision is authored on the Mesh inside the USD (convexDecomposition).
        # Applying CollisionAPI to the root Xform here makes PhysX fall back to
        # triangle-mesh collision, which is illegal on dynamic bodies.
        collision_props=None,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=WINE_GLASS_INIT_QUAT),
)

SUCCESS_MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SuccessMarker",
    spawn=sim_utils.CuboidCfg(
        size=SUCCESS_MARKER_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=SUCCESS_MARKER_COLOR,
            emissive_color=(0.0, 0.3, 0.0),
            roughness=1.0,
            metallic=0.0,
        ),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SUCCESS_MARKER_QUAT),
)


@configclass
class PourWineGlassObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the dexterous pour-wineglass task.

    Object position / up-axis / tilt in ``proprio``; velocities in
    ``privileged``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_tilt_angle = ObsTerm(func=mdp.object_tilt_angle, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class PourWineGlassRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pour-wineglass task."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
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

    tilt_reward = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=5.0,
        params={
            "threshold_rad": POUR_ANGLE_RAD,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PourWineGlassTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for pour-wineglass task."""

    success = DoneTerm(
        func=mdp.lift_and_tilt,
        params={
            "min_height": SUCCESS_MIN_HEIGHT_M,
            "threshold_rad": POUR_ANGLE_RAD,
            "axis_local": (0.0, 1.0, 0.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": True,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PourWineGlassEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for pour-wineglass task."""

    reset_success_marker = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("success_marker"),
            "source_cfg": SceneEntityCfg("object"),
            "z_offset": SUCCESS_MIN_HEIGHT_M,
            "quat": SUCCESS_MARKER_QUAT,
        },
    )


@configclass
class PourWineGlassCommandsCfg(dexverse_base_env.CommandsCfg):

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(10.0, 10.0),
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
        success_vis_asset_name="table",
        position_only=True,
    )


@configclass
class PourWineGlassEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pour-wineglass task configuration (base, robot-agnostic)."""

    observations: PourWineGlassObservationsCfg = PourWineGlassObservationsCfg()
    rewards: PourWineGlassRewardsCfg = PourWineGlassRewardsCfg()
    terminations: PourWineGlassTerminationsCfg = PourWineGlassTerminationsCfg()
    commands: PourWineGlassCommandsCfg = PourWineGlassCommandsCfg()
    events: PourWineGlassEventCfg = PourWineGlassEventCfg()

    @configclass
    class PourWineGlassSceneCfg(dexverse_base_env.SceneCfg):
        object: RigidObjectCfg = WINE_GLASS_CFG
        success_marker: RigidObjectCfg = SUCCESS_MARKER_CFG

    scene: PourWineGlassSceneCfg = PourWineGlassSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=WINE_GLASS_CFG,
        success_marker=SUCCESS_MARKER_CFG,
    )
    supports_object_pose_command: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Keep a single episode target time; no command-driven targets used here.
        self.episode_length_s = 20.0

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        object_z = table_top_z + WINE_GLASS_Z_OFFSET

        # Randomize wineglass reset on the tabletop (center square).
        if self.events.reset_object is not None:
            half_side = CENTER_SQUARE_SIZE * 0.5
            # Ensure initial wineglass z sits on the tabletop.
            object_pos = self.scene.object.init_state.pos
            self.scene.object.init_state.pos = (
                object_pos[0],
                object_pos[1],
                object_z,
            )
            if hasattr(self.commands, "object_pose"):
                self.commands.object_pose.use_world_frame = True
                self.commands.object_pose.position_only = True
                self.commands.object_pose.ranges.pos_x = (object_pos[0], object_pos[0])
                self.commands.object_pose.ranges.pos_y = (object_pos[1], object_pos[1])
                self.commands.object_pose.ranges.pos_z = (object_z, object_z)
                self.commands.object_pose.ranges.roll = (0.0, 0.0)
                self.commands.object_pose.ranges.pitch = (0.0, 0.0)
                self.commands.object_pose.ranges.yaw = (0.0, 0.0)
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-half_side, half_side],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }

        # Place success marker at target height, pointing +Z toward +Y in the YZ plane.
        target_z = object_z + SUCCESS_MIN_HEIGHT_M
        wineglass_pos = self.scene.object.init_state.pos
        self.scene.success_marker.init_state.pos = (wineglass_pos[0], wineglass_pos[1], target_z)
        self.scene.success_marker.init_state.rot = SUCCESS_MARKER_QUAT

        # Setup contact sensors if enabled
        mdp.setup_fingertip_contact_observation(self)

        # Override reward body names with robot-specific values
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


# Unified robot configuration (supports all robot types via robot_type argument)
@configclass
class PourWineGlassEnvFloatingDexHandRightCfg(PourWineGlassEnvCfg):
    """Pour-wineglass environment configuration for Floating DexHand (supports Shadow and Leap).

    Robot configuration is handled by the base class registry. This config only needs to:
    1. Set the default robot_type
    2. Configure teleoperation devices if needed
    """

    # Set default robot_type for this config
    robot_type: str = "floating_shadow_right"
    # XR configuration (needed for teleop)
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        """Post initialization."""
        # Call parent __post_init__ which will configure the robot based on robot_type
        super().__post_init__()
        setup_floating_teleop(self)
