# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pick-up-stick task with tabletop manipulation."""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import CORE_ASSETS_DIR
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

ASSET_DIR = CORE_ASSETS_DIR / "dexverse_authored" / "Stick"
STICK_USD_PATH = str(ASSET_DIR / "stick.usda")

SUCCESS_MIN_HEIGHT_M = 0.2
STICK_HEIGHT_OFFSET = 0.05
MAX_TILT_RAD = math.radians(30.0)

SUCCESS_MARKER_SIZE = (0.01, 0.01, 0.16)
SUCCESS_MARKER_COLOR = (0.1, 0.9, 0.1)
SUCCESS_MARKER_QUAT = (1.0, 0.0, 0.0, 0.0)

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5

HORIZONTAL_QUAT = (0.5, -0.5, 0.5, 0.5)  # rotate +Z to +Y (lying flat, yawed 90deg)


STICK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=STICK_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=HORIZONTAL_QUAT),
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
class PickUpStickObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for pick-up-stick.

    ``state`` (observable, no velocities): stick position + up-axis + tilt angle
    (all derivable from a real-world pose estimate; the up-axis + tilt stand in
    for a full quaternion, which is over-determined for a rotationally-symmetric
    stick). ``privileged``: linear & angular velocity.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_tilt_angle = ObsTerm(func=mdp.object_tilt_angle, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class PickUpStickRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pick-up-stick task."""

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

    lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": SUCCESS_MIN_HEIGHT_M,
        },
    )


@configclass
class PickUpStickTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for pick-up-stick task."""

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
            "threshold_rad": MAX_TILT_RAD,
            "axis_local": (0.0, 0.0, 1.0),
            "tilt_ge": False,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickUpStickEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for pick-up-stick task."""

    object_scale_mass = None

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
class PickUpStickEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pick-up-stick task configuration (base, robot-agnostic)."""

    observations: PickUpStickObservationsCfg = PickUpStickObservationsCfg()
    rewards: PickUpStickRewardsCfg = PickUpStickRewardsCfg()
    terminations: PickUpStickTerminationsCfg = PickUpStickTerminationsCfg()
    events: PickUpStickEventCfg = PickUpStickEventCfg()

    @configclass
    class PickUpStickSceneCfg(dexverse_base_env.SceneCfg):
        object: RigidObjectCfg = STICK_CFG
        success_marker: RigidObjectCfg = SUCCESS_MARKER_CFG

    scene: PickUpStickSceneCfg = PickUpStickSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=STICK_CFG,
        success_marker=SUCCESS_MARKER_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep a single episode target time; no command-driven targets used here.
        self.episode_length_s = 20.0

        # Stick USD has no physics material; skip material randomization.
        self.events.object_physics_material = None

        # Setup fingertip contact sensors for object interaction. Each fingertip link gets a
        # dedicated ContactSensor that filters to the stick; we then expose the aggregated
        # contact force observation for the policy. When sensors are disabled, we explicitly
        # remove the contact observation term to keep the observation space consistent.
        mdp.setup_fingertip_contact_observation(self)


# Unified robot configuration (supports all robot types via robot_type argument)
@configclass
class PickUpStickEnvFloatingDexHandRightCfg(PickUpStickEnvCfg):
    """Pick-up-stick environment configuration for Floating DexHand (supports Shadow and Leap).

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

        # Override reward body names with robot-specific values.
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names

        # Place stick on tabletop and keep it horizontal.
        table_pos = self.scene.table.init_state.pos
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        target_z = table_top_z + STICK_HEIGHT_OFFSET
        stick_pos = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = (stick_pos[0], stick_pos[1], target_z)
        self.scene.object.init_state.rot = HORIZONTAL_QUAT

        # Place success marker at target height, oriented upright.
        target_z = table_top_z + SUCCESS_MIN_HEIGHT_M
        self.scene.success_marker.init_state.pos = (stick_pos[0], stick_pos[1], target_z)
        self.scene.success_marker.init_state.rot = SUCCESS_MARKER_QUAT

        # Randomize stick reset on the tabletop (center square).
        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-0.1, 0.1],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }
