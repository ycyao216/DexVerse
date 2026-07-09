# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for picking up a primitive cube on a tabletop."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import PickupObjectObservationsCfg

CUBE_SIZE = 0.05
CUBE_HALF_SIZE = CUBE_SIZE * 0.5
CUBE_MASS_KG = 0.08
SUCCESS_MIN_HEIGHT_M = 0.20
CENTER_SQUARE_SIZE = 0.35
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5

SUCCESS_MARKER_SIZE = (0.01, 0.01, 0.12)
SUCCESS_MARKER_COLOR = (0.1, 0.9, 0.1)
SUCCESS_MARKER_QUAT = (1.0, 0.0, 0.0, 0.0)


CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.45, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
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
class PickCubeObservationsCfg(PickupObjectObservationsCfg):
    """Observation layout for pick-cube.

    Inherits the pickup-object split unchanged: object pose in ``state``
    (observable, no velocities), object linear / angular velocities in
    ``privileged``. No task-specific terms.
    """


@configclass
class PickCubeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pick-cube."""

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
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": SUCCESS_MIN_HEIGHT_M},
    )


@configclass
class PickCubeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for pick-cube."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    success = DoneTerm(
        func=mdp.object_lifted,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": SUCCESS_MIN_HEIGHT_M},
    )


@configclass
class PickCubeEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for pick-cube."""

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
class PickCubeSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = CUBE_CFG
    success_marker: RigidObjectCfg = SUCCESS_MARKER_CFG


@configclass
class PickCubeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pick-cube task configuration (base, robot-agnostic)."""

    observations: PickCubeObservationsCfg = PickCubeObservationsCfg()
    rewards: PickCubeRewardsCfg = PickCubeRewardsCfg()
    terminations: PickCubeTerminationsCfg = PickCubeTerminationsCfg()
    events: PickCubeEventCfg = PickCubeEventCfg()
    scene: PickCubeSceneCfg = PickCubeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=CUBE_CFG,
        success_marker=SUCCESS_MARKER_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.events.object_physics_material = None

        table_size = self.scene.table.spawn.size
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        object_pos = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = (object_pos[0], object_pos[1], table_top_z + CUBE_HALF_SIZE)
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.success_marker.init_state.pos = (
            object_pos[0],
            object_pos[1],
            table_top_z + CUBE_HALF_SIZE + SUCCESS_MIN_HEIGHT_M,
        )
        self.scene.success_marker.init_state.rot = SUCCESS_MARKER_QUAT

        half_side = CENTER_SQUARE_SIZE * 0.5
        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-half_side, half_side],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-3.14, 3.14],
            }

        if self.terminations.object_out_of_bound is not None:
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
class PickCubeEnvFloatingDexHandRightCfg(PickCubeEnvCfg):
    """Pick-cube config for floating dexterous hands (Shadow / Leap)."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
