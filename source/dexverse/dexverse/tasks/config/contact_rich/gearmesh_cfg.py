# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for gear-mesh task with tabletop manipulation."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import ContactRichEnvCfg

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
GEAR_BASE_USD_PATH = f"{ASSET_DIR}/factory_gear_base.usd"
MEDIUM_GEAR_USD_PATH = f"{ASSET_DIR}/factory_gear_medium.usd"
SMALL_GEAR_USD_PATH = f"{ASSET_DIR}/factory_gear_small.usd"
LARGE_GEAR_USD_PATH = f"{ASSET_DIR}/factory_gear_large.usd"
ASSET_SCALE = 3.0

GEAR_BASE_HEIGHT = 0.02 * ASSET_SCALE
MEDIUM_GEAR_HEIGHT = 0.03 * ASSET_SCALE

# Root-placement offsets relative to the tabletop. Keep these separate from
# task geometry heights, which are used by success logic.
TABLE_CLEARANCE = 0.001
GEAR_BASE_ROOT_OFFSET = TABLE_CLEARANCE
MEDIUM_GEAR_ROOT_OFFSET = TABLE_CLEARANCE - 0.005 * ASSET_SCALE

MEDIUM_GEAR_BASE_OFFSET_X = 0.02025 * ASSET_SCALE
MEDIUM_GEAR_BASE_OFFSET_Z = 0.0

SUCCESS_THRESHOLD_FRAC = 0.05
CENTER_DIST_THRESH = 0.0025

GEAR_BASE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/GearBase",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GEAR_BASE_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1.0e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)

MEDIUM_GEAR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/MediumGear",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MEDIUM_GEAR_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1.0e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.2, 0.65),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)

SMALL_GEAR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/SmallGear",
    spawn=sim_utils.UsdFileCfg(
        usd_path=SMALL_GEAR_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1.0e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)

LARGE_GEAR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/LargeGear",
    spawn=sim_utils.UsdFileCfg(
        usd_path=LARGE_GEAR_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1.0e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)


@configclass
class GearMeshObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the gear-mesh task. See ``InsertPegObservationsCfg`` for rationale."""

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        medium_gear_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("medium_gear"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        gear_base_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("gear_base"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        medium_gear_state_rel_base = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("medium_gear"), "base_asset_cfg": SceneEntityCfg("gear_base")},
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        """Meshing target: gear-base (rotation pivot) position in robot base frame."""

        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("gear_base")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class GearMeshRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards disabled for teleoperation-first setup."""

    action_l2 = None
    action_rate_l2 = None


@configclass
class GearMeshTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for gear-mesh task."""

    success = DoneTerm(
        func=mdp.factory_insert_success,
        params={
            "held_cfg": SceneEntityCfg("medium_gear"),
            "fixed_cfg": SceneEntityCfg("gear_base"),
            "held_base_local_offset": (MEDIUM_GEAR_BASE_OFFSET_X, 0.0, MEDIUM_GEAR_BASE_OFFSET_Z),
            "target_local_offset": (MEDIUM_GEAR_BASE_OFFSET_X, 0.0, MEDIUM_GEAR_BASE_OFFSET_Z),
            "center_dist_thresh": CENTER_DIST_THRESH,
            "z_threshold": GEAR_BASE_HEIGHT * SUCCESS_THRESHOLD_FRAC,
        },
    )
    medium_gear_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("medium_gear"),
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.3, 1.4)},
        },
    )


@configclass
class GearMeshEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for gear-mesh task."""

    reset_gear_base = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("gear_base"),
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, math.radians(15.0)],
            },
        },
    )

    reset_small_gear_to_base = EventTerm(
        func=mdp.update_articulation_root_from_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("small_gear"),
            "source_cfg": SceneEntityCfg("gear_base"),
        },
    )

    reset_large_gear_to_base = EventTerm(
        func=mdp.update_articulation_root_from_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("large_gear"),
            "source_cfg": SceneEntityCfg("gear_base"),
        },
    )

    reset_medium_gear_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("medium_gear"),
            "pose_range": {
                "x": [-0.15, 0.15],
                "y": [-0.15, 0.02],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi, math.pi],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class GearMeshSceneCfg(dexverse_base_env.SceneCfg):
    gear_base: ArticulationCfg = GEAR_BASE_CFG
    medium_gear: ArticulationCfg = MEDIUM_GEAR_CFG
    small_gear: ArticulationCfg = SMALL_GEAR_CFG
    large_gear: ArticulationCfg = LARGE_GEAR_CFG
    object = None


@configclass
class GearMeshEnvCfg(ContactRichEnvCfg):
    """Gear-mesh task configuration (base, robot-agnostic)."""

    observations: GearMeshObservationsCfg = GearMeshObservationsCfg()
    rewards: GearMeshRewardsCfg = GearMeshRewardsCfg()
    terminations: GearMeshTerminationsCfg = GearMeshTerminationsCfg()
    events: GearMeshEventCfg = GearMeshEventCfg()
    scene: GearMeshSceneCfg = GearMeshSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        gear_base=GEAR_BASE_CFG,
        medium_gear=MEDIUM_GEAR_CFG,
        small_gear=SMALL_GEAR_CFG,
        large_gear=LARGE_GEAR_CFG,
    )

    contact_object_prim: str = "MediumGear"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        gear_base_z = table_top_z + GEAR_BASE_ROOT_OFFSET
        medium_gear_z = table_top_z + MEDIUM_GEAR_ROOT_OFFSET

        self.scene.gear_base.init_state.pos = (0.0, 0.0, gear_base_z)
        self.scene.medium_gear.init_state.pos = (0.0, -0.2, medium_gear_z)
        self.scene.small_gear.init_state.pos = self.scene.gear_base.init_state.pos
        self.scene.large_gear.init_state.pos = self.scene.gear_base.init_state.pos

        self.episode_length_s = 20.0


@configclass
class GearMeshEnvFloatingDexHandRightCfg(GearMeshEnvCfg):
    """Gear-mesh environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
