# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for nut-thread task with tabletop manipulation."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
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
BOLT_USD_PATH = f"{ASSET_DIR}/factory_bolt_m16.usd"
NUT_USD_PATH = f"{ASSET_DIR}/factory_nut_m16.usd"
ASSET_SCALE = 3.0
NUT_ASSET_SCALE = 3.05

BOLT_HEAD_HEIGHT = 0.01 * ASSET_SCALE
BOLT_SHANK_HEIGHT = 0.025 * ASSET_SCALE
THREAD_PITCH = 0.002 * ASSET_SCALE
NUT_BASE_HEIGHT = 0.01 * NUT_ASSET_SCALE
TARGET_THREAD_TURNS = 1.5

# Root-placement offsets relative to the tabletop. Keep these separate from
# bolt/nut geometry heights, which are used by success logic.
TABLE_CLEARANCE = 0.001
BOLT_ROOT_OFFSET = TABLE_CLEARANCE
NUT_ROOT_OFFSET = TABLE_CLEARANCE - NUT_BASE_HEIGHT
NUT_TABLE_CLEARANCE = NUT_ROOT_OFFSET  # Backwards-compatible alias; this is a root offset.

SUCCESS_THRESHOLD_TURNS = 0.375
CENTER_DIST_THRESH = 0.0025

BOLT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Bolt",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BOLT_USD_PATH,
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
        pos=(0.0, 0.0, 0.61),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)

NUT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Nut",
    spawn=sim_utils.UsdFileCfg(
        usd_path=NUT_USD_PATH,
        scale=(NUT_ASSET_SCALE, NUT_ASSET_SCALE, NUT_ASSET_SCALE),
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)


@configclass
class NutThreadObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the nut-thread task. See ``InsertPegObservationsCfg`` for rationale."""

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        nut_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("nut"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        bolt_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("bolt"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        nut_state_rel_bolt = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("nut"), "base_asset_cfg": SceneEntityCfg("bolt")},
        )

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class NutThreadRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards disabled for teleoperation-first setup."""

    action_l2 = None
    action_rate_l2 = None


@configclass
class NutThreadTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for nut-thread task."""

    success = DoneTerm(
        func=mdp.nutthread_success,
        params={
            "success_threshold_turns": SUCCESS_THRESHOLD_TURNS,
            "center_dist_thresh": CENTER_DIST_THRESH,
            "thread_pitch": THREAD_PITCH,
            "nut_cfg": SceneEntityCfg("nut"),
            "bolt_cfg": SceneEntityCfg("bolt"),
            "nut_base_height": NUT_BASE_HEIGHT,
            "bolt_head_height": BOLT_HEAD_HEIGHT,
            "bolt_shank_height": BOLT_SHANK_HEIGHT,
            "target_thread_turns": TARGET_THREAD_TURNS,
        },
    )
    nut_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("nut"),
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.3, 1.4)},
        },
    )


@configclass
class NutThreadEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for nut-thread task."""

    reset_bolt = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bolt"),
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [math.radians(105.0), math.radians(135.0)],
            },
        },
    )

    reset_nut_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("nut"),
            "pose_range": {
                "x": [-0.15, 0.15],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi, math.pi],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class NutThreadSceneCfg(dexverse_base_env.SceneCfg):
    bolt: ArticulationCfg = BOLT_CFG
    nut: ArticulationCfg = NUT_CFG
    object = None


@configclass
class NutThreadEnvCfg(ContactRichEnvCfg):
    """Nut-thread task configuration (base, robot-agnostic)."""

    observations: NutThreadObservationsCfg = NutThreadObservationsCfg()
    rewards: NutThreadRewardsCfg = NutThreadRewardsCfg()
    terminations: NutThreadTerminationsCfg = NutThreadTerminationsCfg()
    events: NutThreadEventCfg = NutThreadEventCfg()
    scene: NutThreadSceneCfg = NutThreadSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        bolt=BOLT_CFG,
        nut=NUT_CFG,
    )

    contact_object_prim: str = "Nut"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5
        bolt_z = table_top_z + BOLT_ROOT_OFFSET
        nut_z = table_top_z + NUT_ROOT_OFFSET
        self.scene.bolt.init_state.pos = (0.0, 0.0, bolt_z)
        self.scene.nut.init_state.pos = (0.0, 0.0, nut_z)

        self.episode_length_s = 20.0


@configclass
class NutThreadEnvFloatingDexHandRightCfg(NutThreadEnvCfg):
    """Nut-thread environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
