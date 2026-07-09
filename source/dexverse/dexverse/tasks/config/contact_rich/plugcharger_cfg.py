# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for plug-charger insertion on tabletop manipulation."""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR
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
from .base_cfg import ContactRichEnvCfg

PLUG_CHARGER_ASSET_DIR = DEXVERSE_AUTHORED_ASSETS_DIR / "plug_charger"
PEG_USD_PATH = str((PLUG_CHARGER_ASSET_DIR / "charger" / "charger.usd").resolve())
RECEPTACLE_USD_PATH = str((PLUG_CHARGER_ASSET_DIR / "receptacle" / "receptacle.usd").resolve())

ASSET_SCALE = 2.5

# Asset-root offsets from table top.
# Receptacle's source geometry has ~0.05 m half-height in local z (before scale), so
# place root at half-height above table plus a tiny clearance to avoid table penetration.
RECEPTACLE_HALF_HEIGHT = 0.05 * ASSET_SCALE
TABLE_CLEARANCE = 0.002
RECEPTACLE_ROOT_OFFSET = RECEPTACLE_HALF_HEIGHT + TABLE_CLEARANCE
PEG_ROOT_OFFSET = 0.014 * ASSET_SCALE

ENGAGE_THRESHOLD_FRAC = 0.9
CENTER_DIST_THRESH = 0.005
# Peg tip reference point in charger local frame (asset-source tip at x=0.016 before scale).
SUCCESS_PEG_TIP_LOCAL_OFFSET = (0.016 * ASSET_SCALE, 0.0, 0.0)
# Success when peg-tip x in receptacle frame is greater than -0.02 m.
SUCCESS_INSERTION_X_THRESH = -0.02
# Y/Z alignment tolerances are calibrated from slot-vs-peg geometry after scaling:
# slot half-clearance margin is about 3.125 mm, so use 2.5 mm as robust in-contact threshold.
SUCCESS_INSERTION_Y_THRESH = 0.0025
SUCCESS_INSERTION_Z_THRESH = 0.0025

RECEPTACLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Receptacle",
    spawn=sim_utils.UsdFileCfg(
        usd_path=RECEPTACLE_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Keep receptacle root fixed in simulation (still resettable at episode reset).
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.10, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

CHARGER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Charger",
    spawn=sim_utils.UsdFileCfg(
        usd_path=PEG_USD_PATH,
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
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.62),
        rot=(0.70710678, 0.70710678, 0.0, 0.0),
    ),
)


@configclass
class PlugChargerObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the plug-charger insertion task.

    Two-part assembly (charger + receptacle), states in ``proprio``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        charger_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("charger"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        receptacle_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("receptacle"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        charger_state_rel_receptacle = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("charger"), "base_asset_cfg": SceneEntityCfg("receptacle")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class GoalObsCfg(ObsGroup):
        """Insertion target: receptacle position in the robot base frame."""

        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("receptacle")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class PlugChargerRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards for plug-charger insertion task."""

    engaged = RewTerm(
        func=mdp.factory_insert_engaged_reward,
        weight=2.0,
        params={
            "held_cfg": SceneEntityCfg("charger"),
            "fixed_cfg": SceneEntityCfg("receptacle"),
            "held_base_local_offset": (0.0, 0.0, 0.0),
            "target_local_offset": (0.0, 0.0, 0.0),
            "center_dist_thresh": CENTER_DIST_THRESH,
            "z_threshold": PEG_ROOT_OFFSET * ENGAGE_THRESHOLD_FRAC,
        },
    )

    success = RewTerm(
        func=mdp.plug_charger_pose_success_reward,
        weight=10.0,
        params={
            "held_cfg": SceneEntityCfg("charger"),
            "fixed_cfg": SceneEntityCfg("receptacle"),
            "held_local_offset": SUCCESS_PEG_TIP_LOCAL_OFFSET,
            "insertion_x_threshold": SUCCESS_INSERTION_X_THRESH,
            "insertion_y_threshold": SUCCESS_INSERTION_Y_THRESH,
            "insertion_z_threshold": SUCCESS_INSERTION_Z_THRESH,
        },
    )


@configclass
class PlugChargerTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for plug-charger task."""

    success = DoneTerm(
        func=mdp.plug_charger_pose_success,
        params={
            "held_cfg": SceneEntityCfg("charger"),
            "fixed_cfg": SceneEntityCfg("receptacle"),
            "held_local_offset": SUCCESS_PEG_TIP_LOCAL_OFFSET,
            "insertion_x_threshold": SUCCESS_INSERTION_X_THRESH,
            "insertion_y_threshold": SUCCESS_INSERTION_Y_THRESH,
            "insertion_z_threshold": SUCCESS_INSERTION_Z_THRESH,
        },
    )


@configclass
class PlugChargerEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for plug-charger task."""

    reset_receptacle = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("receptacle"),
            "pose_range": {
                "x": [0.05, 0.15],
                "y": [-0.08, 0.08],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.radians(15.0), math.radians(15.0)],
            },
        },
    )

    reset_charger_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("charger"),
            "pose_range": {
                "x": [-0.2, -0.05],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [-math.pi / 3.0, math.pi / 3.0],
                "yaw": [0.0, 0.0],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class PlugChargerSceneCfg(dexverse_base_env.SceneCfg):
    charger: RigidObjectCfg = CHARGER_CFG
    receptacle: RigidObjectCfg = RECEPTACLE_CFG
    object = None


@configclass
class PlugChargerEnvCfg(ContactRichEnvCfg):
    """Plug-charger task configuration (base, robot-agnostic)."""

    observations: PlugChargerObservationsCfg = PlugChargerObservationsCfg()
    rewards: PlugChargerRewardsCfg = PlugChargerRewardsCfg()
    terminations: PlugChargerTerminationsCfg = PlugChargerTerminationsCfg()
    events: PlugChargerEventCfg = PlugChargerEventCfg()
    scene: PlugChargerSceneCfg = PlugChargerSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        charger=CHARGER_CFG,
        receptacle=RECEPTACLE_CFG,
    )

    contact_object_prim: str = "Charger"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        self.scene.receptacle.init_state.pos = (0.10, 0.0, table_top_z + RECEPTACLE_ROOT_OFFSET)
        self.scene.charger.init_state.pos = (-0.15, 0.0, table_top_z + PEG_ROOT_OFFSET)

        self.episode_length_s = 20.0


@configclass
class PlugChargerEnvFloatingDexHandRightCfg(PlugChargerEnvCfg):
    """Plug-charger environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
