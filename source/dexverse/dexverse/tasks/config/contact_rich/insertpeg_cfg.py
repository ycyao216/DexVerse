# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for ManiSkill-style side Insert Peg on tabletop manipulation."""

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

INSERT_PEG_ASSET_DIR = DEXVERSE_AUTHORED_ASSETS_DIR / "insert_peg"
PEG_USD_PATH = str(INSERT_PEG_ASSET_DIR / "peg" / "peg.usd")
HOLE_USD_PATH = str(INSERT_PEG_ASSET_DIR / "hole" / "hole.usd")

ASSET_SCALE = 1.0

PEG_HALF_LENGTH = 0.10 * ASSET_SCALE
PEG_RADIUS = 0.02 * ASSET_SCALE
HOLE_OUTER_RADIUS = 0.10 * ASSET_SCALE
HOLE_INNER_RADIUS = 0.023 * ASSET_SCALE
TABLE_CLEARANCE = 0.002

PEG_ROOT_OFFSET = PEG_RADIUS + TABLE_CLEARANCE
HOLE_ROOT_OFFSET = HOLE_OUTER_RADIUS + TABLE_CLEARANCE

SUCCESS_PEG_HEAD_LOCAL_OFFSET = (PEG_HALF_LENGTH, 0.0, 0.0)
SUCCESS_HOLE_CENTER_LOCAL_OFFSET = (0.0, 0.0, 0.0)
SUCCESS_HOLE_RADIUS = HOLE_INNER_RADIUS
SUCCESS_INSERTION_X_THRESHOLD = 0.015 * ASSET_SCALE

PEG_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Peg",
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

HOLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Hole",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HOLE_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Keep hole fixed during simulation while still resettable on episode reset.
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.10, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


@configclass
class InsertPegObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the insert-peg task.

    Two-part assembly: peg and hole. Both absolute and relative body states
    live in ``proprio`` (each is a 13-vec ``body_state_b`` — peg/hole velocities
    are dense features the policy consumes as a single composite signal, so
    we do not split them across proprio / privileged).
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        peg_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("peg"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        hole_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("hole"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        peg_state_rel_hole = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("peg"), "base_asset_cfg": SceneEntityCfg("hole")},
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        """Insertion target: hole position in the robot base frame."""

        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("hole")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class InsertPegRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards for Insert Peg task."""

    engaged = RewTerm(
        func=mdp.factory_insert_engaged_reward,
        weight=2.0,
        params={
            "held_cfg": SceneEntityCfg("peg"),
            "fixed_cfg": SceneEntityCfg("hole"),
            "held_base_local_offset": SUCCESS_PEG_HEAD_LOCAL_OFFSET,
            "target_local_offset": SUCCESS_HOLE_CENTER_LOCAL_OFFSET,
            "center_dist_thresh": SUCCESS_HOLE_RADIUS,
            "z_threshold": PEG_HALF_LENGTH * 0.5,
        },
    )

    success = RewTerm(
        func=mdp.insert_peg_success_reward,
        weight=10.0,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "hole_cfg": SceneEntityCfg("hole"),
            "peg_head_local_offset": SUCCESS_PEG_HEAD_LOCAL_OFFSET,
            "hole_center_local_offset": SUCCESS_HOLE_CENTER_LOCAL_OFFSET,
            "hole_radius": SUCCESS_HOLE_RADIUS,
            "insertion_x_threshold": SUCCESS_INSERTION_X_THRESHOLD,
        },
    )


@configclass
class InsertPegTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for Insert Peg task."""

    success = DoneTerm(
        func=mdp.insert_peg_success,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "hole_cfg": SceneEntityCfg("hole"),
            "peg_head_local_offset": SUCCESS_PEG_HEAD_LOCAL_OFFSET,
            "hole_center_local_offset": SUCCESS_HOLE_CENTER_LOCAL_OFFSET,
            "hole_radius": SUCCESS_HOLE_RADIUS,
            "insertion_x_threshold": SUCCESS_INSERTION_X_THRESHOLD,
        },
    )


@configclass
class InsertPegEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for Insert Peg task."""

    reset_hole = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hole"),
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

    reset_peg_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "pose_range": {
                "x": [-0.2, -0.05],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi / 3.0, math.pi / 3.0],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class InsertPegSceneCfg(dexverse_base_env.SceneCfg):
    peg: RigidObjectCfg = PEG_CFG
    hole: RigidObjectCfg = HOLE_CFG
    object = None


@configclass
class InsertPegEnvCfg(ContactRichEnvCfg):
    """Insert Peg task configuration (base, robot-agnostic)."""

    observations: InsertPegObservationsCfg = InsertPegObservationsCfg()
    rewards: InsertPegRewardsCfg = InsertPegRewardsCfg()
    terminations: InsertPegTerminationsCfg = InsertPegTerminationsCfg()
    events: InsertPegEventCfg = InsertPegEventCfg()
    scene: InsertPegSceneCfg = InsertPegSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        peg=PEG_CFG,
        hole=HOLE_CFG,
    )

    contact_object_prim: str = "Peg"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        self.scene.hole.init_state.pos = (0.10, 0.0, table_top_z + HOLE_ROOT_OFFSET)
        self.scene.peg.init_state.pos = (-0.15, 0.0, table_top_z + PEG_ROOT_OFFSET)

        self.episode_length_s = 20.0


@configclass
class InsertPegEnvFloatingDexHandRightCfg(InsertPegEnvCfg):
    """Insert-peg environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
