# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for placing a pen into a pen-holder.

Mirrors the ``insertpeg`` / ``plugcharger`` factory-insertion family, but uses
two ``synthesis`` assets:

* a **pen** (held object) that starts lying on the tabletop, and
* a **pen-holder** cup (fixed receptacle, kinematic) standing upright.

The robot must pick up the pen, bring it vertical, and drop it into the holder.
Success is the factory-style insertion test: the pen's lower tip is centered
over the holder opening (XY) and pushed below the rim (Z).

Geometry (from the authored USDs, metersPerUnit=1.0, Z-up):
* pen: ~0.185 m long along local +X, tube centerline at local z~0.0095, one tip
  at the local -X end (~-0.090 m); lies flat on the table.
* pen_holder001: ~0.120 m tall along local +Z, bottom at local z=0, opening at
  the top with an inner radius of ~0.038 m.
"""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
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
from .synthesis_insertion_assets import prepare_insertion_usd

# Collapse the multi-body synthesis USDs to a single rigid body. For the
# kinematic pen-holder we also re-author the colliders to an exact triangle mesh
# so the cup cavity stays hollow and the pen can actually be inserted.
PEN_USD_PATH = prepare_insertion_usd(SYNTHESIS_DIR / "pen" / "model_pen_0.usd")
PEN_HOLDER_USD_PATH = prepare_insertion_usd(
    SYNTHESIS_DIR / "pen_holder001" / "model_pen_holder001_0.usd",
    collision_approximation="none",
)

ASSET_SCALE = 1.0

# --- Authored geometry (meters, before scale) ---
PEN_LENGTH = 0.1851 * ASSET_SCALE
PEN_RADIUS = 0.0095 * ASSET_SCALE
PEN_CENTERLINE_Z = 0.0095 * ASSET_SCALE  # tube axis height in the local frame
PEN_HOLDER_HEIGHT = 0.1200 * ASSET_SCALE
PEN_HOLDER_OPENING_INNER_RADIUS = 0.038 * ASSET_SCALE
TABLE_CLEARANCE = 0.002

# Root offsets from the table top.
PEN_ROOT_OFFSET = PEN_RADIUS + TABLE_CLEARANCE  # lies flat on its side
PEN_HOLDER_ROOT_OFFSET = TABLE_CLEARANCE  # stands on its flat bottom (local z=0)

# Reference points in each asset's local frame.
# The pen runs along local +X (measured extent ~[-0.0907, +0.0944]); a tip sits
# at each end. Insertion counts as success when *either* end goes into the
# holder, so both tips are tracked (see the success / engaged terms below).
PEN_TIP_LOCAL_OFFSET = (-0.090 * ASSET_SCALE, 0.0, PEN_CENTERLINE_Z)  # -X end
PEN_TIP2_LOCAL_OFFSET = (0.0944 * ASSET_SCALE, 0.0, PEN_CENTERLINE_Z)  # +X end
PEN_HOLDER_OPENING_LOCAL_OFFSET = (0.0, 0.0, PEN_HOLDER_HEIGHT)  # opening rim center

# Engaged (shaping) tolerances: pen tip near/above the opening and roughly centered.
ENGAGE_CENTER_DIST_THRESH = PEN_HOLDER_OPENING_INNER_RADIUS  # ~0.038 m
ENGAGE_Z_THRESHOLD = 0.03  # tip within 3 cm above the rim counts as engaged

# Success tolerances: pen tip clearly inside the holder and pushed below the rim.
# XY tolerance set to ~the holder's inner radius: the cavity lets a seated pen's
# tip sit up to (inner_radius - pen_radius) ~= 0.0285 m off-center, and a long
# pen in this wide/shallow cup naturally leans toward the wall, so a tighter
# bound (the old 0.025) rejected pens that are genuinely inserted. The Z gate
# below (tip >= 3 cm under the rim) still prevents a pen merely resting on top
# from counting, so loosening XY alone is safe.
SUCCESS_CENTER_DIST_THRESH = 0.038
SUCCESS_Z_THRESHOLD = -0.03  # tip at least 3 cm below the rim

PEN_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pen",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=PEN_USD_PATH,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        # Collision is authored in the (collapsed) USD; keep convexDecomposition.
        collision_props=None,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

PEN_HOLDER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/PenHolder",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=PEN_HOLDER_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Keep the holder fixed during simulation (still resettable on
            # episode reset).
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
        # Exact triangle-mesh collision authored on the cleaned USD; do not
        # override it with a convex hull here.
        collision_props=None,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.10, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


@configclass
class InsertPenObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the insert-pen task.

    Two-part assembly: pen (held) and pen-holder (fixed). Absolute and relative
    body states live in ``privileged``; the goal group exposes the pen-holder
    opening position in the robot base frame.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        pen_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("pen"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        pen_holder_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("pen_holder"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        pen_state_rel_holder = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("pen"), "base_asset_cfg": SceneEntityCfg("pen_holder")},
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        """Insertion target: pen-holder position in the robot base frame."""

        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("pen_holder")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class InsertPenRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards for the insert-pen task."""

    engaged = RewTerm(
        func=mdp.factory_insert_engaged_reward,
        weight=2.0,
        params={
            "held_cfg": SceneEntityCfg("pen"),
            "fixed_cfg": SceneEntityCfg("pen_holder"),
            "held_base_local_offset": PEN_TIP_LOCAL_OFFSET,
            "held_base_local_offset_2": PEN_TIP2_LOCAL_OFFSET,
            "target_local_offset": PEN_HOLDER_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": ENGAGE_CENTER_DIST_THRESH,
            "z_threshold": ENGAGE_Z_THRESHOLD,
        },
    )

    success = RewTerm(
        func=mdp.factory_insert_success_reward,
        weight=10.0,
        params={
            "held_cfg": SceneEntityCfg("pen"),
            "fixed_cfg": SceneEntityCfg("pen_holder"),
            "held_base_local_offset": PEN_TIP_LOCAL_OFFSET,
            "held_base_local_offset_2": PEN_TIP2_LOCAL_OFFSET,
            "target_local_offset": PEN_HOLDER_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": SUCCESS_CENTER_DIST_THRESH,
            "z_threshold": SUCCESS_Z_THRESHOLD,
        },
    )


@configclass
class InsertPenTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the insert-pen task."""

    success = DoneTerm(
        func=mdp.factory_insert_success,
        params={
            "held_cfg": SceneEntityCfg("pen"),
            "fixed_cfg": SceneEntityCfg("pen_holder"),
            "held_base_local_offset": PEN_TIP_LOCAL_OFFSET,
            "held_base_local_offset_2": PEN_TIP2_LOCAL_OFFSET,
            "target_local_offset": PEN_HOLDER_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": SUCCESS_CENTER_DIST_THRESH,
            "z_threshold": SUCCESS_Z_THRESHOLD,
        },
    )


@configclass
class InsertPenEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for the insert-pen task."""

    reset_pen_holder = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("pen_holder"),
            # Tightened for an easier task: holder stays closer to its nominal
            # spot (world x 0.15-0.20, y +/-0.05) with little yaw, so the
            # insertion target is more predictable. Widen back to broaden the
            # distribution once the policy/teleop is reliable.
            "pose_range": {
                "x": [0.05, 0.10],
                "y": [-0.05, 0.05],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.radians(10.0), math.radians(10.0)],
            },
        },
    )

    reset_pen_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("pen"),
            # Tightened for an easier task: pen spawns in a smaller patch right
            # under the hand's start (world x -0.30..-0.20, y +/-0.10) with
            # +/-30deg yaw, so it is easier to grasp and orient. Widen back
            # (toward x [-0.2,-0.05], y [-0.2,0.2], yaw +/-60deg) to harden.
            "pose_range": {
                "x": [-0.15, -0.05],
                "y": [-0.10, 0.10],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi / 6.0, math.pi / 6.0],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class InsertPenSceneCfg(dexverse_base_env.SceneCfg):
    pen: RigidObjectCfg = PEN_CFG
    pen_holder: RigidObjectCfg = PEN_HOLDER_CFG
    object = None


@configclass
class InsertPenEnvCfg(ContactRichEnvCfg):
    """Insert-pen task configuration (base, robot-agnostic)."""

    observations: InsertPenObservationsCfg = InsertPenObservationsCfg()
    rewards: InsertPenRewardsCfg = InsertPenRewardsCfg()
    terminations: InsertPenTerminationsCfg = InsertPenTerminationsCfg()
    events: InsertPenEventCfg = InsertPenEventCfg()
    scene: InsertPenSceneCfg = InsertPenSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        pen=PEN_CFG,
        pen_holder=PEN_HOLDER_CFG,
    )

    contact_object_prim: str = "Pen"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        self.scene.pen_holder.init_state.pos = (0.10, 0.0, table_top_z + PEN_HOLDER_ROOT_OFFSET)
        self.scene.pen.init_state.pos = (-0.15, 0.0, table_top_z + PEN_ROOT_OFFSET)

        self.episode_length_s = 25.0


@configclass
class InsertPenEnvFloatingDexHandRightCfg(InsertPenEnvCfg):
    """Insert-pen environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
