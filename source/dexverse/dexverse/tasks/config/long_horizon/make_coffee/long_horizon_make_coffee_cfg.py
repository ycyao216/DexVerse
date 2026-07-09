# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon task config: make a coffee.

Task intent
~~~~~~~~~~~
A mug starts seated upright on a wooden cup holder (mug tree); a milk bottle and
a (fixed-base) espresso machine also sit on the table. Grasping the mug off the
holder is an implicit prerequisite the policy must perform, but only the
coffee-making sub-sequence drives the success criteria::

    1. pour_milk          -- tilt the milk bottle over the mug (spout reaches it)
    2. place_on_machine   -- set the mug under the coffee machine's group head
    3. rotate_knob        -- turn the machine's switch lever (start "brewing")

Success is driven by a strict, persistent :class:`mdp.StageGraphSpec`; the
terminal stage is ``rotate_knob``.

Asset notes
~~~~~~~~~~~
- Mug (``dexverse_authored/.../mug/SM_Mug_A2.usd``) is the canonical
  ``scene.object`` (dynamic, graspable). Its USD bakes a ``scale=0.01`` on its
  root, but Isaac Lab's spawn ``scale`` *overrides* that (it is authored on the
  referencing prim), so the effective world size is the raw mesh extent (~9.1)
  times the spawn scale: ``MUG_SCALE`` ~= 0.011 gives a ~10 cm mug. Its convex
  collider is kept so it rests on the holder / table / machine.
- Cup holder (``synthesis/cup_holder/model_woodenstand3.usd``) is a kinematic
  mug tree: a flat base plate (top at local z=0.012) + a central post + angled
  pegs up high. The mug rests UPRIGHT on the base plate, offset clear of the
  post, and is re-synced relative to the (re-randomized) holder each reset via
  :func:`mdp.sync_object` -- the remove-cup-from-rack mechanism. The holder
  authors a nested rigid body, so :func:`ensure_single_rigid_body` collapses it
  to a single (spawn-applied kinematic) body, keeping the convex colliders.
- Milk bottle (``synthesis/drink095/model_drink095.usd``) is a second dynamic
  graspable rigid object (``scene.milk``, ~20 cm tall). It authors its rigid
  body on a child link, so :func:`ensure_single_rigid_body` collapses it too.
  "Pour" = tilt it past ``POUR_TILT_RAD`` about its up-axis with its top spout
  over the mug (see :func:`milk_poured_over_mug`), mirroring grasp_kettle_cfg.
- Coffee machine (``synthesis/coffee_machine001/model_coffee_machine_6.usd``) is
  a clean single-articulation-root asset (root body ``E_body_3``). Its front
  (steam wand / portafilter / switch / buttons) is local -Y; we yaw it -90 deg
  about +Z so the front faces the robot at -X. The switch lever
  (``RevoluteJoint_coffee_machine_6_up``, about Y, +/-90 deg) is the "knob" the
  task rotates. All joints are driven with a friction-detent
  :class:`ImplicitActuatorCfg` (``stiffness=0`` -> no setpoint; static friction +
  ``armature`` hold each joint where the hand leaves it). PartNet-style, it
  authors no link mass, so :func:`prepare_coffee_machine_usd` gives the control
  links a small real mass (see ``usd_prep.py``).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR, SYNTHESIS_DIR
from dexverse.tasks.config.bimanual.usd_helpers import ensure_single_rigid_body
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from scipy.spatial.transform import Rotation as R

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ....mdp.utils import axis_tilt_angle
from ...floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .usd_prep import prepare_coffee_machine_usd

# =====================================================================
# USD asset paths (holder + milk collapsed to one rigid body; machine mass-prepped)
# =====================================================================

MUG_USD_PATH = str(DEXVERSE_AUTHORED_ASSETS_DIR / "mug" / "SM_Mug_A2.usd")
# The holder / milk author a nested rigid body; collapse so the spawn-applied
# body is the only one (the convex colliders are kept).
CUP_HOLDER_USD_PATH = ensure_single_rigid_body(str(SYNTHESIS_DIR / "cup_holder" / "model_woodenstand3.usd"))
MILK_USD_PATH = ensure_single_rigid_body(str(SYNTHESIS_DIR / "drink095" / "model_drink095.usd"))
COFFEE_MACHINE_USD_PATH = prepare_coffee_machine_usd(SYNTHESIS_DIR / "coffee_machine001" / "model_coffee_machine_6.usd")


# =====================================================================
# Layout (env-local; table centred at origin, top at DEFAULT_TABLE_TOP_HEIGHT)
# Robot reaches +X; objects span roughly x in [-0.1, 0.45], y in [-0.35, 0.35].
# =====================================================================

TABLE_TOP_Z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT  # 0.6

# ---- Mug (scene.object): ~10 cm; starts hung TILTED on a holder pin. ---------
MUG_SCALE = (0.009, 0.009, 0.009)  # TUNE: raw extent ~9.1 * scale -> ~10 cm
MUG_MASS = 0.3
MUG_HALF_HEIGHT = 0.05  # ~ half of (9.1 * scale); tied to MUG_SCALE
MUG_UPRIGHT_ROT = (1.0, 0.0, 0.0, 0.0)
# The mug's WORLD orientation when seated on the holder (with the holder at its
# init rotation): tilted -105 deg about +X so it hangs on an angled pin, exactly
# like the cup in remove_cup_from_rack_cfg.
MUG_ROT_ON_HOLDER = tuple(R.from_euler("x", -105, degrees=True).as_quat(scalar_first=True))

# ---- Cup holder (kinematic mug tree), enlarged. -----------------------------
# remove_cup_from_rack uses scale 1.5 with the same woodenstand3; we go larger.
CUP_HOLDER_SCALE_FACTOR = 1.6  # TUNE: ~0.47 m tall at 1.6
CUP_HOLDER_SCALE = (CUP_HOLDER_SCALE_FACTOR,) * 3
# -90 deg yaw about +Z (matches remove_cup_from_rack), so the pin offset below is
# expressed in the same holder-local frame it tuned.
CUP_HOLDER_INIT_ROT = tuple(R.from_euler("z", -90, degrees=True).as_quat(scalar_first=True))
CUP_HOLDER_CENTER_X = -0.05
CUP_HOLDER_CENTER_Y = 0.28
# Holder origin sits at its base (geometry z starts at 0), so seat it on the table.
CUP_HOLDER_HALF_HEIGHT = 0.0
# Where the mug hangs on a pin, in the holder's local frame (metres). Ported from
# remove_cup_from_rack's (0.16, -0.01, 0.255) at holder scale 1.5 and scaled to
# our holder size (the pin positions scale with the holder spawn scale). z lands
# it on the +x pin; x reaches just past the pin tip. TUNE after previewing.
_REMOVE_CUP_OFFSET_AT_1p5 = (0.16, -0.01, 0.255)
MUG_OFFSET_ON_HOLDER = tuple(v * CUP_HOLDER_SCALE_FACTOR / 1.5 for v in _REMOVE_CUP_OFFSET_AT_1p5)
# Mug-on-holder quat expressed in the holder's local frame, so sync_object's
# ``R(holder_world) * quat_local`` recomposes MUG_ROT_ON_HOLDER at init rotation
# AND tracks the holder's reset yaw (same trick as remove_cup_from_rack).
_holder_init_R = R.from_quat(CUP_HOLDER_INIT_ROT, scalar_first=True)
_mug_world_R = R.from_quat(MUG_ROT_ON_HOLDER, scalar_first=True)
MUG_ROT_LOCAL_ON_HOLDER = tuple(float(v) for v in (_holder_init_R.inv() * _mug_world_R).as_quat(scalar_first=True))
# Holder reset jitter (offsets from its seated init pose).
CUP_HOLDER_RESET_X_RANGE = (-0.04, 0.06)
CUP_HOLDER_RESET_Y_RANGE = (-0.06, 0.06)
CUP_HOLDER_RESET_YAW_RANGE = (-0.3, 0.3)

# ---- Milk bottle (scene.milk): dynamic, graspable; spawns on the table. ------
MILK_SCALE = (1.0, 1.0, 1.0)  # ~20 cm tall at 1.0
MILK_MASS = 0.5
MILK_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
MILK_CENTER_X = -0.05
MILK_CENTER_Y = -0.28
MILK_SPAWN_CLEARANCE = 0.01  # spawn a touch above the table; settles in
MILK_RESET_X_RANGE = (-0.05, 0.10)
MILK_RESET_Y_RANGE = (-0.08, 0.08)
MILK_RESET_YAW_RANGE = (-0.4, 0.4)

# ---- Coffee machine (fixed-base articulation): back-centre, front faces -X. ---
# Enlarged from 1.0 (~0.30 x 0.35 x 0.29 m) to 1.4 (~0.41 x 0.49 x 0.41 m).
COFFEE_SCALE_FACTOR = 1.8
COFFEE_SCALE = (COFFEE_SCALE_FACTOR,) * 3
COFFEE_CENTER_X = 0.42
COFFEE_CENTER_Y = -0.05
# Root (E_body_3) origin sits 0.042 (unscaled) above the body's lowest geometry,
# which scales with the spawn scale; lift the root so the base rests on the table.
COFFEE_BASE_BOTTOM_LOCAL_Z = -0.042 * COFFEE_SCALE_FACTOR
COFFEE_BASE_CLEARANCE = 0.003
COFFEE_Z = TABLE_TOP_Z - COFFEE_BASE_BOTTOM_LOCAL_Z + COFFEE_BASE_CLEARANCE
# -90 deg yaw about +Z: local -Y front -> world -X (toward the robot).
COFFEE_INIT_ROT = (math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4))
COFFEE_BODY = "E_body_3"  # articulation root body

# Coffee-machine joints (verified from the USD).
COFFEE_KNOB_JOINT = "RevoluteJoint_coffee_machine_6_up"  # switch lever, Y, +/-90 deg
COFFEE_OTHER_JOINTS = [
    "RevoluteJoint_coffee_machine_6_right",  # steam wand (Z)
    "RevoluteJoint_coffee_machine_6_middle",  # portafilter (Z)
    "PrismaticJoint_coffee_machine_6_left",  # button 1 (Y)
    "PrismaticJoint_coffee_machine_6_middle",  # button 2 (Y)
    "PrismaticJoint_coffee_machine_6_right",  # button 3 (Y)
]

# Friction-detent drive. TUNE COFFEE_KNOB_*_FRICTION if the lever sags under
# gravity (its axis is horizontal Y) or the hand cannot turn it.
COFFEE_KNOB_STATIC_FRICTION = 0.15
COFFEE_KNOB_DYNAMIC_FRICTION = 0.10
COFFEE_KNOB_DAMPING = 0.5
COFFEE_KNOB_ARMATURE = 0.005
COFFEE_OTHER_STATIC_FRICTION = 0.5
COFFEE_OTHER_DYNAMIC_FRICTION = 0.3
COFFEE_OTHER_DAMPING = 1.0
COFFEE_OTHER_ARMATURE = 0.005

# Mug placement spot ON the machine, in the machine root (E_body_3) local frame:
# under the group head / portafilter outlet, in front of the body face. The body
# front face is at local y~-0.138 and the portafilter sticks out to y~-0.209, so a
# mug placed a bit further forward (more -Y) rests on the table in front of the
# body, under the spout. The y reach scales with the machine spawn scale (the
# geometry scales with it). The mug's origin is at its BASE, so its resting root z
# is the table top; the local z is chosen so the tracked target sits there (the yaw
# about +Z does not change the world z of the offset). TUNE after previewing.
COFFEE_PLACE_LOCAL = (
    0.0,
    -0.08 * COFFEE_SCALE_FACTOR,
    TABLE_TOP_Z - COFFEE_Z + 0.1,
)


# =====================================================================
# Goal locations (table place goal; offsets from the table centre)
# =====================================================================

# "place_on_table" goal: open front-centre area, clear of holder / milk / machine.
# The mug's origin is at its base, so the resting root z is the table top.
TABLE_GOAL_X = -0.08
TABLE_GOAL_Y = 0.00
TABLE_GOAL_Z = TABLE_TOP_Z


# =====================================================================
# Stage-graph thresholds
# =====================================================================

GRASP_MUG_LIFT_MIN = 0.01  # mug lifted >= 1 cm above its holder height
PLACE_TABLE_POS_TOL = 0.06  # mug within 6 cm of the table goal ...
PLACE_TABLE_MAX_TILT = math.radians(20.0)  # ... and roughly upright
POUR_TILT_RAD = math.radians(70.0)  # milk tilted >= 70 deg from upright
PLACE_MACHINE_XY_TOL = 0.09  # mug within this horizontal dist of the spot
PLACE_MACHINE_Z_TOL = 0.07  # ... and within this vertical band of it
KNOB_ROTATE_DISP = math.radians(35.0)  # switch lever turned >= 35 deg from rest

# Pour proximity is a pair of spheres (cx, cy, cz, radius) in each asset's LOCAL
# frame; the pour "reaches" when their world centres are within (sum of radii).
#   * MUG_GOAL_ZONE  : the pour goal, on top of the mug (rim ~ local z 0.10).
#   * MILK_SPOUT_ZONE: the point to reach with it, at the milk bottle's lip.
MUG_GOAL_ZONE = (0.0, 0.0, 0.10, 0.04)
MILK_SPOUT_ZONE = (0.0, 0.0, 0.18, 0.03)

LONG_HORIZON_STAGE_GRAPH_KEY = "long_horizon.make_coffee"

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6


# =====================================================================
# Custom predicates
# =====================================================================


def milk_poured_over_mug(
    env,
    tilt_rad: float = POUR_TILT_RAD,
    spout_zone: tuple[float, float, float, float] = MILK_SPOUT_ZONE,
    goal_zone: tuple[float, float, float, float] = MUG_GOAL_ZONE,
) -> torch.Tensor:
    """True per-env when the milk bottle is tilted with its spout over the mug.

    Mirrors the kettle pour: the bottle's up-axis (local +Z) must tilt past
    ``tilt_rad`` from world +Z, AND the milk spout sphere (``spout_zone`` on the
    milk) must overlap the pour-goal sphere (``goal_zone`` on top of the mug) --
    i.e. their world centres are within (spout radius + goal radius).
    """
    milk = env.scene["milk"]
    mug = env.scene["object"]
    milk_quat = milk.data.root_quat_w
    mug_quat = mug.data.root_quat_w
    tilt = axis_tilt_angle(milk_quat, axis_local=(0.0, 0.0, 1.0), world_axis=(0.0, 0.0, 1.0))
    tilted = tilt >= tilt_rad

    sx, sy, sz, sr = spout_zone
    gx, gy, gz, gr = goal_zone
    spout_off = torch.tensor((sx, sy, sz), device=env.device, dtype=milk_quat.dtype)
    goal_off = torch.tensor((gx, gy, gz), device=env.device, dtype=mug_quat.dtype)
    spout_w = milk.data.root_pos_w + quat_apply(milk_quat, spout_off.unsqueeze(0).expand(env.num_envs, -1))
    goal_w = mug.data.root_pos_w + quat_apply(mug_quat, goal_off.unsqueeze(0).expand(env.num_envs, -1))
    reached = torch.norm(spout_w - goal_w, dim=-1) <= (sr + gr)
    return tilted & reached


def mug_on_coffee_maker(
    env,
    xy_tol: float = PLACE_MACHINE_XY_TOL,
    z_tol: float = PLACE_MACHINE_Z_TOL,
    place_local: tuple[float, float, float] = COFFEE_PLACE_LOCAL,
) -> torch.Tensor:
    """True per-env when the mug rests at the machine's group-head spot.

    The spot is a fixed point in the machine root body's frame, tracked live (so
    it follows any machine pose jitter): the mug must be horizontally within
    ``xy_tol`` and vertically within ``z_tol`` of it.
    """
    machine = env.scene["coffee_machine"]
    mug = env.scene["object"]
    body_ids, _ = machine.find_bodies(COFFEE_BODY)
    body_pos = machine.data.body_pos_w[:, body_ids[0], :]
    body_quat = machine.data.body_quat_w[:, body_ids[0], :]
    offset = torch.tensor(place_local, device=env.device, dtype=body_quat.dtype)
    target_w = body_pos + quat_apply(body_quat, offset.unsqueeze(0).expand(env.num_envs, -1))
    mug_pos = mug.data.root_pos_w
    horiz = torch.norm(mug_pos[:, :2] - target_w[:, :2], dim=-1) < xy_tol
    vert = torch.abs(mug_pos[:, 2] - target_w[:, 2]) < z_tol
    return horiz & vert


def stage_progress_reward(env, task_key: str, persistent: bool = True) -> torch.Tensor:
    """Dense long-horizon shaping: the per-env stage-completion ratio in [0, 1]."""
    return mdp.stage_progress(env, task_key=task_key, persistent=persistent).squeeze(-1)


# =====================================================================
# Stage graph: strict, persistent -- one sub-stage per step of the task
# =====================================================================

MAKE_COFFEE_STAGE_GRAPH = mdp.StageGraphSpec(
    stages=(
        mdp.StageSpec(
            name="pour_milk",
            func=milk_poured_over_mug,
            params={
                "tilt_rad": POUR_TILT_RAD,
                "spout_zone": MILK_SPOUT_ZONE,
                "goal_zone": MUG_GOAL_ZONE,
            },
        ),
        mdp.StageSpec(
            name="place_on_machine",
            func=mug_on_coffee_maker,
            params={
                "xy_tol": PLACE_MACHINE_XY_TOL,
                "z_tol": PLACE_MACHINE_Z_TOL,
                "place_local": COFFEE_PLACE_LOCAL,
            },
            deps=("pour_milk",),
        ),
        mdp.StageSpec(
            name="rotate_knob",
            func=mdp.joint_relative_move,
            params={
                "threshold": KNOB_ROTATE_DISP,
                "asset_cfg": SceneEntityCfg("coffee_machine", joint_names=[COFFEE_KNOB_JOINT]),
                "mode": "displacement",
                "op": ">=",
                "reduce": "any",
            },
            deps=("place_on_machine",),
        ),
    ),
    terminal_stage="rotate_knob",
    ordering_mode="strict",
    success_mode="substage",
)
mdp.register_stage_graph(LONG_HORIZON_STAGE_GRAPH_KEY, MAKE_COFFEE_STAGE_GRAPH, override=True)


# =====================================================================
# Asset configs
# =====================================================================


def _build_mug_cfg() -> RigidObjectCfg:
    """Mug -- the canonical ``scene.object`` (dynamic, graspable)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=MUG_USD_PATH,
            scale=MUG_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            # collision_props left None: keep the authored convex collider.
            mass_props=sim_utils.MassPropertiesCfg(mass=MUG_MASS),
        ),
        # Spawn at the mug's hung pose on the holder pin. This is the
        # ``default_root_state`` that ``object_lifted`` measures the lift against,
        # so its z must match where ``reset_mug_on_holder`` (sync_object) seats it
        # (the pin height = CUP_HOLDER_CENTER z + offset z; yaw does not change z).
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                CUP_HOLDER_CENTER_X + MUG_OFFSET_ON_HOLDER[0],
                CUP_HOLDER_CENTER_Y + MUG_OFFSET_ON_HOLDER[1],
                TABLE_TOP_Z + CUP_HOLDER_HALF_HEIGHT + MUG_OFFSET_ON_HOLDER[2],
            ),
            rot=MUG_ROT_ON_HOLDER,
        ),
    )


def _build_cup_holder_cfg() -> RigidObjectCfg:
    """Kinematic cup-holder (mug tree). Re-randomized each reset."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupHolder",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=CUP_HOLDER_USD_PATH,
            scale=CUP_HOLDER_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
            ),
            collision_props=None,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CUP_HOLDER_CENTER_X, CUP_HOLDER_CENTER_Y, TABLE_TOP_Z),
            rot=CUP_HOLDER_INIT_ROT,
        ),
    )


def _build_milk_cfg() -> RigidObjectCfg:
    """Milk bottle -- a second dynamic, graspable rigid object (``scene.milk``)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Milk",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=MILK_USD_PATH,
            scale=MILK_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            collision_props=None,
            mass_props=sim_utils.MassPropertiesCfg(mass=MILK_MASS),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(MILK_CENTER_X, MILK_CENTER_Y, TABLE_TOP_Z + MILK_SPAWN_CLEARANCE),
            rot=MILK_INIT_ROT,
        ),
    )


def _build_coffee_machine_cfg() -> ArticulationCfg:
    """Fixed-base coffee machine with friction-detent manipulable joints."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/CoffeeMachine",
        spawn=sim_utils.UsdFileCfg(
            usd_path=COFFEE_MACHINE_USD_PATH,
            scale=COFFEE_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            # Free baseline drive; the friction-detent actuators below own the joints.
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=0.0, stiffness=0.0, damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(COFFEE_CENTER_X, COFFEE_CENTER_Y, COFFEE_Z),
            rot=COFFEE_INIT_ROT,
            joint_pos={COFFEE_KNOB_JOINT: 0.0, **{j: 0.0 for j in COFFEE_OTHER_JOINTS}},
        ),
        actuators={
            # The switch lever (the rotated "knob"): low friction + armature so a
            # fingertip can turn it, but enough to hold it (its axis is
            # horizontal, so gravity exerts a torque about it).
            "coffee_knob": ImplicitActuatorCfg(
                joint_names_expr=[COFFEE_KNOB_JOINT],
                effort_limit_sim=50.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=COFFEE_KNOB_DAMPING,
                friction=COFFEE_KNOB_STATIC_FRICTION,
                dynamic_friction=COFFEE_KNOB_DYNAMIC_FRICTION,
                armature=COFFEE_KNOB_ARMATURE,
            ),
            # Everything else: hold where left (not part of the task).
            "coffee_other": ImplicitActuatorCfg(
                joint_names_expr=COFFEE_OTHER_JOINTS,
                effort_limit_sim=80.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=COFFEE_OTHER_DAMPING,
                friction=COFFEE_OTHER_STATIC_FRICTION,
                dynamic_friction=COFFEE_OTHER_DYNAMIC_FRICTION,
                armature=COFFEE_OTHER_ARMATURE,
            ),
        },
    )


# =====================================================================
# Command (the table-place goal)
# =====================================================================


@configclass
class MakeCoffeeCommandsCfg(dexverse_base_env.CommandsCfg):
    """Tabletop position target retained for the ``place_tracking`` shaping reward."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="object",
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(TABLE_GOAL_X, TABLE_GOAL_X),
            pos_y=(TABLE_GOAL_Y, TABLE_GOAL_Y),
            pos_z=(TABLE_GOAL_Z, TABLE_GOAL_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        position_only=True,
    )


# =====================================================================
# Manager configclasses
# =====================================================================


@configclass
class MakeCoffeeObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Robot state (base) + mug + milk + coffee machine joints/body + stage progress.

    ``state`` (observable, no velocities): mug/milk poses, coffee machine body
    position, knob joint position, and stage progress. ``privileged``: mug/milk
    linear and angular velocities (+ inherited robot ``joint_vel`` /
    ``hand_tips_state_b``). ``policy`` stays as the base's last-action-only
    group; ``proprio`` stays as the base's joint-pos-only group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        milk_pos_b = ObsTerm(
            func=mdp.object_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("milk")},
        )
        milk_quat_b = ObsTerm(
            func=mdp.object_quat_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("milk")},
        )
        machine_body_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("coffee_machine")},
        )
        coffee_knob_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("coffee_machine", joint_names=[COFFEE_KNOB_JOINT])},
        )
        stage_progress = ObsTerm(
            func=mdp.stage_progress,
            params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        milk_lin_vel_b = ObsTerm(
            func=mdp.object_lin_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("milk")},
        )
        milk_ang_vel_b = ObsTerm(
            func=mdp.object_ang_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("milk")},
        )

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class MakeCoffeeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Light per-step regularization + behaviour shaping + per-stage credit."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.0005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0005)

    # Bootstrap reaching + lifting the mug off the holder.
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=2.0,
    )
    mug_lift = RewTerm(
        func=mdp.object_lift_height,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": GRASP_MUG_LIFT_MIN},
    )
    # Drive the mug toward the table place goal.
    place_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=2.0,
        params={"std": 0.12, "command_name": "object_pose"},
    )
    # Reward turning the switch lever away from its start (|angle - start|).
    knob_progress = RewTerm(
        func=mdp.joint_range_progress_from_init,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("coffee_machine", joint_names=[COFFEE_KNOB_JOINT])},
    )
    stage_progress = RewTerm(
        func=stage_progress_reward,
        weight=3.0,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
    )
    success = RewTerm(
        func=mdp.stage_success_reward,
        weight=10.0,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
    )


@configclass
class MakeCoffeeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Time-out (base) + mug/milk out-of-bound + terminal-stage success."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.85, 0.85), "y": (-0.85, 0.85), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    milk_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.85, 0.85), "y": (-0.85, 0.85), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("milk"),
        },
    )
    success = DoneTerm(
        func=mdp.stage_success,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
    )


@configclass
class MakeCoffeeEventCfg(dexverse_base_env.EventCfg):
    """Resets: coffee machine, cup holder + mug (synced), milk."""

    # The mug resets relative to the holder (below), not on its own.
    reset_object = None

    reset_coffee_machine = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            },
            "asset_cfg": SceneEntityCfg("coffee_machine"),
        },
    )
    reset_coffee_machine_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("coffee_machine", joint_names=".*")},
    )

    # Holder is randomized first; the mug is then seated relative to its new pose.
    reset_cup_holder = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": list(CUP_HOLDER_RESET_X_RANGE),
                "y": list(CUP_HOLDER_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(CUP_HOLDER_RESET_YAW_RANGE),
            },
            "asset_cfg": SceneEntityCfg("cup_holder"),
        },
    )
    # Seat the mug TILTED on the (post-randomization) holder pin. The local offset
    # (holder frame, metres) already encodes the pin height, so z_offset=0.
    # ``quat_local`` recomposes the tilted hang pose and tracks the holder yaw
    # (the remove_cup_from_rack mechanism).
    reset_mug_on_holder = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("object"),
            "source_cfg": SceneEntityCfg("cup_holder"),
            "source_local_offset": MUG_OFFSET_ON_HOLDER,
            "z_offset": 0.0,
            "quat_local": MUG_ROT_LOCAL_ON_HOLDER,
        },
    )

    reset_milk = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": list(MILK_RESET_X_RANGE),
                "y": list(MILK_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(MILK_RESET_YAW_RANGE),
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("milk"),
        },
    )


# =====================================================================
# Scene + env config
# =====================================================================


@configclass
class MakeCoffeeSceneCfg(dexverse_base_env.SceneCfg):
    """Mug (object) + kinematic cup holder + milk bottle + coffee machine."""

    object: RigidObjectCfg = _build_mug_cfg()
    cup_holder: RigidObjectCfg = _build_cup_holder_cfg()
    milk: RigidObjectCfg = _build_milk_cfg()
    coffee_machine: ArticulationCfg = _build_coffee_machine_cfg()


@configclass
class LongHorizonMakeCoffeeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pour milk -> place on machine -> turn knob."""

    supports_object_pose_command: bool = True

    scene: MakeCoffeeSceneCfg = MakeCoffeeSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=False,
    )
    commands: MakeCoffeeCommandsCfg = MakeCoffeeCommandsCfg()
    observations: MakeCoffeeObservationsCfg = MakeCoffeeObservationsCfg()
    rewards: MakeCoffeeRewardsCfg = MakeCoffeeRewardsCfg()
    events: MakeCoffeeEventCfg = MakeCoffeeEventCfg()
    terminations: MakeCoffeeTerminationsCfg = MakeCoffeeTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Five sub-stages -- give the episode room to complete them.
        self.episode_length_s = 45.0
        self.is_finite_horizon = True
        self.commands.object_pose.resampling_time_range = (
            self.episode_length_s + 1.0,
            self.episode_length_s + 1.0,
        )

        # Wire fingertip body names through to privileged observations.
        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names

        # Fingertip-vs-mug contact sensors (when the robot setup enables them).
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            finger_tip_body_list = self.robot_config.fingertip_body_names
            for link_name in finger_tip_body_list:
                setattr(
                    self.scene,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path=f"{tip_prim_prefix}{link_name}",
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )
            self.observations.contact.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None


@configclass
class LongHorizonMakeCoffeeEnvFloatingDexHandRightCfg(LongHorizonMakeCoffeeEnvCfg):
    """Single right floating dexterous hand variant."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()

        setup_floating_teleop(self)
