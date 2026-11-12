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

    1. pour_milk          -- hold the tipped bottle's spout over the upright mug
    2. place_on_machine   -- carry the mug unspilled to the machine's tray, let go
    3. rotate_knob        -- turn the machine's switch lever (start "brewing")

Stages latch through a strict, persistent :class:`mdp.StageGraphSpec`. Success also
needs the lever turned and the mug on the tray at that moment, so it can be lost
again if the mug gets knocked. See ``checks.py`` for what each stage verifies.

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
  above the mug's mouth (see :func:`checks.milk_poured_over_mug`).
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
from dexverse.robot_agents.shadow.floating import FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES
from dexverse.tasks.config.bimanual.usd_helpers import ensure_single_rigid_body
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from scipy.spatial.transform import Rotation as R

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ...floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ..handling import (
    configure_cpu_camera_copies,
    configure_light_virtual_wrist_links,
    grip_material_event,
    tune_wrist_damping,
    tune_wrist_camera_bodies,
)
from .checks import (
    PLACE_MARK_RADIUS,
    milk_poured_over_mug,
    mug_placed_and_released,
    mug_settled_on_tray,
    report_lever_stage,
    report_place_stage,
)
from .usd_prep import prepare_coffee_machine_usd
from .feedback import reset_coffee_feedback, update_coffee_visuals

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
COFFEE_KNOB_JOINT = "RevoluteJoint_coffee_machine_6_up"  # switch lever, Y, +/-90 deg in the USD
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

# Mug placement spot ON the machine, in the machine root (E_body_3) local frame: on
# the drip tray, midway under the portafilter's two outlets, which is where people set
# the mug. The y reach scales with the machine spawn scale (the geometry scales with
# it). The mug's origin is at its BASE, so the local z puts the target at the height a
# mug's root rests at on the tray, measured from recorded demos (the yaw about +Z does
# not change the world z of the offset).
COFFEE_TRAY_REST_HEIGHT = 0.0855  # mug root above the table top when on the tray
COFFEE_PLACE_LOCAL = (
    0.0,
    -0.054 * COFFEE_SCALE_FACTOR,
    TABLE_TOP_Z - COFFEE_Z + COFFEE_TRAY_REST_HEIGHT,
)


# The visible mark at that spot. The machine's -90 deg yaw maps local (x, y) to world
# (y, -x) offsets from its centre. Its top is level with the base of a mug resting there.
PLACE_MARK_THICKNESS = 0.002
PLACE_MARK_COLOR = (0.1, 0.85, 0.3)
PLACE_MARK_POS = (
    COFFEE_CENTER_X + COFFEE_PLACE_LOCAL[1],
    COFFEE_CENTER_Y - COFFEE_PLACE_LOCAL[0],
    TABLE_TOP_Z + COFFEE_TRAY_REST_HEIGHT - 0.5 * PLACE_MARK_THICKNESS,
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
KNOB_ROTATE_DISP = math.radians(35.0)  # switch lever turned >= 35 deg from rest
# The lever stops just past that in either direction, so pushing it to its stop always counts.
KNOB_STOP_RAD = math.radians(38.0)

LONG_HORIZON_STAGE_GRAPH_KEY = "long_horizon.make_coffee"

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6


# =====================================================================
# Custom predicates
# =====================================================================


def knob_turned_with_mug_on_tray(env, threshold_rad: float = KNOB_ROTATE_DISP) -> torch.Tensor:
    """True per-env while the switch lever is turned and the mug still sits on the tray."""
    knob = mdp.joint_relative_move(
        env,
        threshold=threshold_rad,
        asset_cfg=SceneEntityCfg("coffee_machine", joint_names=[COFFEE_KNOB_JOINT]),
        mode="displacement",
        op=">=",
        reduce="any",
    )
    return knob & mug_settled_on_tray(env, place_local=COFFEE_PLACE_LOCAL, machine_body=COFFEE_BODY)


def make_coffee_success(env, task_key: str, threshold_rad: float = KNOB_ROTATE_DISP) -> torch.Tensor:
    """Every stage done in order, and the finished state still true right now.

    The stage flags latch. Checking the end state live as well makes the consecutive
    steps counted by ``--num_success_steps`` mean something.
    """
    stages_done = mdp.stage_success(env, task_key=task_key, persistent=True)
    finished_now = knob_turned_with_mug_on_tray(env, threshold_rad)
    flags = mdp.evaluate_stage_graph(env, task_key=task_key, persistent=True)
    poured, placed = flags["pour_milk"], flags["place_on_machine"]
    tips = tuple(FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES)
    report_place_stage(env, poured & ~placed, COFFEE_PLACE_LOCAL, COFFEE_BODY, tips)
    report_lever_stage(env, placed & ~finished_now, COFFEE_KNOB_JOINT, threshold_rad, COFFEE_PLACE_LOCAL, COFFEE_BODY)
    success = stages_done & finished_now
    update_coffee_visuals(env, flags, success)
    return success


def stage_progress_reward(env, task_key: str, persistent: bool = True) -> torch.Tensor:
    """Dense long-horizon shaping: the per-env stage-completion ratio in [0, 1]."""
    return mdp.stage_progress(env, task_key=task_key, persistent=persistent).squeeze(-1)


# =====================================================================
# Stage graph: strict, persistent -- one sub-stage per step of the task
# =====================================================================

MAKE_COFFEE_STAGE_GRAPH = mdp.StageGraphSpec(
    stages=(
        mdp.StageSpec(name="pour_milk", func=milk_poured_over_mug),
        mdp.StageSpec(
            name="place_on_machine",
            func=mug_placed_and_released,
            params={
                "place_local": COFFEE_PLACE_LOCAL,
                "machine_body": COFFEE_BODY,
                "fingertip_bodies": tuple(FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES),
            },
            deps=("pour_milk",),
        ),
        mdp.StageSpec(
            name="rotate_knob",
            func=knob_turned_with_mug_on_tray,
            params={"threshold_rad": KNOB_ROTATE_DISP},
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


def _build_place_mark_cfg() -> AssetBaseCfg:
    """A disc on the drip tray showing where the mug goes. Visual only; nothing touches it.

    Part of the scene like a printed mark, so it also appears in camera observations.
    """
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlaceMark",
        spawn=sim_utils.CylinderCfg(
            radius=PLACE_MARK_RADIUS,
            height=PLACE_MARK_THICKNESS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=PLACE_MARK_COLOR, emissive_color=tuple(0.4 * c for c in PLACE_MARK_COLOR)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PLACE_MARK_POS),
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
        func=make_coffee_success,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY},
    )


@configclass
class MakeCoffeeEventCfg(dexverse_base_env.EventCfg):
    """Resets: grip materials, lever stop, coffee machine, cup holder + mug (synced), milk."""

    robot_physics_material = grip_material_event("robot")
    mug_physics_material = grip_material_event("object")
    milk_physics_material = grip_material_event("milk")

    # The mug resets relative to the holder (below), not on its own.
    reset_object = None

    # On reset, not startup: record_demos.py restarts the physics scene on every reset,
    # which silently restores the lever limits authored in the USD.
    limit_lever_travel = EventTerm(
        func=mdp.events.set_joint_position_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("coffee_machine", joint_names=[COFFEE_KNOB_JOINT]),
            "lower": -KNOB_STOP_RAD,
            "upper": KNOB_STOP_RAD,
        },
    )

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
    """Mug (object) + kinematic cup holder + milk bottle + coffee machine + mark on its tray."""

    object: RigidObjectCfg = _build_mug_cfg()
    cup_holder: RigidObjectCfg = _build_cup_holder_cfg()
    milk: RigidObjectCfg = _build_milk_cfg()
    coffee_machine: ArticulationCfg = _build_coffee_machine_cfg()
    place_mark: AssetBaseCfg = _build_place_mark_cfg()


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
        self.events.clear_stage_latches = EventTerm(
            func=reset_coffee_feedback, mode="reset", params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY}
        )
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

        tune_wrist_camera_bodies(self)
        configure_cpu_camera_copies(self)
        tune_wrist_damping(self)
        configure_light_virtual_wrist_links(self)
        setup_floating_teleop(self)
