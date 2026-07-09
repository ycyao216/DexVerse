# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon bimanual task: season a salmon fillet and bake it.

A salmon fillet rests in a *movable* red tray; salt and pepper spice jars and a
(closed) toaster oven sit on the table. The bimanual policy / teleoperator must:

    - season  -- pour the salt jar AND the pepper jar over the salmon (lift +
                 tilt past ~100 deg while held over the salmon). Either jar, in
                 any order, at ANY time in the episode.
    - load    -- open the door (implicit prerequisite) and carry the tray +
                 salmon into the oven cavity.
    - close   -- swing the oven door shut while the salmon is contained.
    - knob    -- rotate an oven control knob while the salmon is contained.

Success (strict, persistent :class:`mdp.StageGraphSpec`): salt poured AND pepper
poured (each latched the moment it happens, any time) AND the salmon is inside
the oven cavity AND the door is closed AND the knob is rotated while the salmon
is in the oven. The terminal stage is ``rotate_knob``.

This replaces the earlier single-hand tong-grasp version: the tong was too hard
to grasp, so the salmon now simply rides the movable tray into the oven, and the
"pour" seasoning sub-task (modeled on ``pour_can_cfg``) is added.

Asset notes
~~~~~~~~~~~
- Oven (``synthesis/oven001/model_oven_2.usd``): clean single-articulation-root
  asset (root ``E_body_4``), all-``convexDecomposition`` colliders. Front face
  (door + three knobs) authored on local ``-Y``; yawed ``-90 deg`` about ``+Z`` so
  the front faces the robot at ``-X``. The door is bottom-hinged; every manipulable
  joint is driven by a *friction-detent* :class:`ImplicitActuatorCfg`
  (``stiffness=0``; static ``friction`` holds it where the hand leaves it). Scaled
  up by ``OVEN_SCALE``.
- Salmon (``synthesis/salmon/salmon.usd``), tray
  (``tray001/...``) and the two spice jars (``SpiceJarSalt`` / ``SpiceJarPepper``)
  are not physics-ready as shipped; :func:`prepare_rigid_mesh_usd` bakes
  single-rigid-body, collidable copies. The salmon is the canonical
  ``scene.object``; the tray is now a *dynamic* (movable) carrier.

Placement / non-overlap
~~~~~~~~~~~~~~~~~~~~~~~~~
The oven sits at the back (large ``+X``). Its bottom-hinged door, when swung fully
open, lies flat over a keep-out zone in front of the oven (roughly world
``x in [-0.24, 0.16], y in [-0.23, 0.33]`` at ``OVEN_SCALE=1.4``). The tray
(front-centre, well forward) and the two jars (front-left / front-right, beyond
the door's lateral reach) start outside that zone so opening the door cannot knock
them. The tray is then lifted *over* the open door and into the cavity.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from dexverse.assets import SYNTHESIS_DIR
from dexverse.tasks.config.bimanual.bimanual_contact_links import (
    contact_sensor_names,
    resolve_bimanual_contact_links,
)
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
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ...floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ...robot_init import (
    align_retargeter_wrist_origin_to_init,
    set_robot_wrist_init_world_pos,
)
from .usd_prep import prepare_rigid_mesh_usd

# =====================================================================
# USD asset paths (salmon / tray / jars prepped into rigid, collidable copies)
# =====================================================================

OVEN_USD_PATH = str(SYNTHESIS_DIR / "oven001" / "model_oven_2.usd")
SALMON_USD_PATH = prepare_rigid_mesh_usd(SYNTHESIS_DIR / "salmon" / "salmon.usd", approximation="convexHull")
# convexDecomposition keeps the tray cavity hollow so the salmon nests inside.
TRAY_USD_PATH = prepare_rigid_mesh_usd(
    SYNTHESIS_DIR / "tray001" / "model_redtray__rb_cleaned.usd",
    approximation="convexDecomposition",
)
# Spice jars: have colliders but no rigid body; prep adds a single rigid body and
# a convexHull collider (graspable). Upright with +Z up, base at local z=0.
SALT_USD_PATH = prepare_rigid_mesh_usd(
    SYNTHESIS_DIR / "SpiceJarSalt" / "model_SpiceJarSalt_69323.usd",
    approximation="convexHull",
)
PEPPER_USD_PATH = prepare_rigid_mesh_usd(
    SYNTHESIS_DIR / "SpiceJarPepper" / "model_SpiceJarPepper_69323.usd",
    approximation="convexHull",
)


# =====================================================================
# Oven joints / bodies and geometry
# =====================================================================

OVEN_DOOR_JOINT = "RevoluteJoint_oven_2_middle"  # revolute about X, 0..90 deg
OVEN_RACK_JOINT = "PrismaticJoint_oven_2_middle"  # prismatic rack (held closed)
OVEN_KNOB_JOINT = "RevoluteJoint_oven_2_up"  # the knob we must rotate
OVEN_OTHER_KNOB_JOINTS = ["RevoluteJoint_oven_2_middle_1", "RevoluteJoint_oven_2_down"]
OVEN_BODY = "E_body_4"  # articulation root body
OVEN_GRILL_BODY = "E_grill_1"  # the wire rack body

# Uniform spawn scale for the whole oven articulation (geometry + joint frames).
# At 1.0 the oven is ~0.50 m wide / 0.39 m deep / 0.33 m tall; bumped up so the
# cavity is roomy enough to receive the tray. Single knob -- every oven-frame
# length below (rack top z, cavity AABB) scales off it. Keep <= ~1.45 or the back
# edge runs off the 1.5 m table at OVEN_CENTER_X=0.44 (back ~ 0.44 + 0.20*scale).
# A larger oven also enlarges the open-door keep-out zone in front (see the module
# docstring / the tray + jar start positions); lower this to ease the layout.
OVEN_SCALE = 1.4

# Rack-top height above the oven root origin (local z of the grill surface).
OVEN_RACK_TOP_LOCAL_Z = 0.148 * OVEN_SCALE

# Friction-detent drive (stiffness=0 -> no setpoint; static friction holds each
# joint in place). TUNE OVEN_DOOR_STATIC_FRICTION FIRST if the door sags open or
# won't open. (These values need an in-sim re-check at OVEN_SCALE != 1: the
# no-mass joints' auto-computed inertia changes with scale.)
OVEN_DOOR_STATIC_FRICTION = 3.0
OVEN_DOOR_DYNAMIC_FRICTION = 2.0
OVEN_DOOR_DAMPING = 2.0
OVEN_RACK_STATIC_FRICTION = 3.0
OVEN_RACK_DYNAMIC_FRICTION = 2.0
OVEN_RACK_DAMPING = 2.0
OVEN_KNOB_STATIC_FRICTION = 0.12
OVEN_KNOB_DYNAMIC_FRICTION = 0.06
OVEN_KNOB_DAMPING = 0.3


# =====================================================================
# Layout (env-local; table centred at origin, top at DEFAULT_TABLE_TOP_HEIGHT)
# =====================================================================

TABLE_TOP_Z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT  # 0.6

# Oven: back-centre, front (-Y local) yawed to face the robot at -X.
OVEN_CENTER_X = 0.44
OVEN_CENTER_Y = 0.0
OVEN_Z = TABLE_TOP_Z  # oven origin is at its base
# -90 deg yaw about +Z: local -Y front -> world -X (toward the robot).
OVEN_INIT_ROT = (math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4))
OVEN_RESET_X_RANGE = (-0.03, 0.03)
OVEN_RESET_Y_RANGE = (-0.04, 0.04)
OVEN_RESET_YAW_RANGE = (-0.08, 0.08)

# Tray + salmon: front-centre, well forward of the oven so the open door (which
# swings flat to ~world x=-0.24 at OVEN_SCALE=1.4) cannot reach them. The robot
# lifts the tray (bimanual) over the open door and into the cavity. NOTE: this is
# a long carry; reduce OVEN_SCALE to bring the oven (and the keep-out zone) in.
TRAY_CENTER_X = -0.46
TRAY_CENTER_Y = 0.0
TRAY_Z = TABLE_TOP_Z + 0.01
TRAY_MASS = 0.3
SALMON_MASS = 0.2
# Spawn the salmon above the tray floor so it settles into the tray cavity.
SALMON_Z = TABLE_TOP_Z + 0.05
# +90 deg roll about +X stands the fillet's thin (4 cm) axis up so it lies flat.
SALMON_INIT_ROT = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
# Per-reset jitter applied to the tray+salmon rig as one rigid transform. Kept
# small so the (large) tray stays clear of the door keep-out zone.
TRAY_RESET_X_RANGE = (-0.03, 0.03)
TRAY_RESET_Y_RANGE = (-0.03, 0.03)
TRAY_RESET_YAW_RANGE = (-0.15, 0.15)


# =====================================================================
# Spice jars (salt + pepper): dynamic, graspable, poured over the salmon
# =====================================================================

JAR_MASS = 0.15
JAR_Z = TABLE_TOP_Z + 0.01  # base sits on the table
JAR_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # upright; local +Z points up
# Salt front-left, pepper front-right -- both beyond the door's lateral reach.
SALT_CENTER_X = -0.10
SALT_CENTER_Y = 0.46
PEPPER_CENTER_X = -0.10
PEPPER_CENTER_Y = -0.42
JAR_RESET_X_RANGE = (-0.05, 0.05)
JAR_RESET_Y_RANGE = (-0.05, 0.05)
JAR_RESET_YAW_RANGE = (-math.pi, math.pi)


# =====================================================================
# Stage-graph thresholds
# =====================================================================

# Pour (modeled on pour_can_cfg): a jar counts as "poured" when it is lifted off
# the table AND tilted past POUR_ANGLE_RAD AND held over the salmon (xy).
POUR_ANGLE_RAD = math.radians(100.0)  # local +Z tilted >= 100 deg from world +Z
POUR_LIFT_MIN = 0.10  # jar lifted >= 10 cm above its table rest
POUR_OVER_XY_TOL = 0.15  # jar within this horizontal dist of the salmon
POUR_AXIS_LOCAL = (0.0, 0.0, 1.0)  # jar's "up" axis (upright spawn)

CLOSE_DOOR_RATIO = 0.12  # door range-ratio <= 0.12 (~11 deg, ~shut)
KNOB_ROTATE_DISP = 0.7  # knob rotated >= 0.7 rad (~40 deg) from rest

# "Salmon inside the oven cavity" acceptance AABB, in the oven ROOT (E_body_4)
# frame -- same idea/encoding as trash_drawer_sort_simple's container AABBs. The
# oven is fixed-base, so this box is static in world. Measured from
# model_oven_2.usd (root frame): rack surface at local z=0.148, rack footprint
# x[-0.22, 0.15] y[-0.14, 0.16], outer shell x[-0.25, 0.25] y[-0.19, 0.20]
# z[0, 0.33]. Inset for the walls; floor at the rack (the tray rests on it),
# ceiling below the top. Base extents are for OVEN_SCALE=1.0; scaled since the
# box lives in the (scaled) oven frame in meters.
SALMON_IN_OVEN_LOCAL_MIN: tuple[float, float, float] = tuple(v * OVEN_SCALE for v in (-0.18, -0.13, 0.14))
SALMON_IN_OVEN_LOCAL_MAX: tuple[float, float, float] = tuple(v * OVEN_SCALE for v in (0.13, 0.15, 0.34))

LONG_HORIZON_STAGE_GRAPH_KEY = "long_horizon.oven_bake_salmon"

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6


# =====================================================================
# Custom predicates
# =====================================================================


def _salmon_pos_in_oven_frame(env) -> torch.Tensor:
    """Salmon root position expressed in the oven root (E_body_4) frame."""
    oven = env.scene["oven"]
    salmon = env.scene["object"]
    return quat_apply_inverse(
        oven.data.root_quat_w,
        salmon.data.root_pos_w - oven.data.root_pos_w,
    )


def salmon_in_oven_cavity(
    env,
    lower: tuple[float, float, float] = SALMON_IN_OVEN_LOCAL_MIN,
    upper: tuple[float, float, float] = SALMON_IN_OVEN_LOCAL_MAX,
) -> torch.Tensor:
    """True per-env when the salmon sits inside the oven-cavity AABB (oven frame).

    Mirrors ``_object_in_aabb_mask`` in trash_drawer_sort_simple.
    """
    pos_b = _salmon_pos_in_oven_frame(env)
    lower_t = torch.tensor(lower, device=env.device, dtype=pos_b.dtype)
    upper_t = torch.tensor(upper, device=env.device, dtype=pos_b.dtype)
    return ((pos_b >= lower_t) & (pos_b <= upper_t)).all(dim=1)


def pour_over_salmon(
    env,
    jar_name: str,
    min_height: float = POUR_LIFT_MIN,
    threshold_rad: float = POUR_ANGLE_RAD,
    axis_local: tuple[float, float, float] = POUR_AXIS_LOCAL,
    xy_tol: float = POUR_OVER_XY_TOL,
) -> torch.Tensor:
    """True per-env when ``jar_name`` is being poured over the salmon.

    "Poured" = lifted above ``min_height`` AND its local ``axis_local`` tilted at
    least ``threshold_rad`` from world +Z (the lift_and_tilt pour criterion from
    pour_can_cfg) AND the jar is horizontally over the salmon within ``xy_tol``
    (so it actually seasons the salmon, not just any tilt). Used as a *persistent*
    stage with no deps -> it latches the first time it happens, at any point.
    """
    poured = mdp.lift_and_tilt(
        env,
        min_height=min_height,
        threshold_rad=threshold_rad,
        axis_local=axis_local,
        tilt_ge=True,
        object_cfg=SceneEntityCfg(jar_name),
    )
    jar = env.scene[jar_name]
    salmon = env.scene["object"]
    over = torch.norm(jar.data.root_pos_w[:, :2] - salmon.data.root_pos_w[:, :2], dim=-1) < xy_tol
    return poured & over


def door_closed_with_salmon_contained(env, close_ratio: float = CLOSE_DOOR_RATIO) -> torch.Tensor:
    """trash_drawer-style "receptacle closed AND object inside": the oven door is
    shut (range-ratio <= ``close_ratio``) AND the salmon is inside the cavity.
    """
    door_closed = mdp.joint_relative_move(
        env,
        threshold=close_ratio,
        asset_cfg=SceneEntityCfg("oven", joint_names=[OVEN_DOOR_JOINT]),
        mode="ratio",
        op="<=",
        reduce="any",
    )
    return door_closed & salmon_in_oven_cavity(env)


def knob_turned_with_salmon_contained(env, threshold_rad: float = KNOB_ROTATE_DISP) -> torch.Tensor:
    """Terminal predicate: the knob is rotated WHILE the salmon is in the oven."""
    knob = mdp.joint_relative_move(
        env,
        threshold=threshold_rad,
        asset_cfg=SceneEntityCfg("oven", joint_names=[OVEN_KNOB_JOINT]),
        mode="displacement",
        op=">=",
        reduce="any",
    )
    return knob & salmon_in_oven_cavity(env)


def stage_progress_reward(env, task_key: str, persistent: bool = True) -> torch.Tensor:
    """Dense long-horizon shaping: the per-env stage-completion ratio in [0, 1]."""
    return mdp.stage_progress(env, task_key=task_key, persistent=persistent).squeeze(-1)


# =====================================================================
# Stage graph: pours latch any time; load -> close -> knob is ordered. The
# door-opening step is implicit (the salmon cannot enter the cavity without
# it), so it is not its own stage.
# =====================================================================

OVEN_STAGE_GRAPH = mdp.StageGraphSpec(
    stages=(
        # --- Seasoning: unordered, latch the first time each pour happens. ---
        mdp.StageSpec(
            name="salt_poured",
            func=pour_over_salmon,
            params={"jar_name": "salt"},
        ),
        mdp.StageSpec(
            name="pepper_poured",
            func=pour_over_salmon,
            params={"jar_name": "pepper"},
        ),
        # --- Load -> close -> knob: gated on both pours via salmon_in_oven. ---
        mdp.StageSpec(
            name="salmon_in_oven",
            func=salmon_in_oven_cavity,
            params={},
            deps=("salt_poured", "pepper_poured"),
        ),
        mdp.StageSpec(
            name="close_door",
            func=door_closed_with_salmon_contained,
            params={"close_ratio": CLOSE_DOOR_RATIO},
            deps=("salmon_in_oven",),
        ),
        mdp.StageSpec(
            name="rotate_knob",
            func=knob_turned_with_salmon_contained,
            params={"threshold_rad": KNOB_ROTATE_DISP},
            deps=("close_door",),
        ),
    ),
    terminal_stage="rotate_knob",
    ordering_mode="strict",
    success_mode="substage",
)
mdp.register_stage_graph(LONG_HORIZON_STAGE_GRAPH_KEY, OVEN_STAGE_GRAPH, override=True)


# =====================================================================
# Asset configs
# =====================================================================


def _build_oven_cfg() -> ArticulationCfg:
    """Fixed-base oven articulation with friction-detent manipulable joints."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Oven",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OVEN_USD_PATH,
            scale=(OVEN_SCALE, OVEN_SCALE, OVEN_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=0.0, stiffness=0.0, damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(OVEN_CENTER_X, OVEN_CENTER_Y, OVEN_Z),
            rot=OVEN_INIT_ROT,
            joint_pos={
                OVEN_DOOR_JOINT: 0.0,  # door starts CLOSED
                OVEN_RACK_JOINT: 0.0,
                OVEN_KNOB_JOINT: 0.0,
                OVEN_OTHER_KNOB_JOINTS[0]: 0.0,
                OVEN_OTHER_KNOB_JOINTS[1]: 0.0,
            },
        ),
        actuators={
            "oven_door": ImplicitActuatorCfg(
                joint_names_expr=[OVEN_DOOR_JOINT],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=OVEN_DOOR_DAMPING,
                friction=OVEN_DOOR_STATIC_FRICTION,
                dynamic_friction=OVEN_DOOR_DYNAMIC_FRICTION,
            ),
            "oven_rack": ImplicitActuatorCfg(
                joint_names_expr=[OVEN_RACK_JOINT],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=OVEN_RACK_DAMPING,
                friction=OVEN_RACK_STATIC_FRICTION,
                dynamic_friction=OVEN_RACK_DYNAMIC_FRICTION,
            ),
            "oven_knobs": ImplicitActuatorCfg(
                joint_names_expr=[OVEN_KNOB_JOINT, *OVEN_OTHER_KNOB_JOINTS],
                effort_limit_sim=50.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=OVEN_KNOB_DAMPING,
                friction=OVEN_KNOB_STATIC_FRICTION,
                dynamic_friction=OVEN_KNOB_DYNAMIC_FRICTION,
            ),
        },
    )


def _build_salmon_cfg() -> RigidObjectCfg:
    """Salmon fillet -- the canonical ``scene.object`` (dynamic, rides the tray)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=SALMON_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SALMON_MASS),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TRAY_CENTER_X, TRAY_CENTER_Y, SALMON_Z),
            rot=SALMON_INIT_ROT,
        ),
    )


def _build_tray_cfg() -> RigidObjectCfg:
    """Red tray -- DYNAMIC (movable) carrier the salmon nests in; lifted into oven."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=TRAY_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # movable: robot picks it up
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TRAY_MASS),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TRAY_CENTER_X, TRAY_CENTER_Y, TRAY_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _build_jar_cfg(name: str, usd_path: str, center_x: float, center_y: float) -> RigidObjectCfg:
    """Dynamic, graspable spice jar (salt / pepper), upright on the table."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=JAR_MASS),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(center_x, center_y, JAR_Z),
            rot=JAR_INIT_ROT,
        ),
    )


# =====================================================================
# Manager configclasses
# =====================================================================


@configclass
class OvenBakeSalmonObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Robot state (base) + salmon + tray + jars + oven joints/bodies + stage progress.

    ``state`` (observable, no velocities): salmon/tray/salt/pepper/oven-body
    poses, oven door/knob joint positions, and stage progress. ``privileged``:
    salmon/tray/salt/pepper/oven-body velocities (+ inherited robot
    ``joint_vel`` / ``hand_tips_state_b``). ``policy`` stays as the base's
    last-action-only group; ``proprio`` stays as the base's joint-pos-only
    group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        tray_pose_b = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("tray"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        salt_pose_b = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("salt"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        pepper_pose_b = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("pepper"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        oven_body_pose_b = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("oven", body_names=[OVEN_BODY]),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        oven_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("oven", joint_names=[OVEN_DOOR_JOINT, OVEN_KNOB_JOINT])},
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
        tray_vel_b = ObsTerm(
            func=mdp.body_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("tray"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        salt_vel_b = ObsTerm(
            func=mdp.body_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("salt"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        pepper_vel_b = ObsTerm(
            func=mdp.body_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("pepper"),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        oven_body_vel_b = ObsTerm(
            func=mdp.body_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("oven", body_names=[OVEN_BODY]),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class OvenBakeSalmonRewardsCfg(dexverse_base_env.RewardsCfg):
    """Light per-step regularization + per-stage dense shaping + sparse success."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.0005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0005)

    door_open = RewTerm(
        func=mdp.joint_range_progress,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("oven", joint_names=[OVEN_DOOR_JOINT])},
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
class OvenBakeSalmonTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Time-out (base) + salmon out-of-bound + terminal-stage success."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.85, 0.85), "y": (-0.85, 0.85), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    success = DoneTerm(
        func=mdp.stage_success,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
    )


@configclass
class OvenBakeSalmonEventCfg(dexverse_base_env.EventCfg):
    """Resets for the oven, the movable tray+salmon rig, and the two jars."""

    # The salmon resets together with the tray (below), not on its own.
    reset_object = None

    reset_oven = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": list(OVEN_RESET_X_RANGE),
                "y": list(OVEN_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(OVEN_RESET_YAW_RANGE),
            },
            "asset_cfg": SceneEntityCfg("oven"),
        },
    )
    reset_oven_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("oven", joint_names=".*")},
    )

    # Tray + salmon reset together as one rigid rig (salmon main, tray support).
    # Both are now dynamic, so zero_support_velocity clears the tray's carried-over
    # velocity too. The salmon's spawn xy == the tray's spawn xy -> pivot is the
    # tray centre.
    reset_tray_with_salmon = EventTerm(
        func=mdp.reset_articulation_with_supports_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "support_cfgs": [SceneEntityCfg("tray")],
            "zero_support_velocity": True,
            "pose_range": {
                "x": list(TRAY_RESET_X_RANGE),
                "y": list(TRAY_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(TRAY_RESET_YAW_RANGE),
            },
        },
    )

    reset_salt = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": list(JAR_RESET_X_RANGE),
                "y": list(JAR_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(JAR_RESET_YAW_RANGE),
            },
            "asset_cfg": SceneEntityCfg("salt"),
        },
    )
    reset_pepper = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": list(JAR_RESET_X_RANGE),
                "y": list(JAR_RESET_Y_RANGE),
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": list(JAR_RESET_YAW_RANGE),
            },
            "asset_cfg": SceneEntityCfg("pepper"),
        },
    )


# =====================================================================
# Scene + env config
# =====================================================================


@configclass
class OvenBakeSalmonSceneCfg(dexverse_base_env.SceneCfg):
    """Oven articulation + salmon (object) + movable tray + salt + pepper jars."""

    oven: ArticulationCfg = _build_oven_cfg()
    object: RigidObjectCfg = _build_salmon_cfg()
    tray: RigidObjectCfg = _build_tray_cfg()
    salt: RigidObjectCfg = _build_jar_cfg("salt", SALT_USD_PATH, SALT_CENTER_X, SALT_CENTER_Y)
    pepper: RigidObjectCfg = _build_jar_cfg("pepper", PEPPER_USD_PATH, PEPPER_CENTER_X, PEPPER_CENTER_Y)


@configclass
class LongHorizonOvenBakeSalmonEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Season (salt + pepper) -> load tray into oven -> close door -> rotate knob."""

    supports_object_pose_command: bool = False

    scene: OvenBakeSalmonSceneCfg = OvenBakeSalmonSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=False,
    )
    observations: OvenBakeSalmonObservationsCfg = OvenBakeSalmonObservationsCfg()
    rewards: OvenBakeSalmonRewardsCfg = OvenBakeSalmonRewardsCfg()
    events: OvenBakeSalmonEventCfg = OvenBakeSalmonEventCfg()
    terminations: OvenBakeSalmonTerminationsCfg = OvenBakeSalmonTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Multi-stage bimanual task -- give the episode room to complete it.
        self.episode_length_s = 50.0
        self.is_finite_horizon = True

        # Pull both palms ~5 cm back along world -x so the hands don't start
        # right on top of the tray. World-frame so the same target applies
        # across embodiments: floating-shadow bimanual writes per-hand
        # rh_/lh_*_translation_joint values (default palm world-x = 0.30).
        # Override in a subclass if a given embodiment needs a different x.
        set_robot_wrist_init_world_pos(self, x=-0.35)

        # Wire fingertip body names through to privileged observations.
        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )

        # Hand-vs-salmon contact sensors (when the robot setup enables them).
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            contact_links = resolve_bimanual_contact_links(
                robot_type=self.robot_type,
                robot_config=self.robot_config,
                robot_cfg=self.scene.robot,
            )
            for link_name in contact_links.all:
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
                params={"contact_sensor_names": contact_sensor_names(contact_links.all)},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None


@configclass
class LongHorizonOvenBakeSalmonEnvFloatingShadowBimanualCfg(LongHorizonOvenBakeSalmonEnvCfg):
    """Bimanual floating Shadow-hand variant (the canonical task variant)."""

    robot_type: str = "floating_shadow_bimanual"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()

        # Boost the floating-base translation + rotation joints' effort, stiffness,
        # and damping for this env: the oven door is heavy and bottom-hinged
        # (gravity + friction-detent both resist opening) and the tray-with-salmon
        # is then lifted over the open door into the cavity. Default values from
        # FLOATING_SHADOW_BIMANUAL_CFG are tuned for lighter table-top work and
        # the hand can stall against the door. Finger-joint actuators left alone.
        if self.robot_type == "floating_shadow_bimanual":
            robot_actuator = self.scene.robot.actuators.get("floating_shadow_bimanual_actuators")
            if robot_actuator is not None:
                robot_actuator.effort_limit_sim = {
                    **robot_actuator.effort_limit_sim,
                    "(lh|rh)_(x|y|z)_translation_joint": 40.0,
                    "(lh|rh)_(x|y|z)_rotation_joint": 40.0,
                }
                robot_actuator.stiffness = {
                    **robot_actuator.stiffness,
                    "(lh|rh)_(x|y|z)_translation_joint": 4000.0,
                    "(lh|rh)_(x|y|z)_rotation_joint": 4000.0,
                }
                robot_actuator.damping = {
                    **robot_actuator.damping,
                    "(lh|rh)_(x|y|z)_translation_joint": 600.0,
                    "(lh|rh)_(x|y|z)_rotation_joint": 600.0,
                }

        setup_floating_teleop(self)

        # Re-stamp the absolute retargeter's wrist_joint_origin to the post-shift
        # init palm pose. Required because the base class shifts the init joint
        # values via set_robot_wrist_init_world_pos, but the retargeter origin
        # was precomputed from the OLD default (FLOATING_SHADOW_BIMANUAL_SIMPLE_
        # ABSOLUTE_WRIST_ORIGIN); without this, teleop pulls the wrist back to
        # the old origin and the init shift appears to have no effect.
        align_retargeter_wrist_origin_to_init(self)
