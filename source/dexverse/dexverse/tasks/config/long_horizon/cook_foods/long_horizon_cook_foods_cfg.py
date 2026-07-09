# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon: cook foods (place pot on stove + pour water into pot).

Two-stage task that re-uses the same single-object pour template as
``pour_can_cfg`` (small water bottle is the pour object) together with the
flat-placement template from ``grasp_pan_cfg`` (pot002 is placed on the
stove burner). Success requires BOTH sub-tasks:

  - the pot is on the stove burner xy and lying roughly flat, and
  - the bottle is lifted, tilted past the pour threshold, and its spout xy
    is over the pot.

The pour-goal indicator is slaved to the pot each reset, so the policy must
move the pot before the pour target is in a useful place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import torch
from dexverse.assets import LONG_HORIZON_EXTRA_COOKING_DIR, SYNTHESIS_DIR, YCB_DIR
from dexverse.tasks.config.floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG
from dexverse.tasks.config.functional.base_cfg import (
    POUR_ANGLE_RAD,
    ForbiddenZone,
    FunctionalPourEnvFloatingDexHandRightCfg,
    build_object_cfg_from_usd,
)
from dexverse.tasks.config.robot_init import (
    align_retargeter_wrist_origin_to_init,
    set_robot_wrist_init_world_pos,
)
from dexverse.tasks.mdp.utils import axis_tilt_angle
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ...articulation.articulation_base.usd_helpers import (
    ensure_single_articulation_root,
)

SYNTHESIS_COOKING_DIR = LONG_HORIZON_EXTRA_COOKING_DIR

BOTTLE_USD_PATH = str(SYNTHESIS_COOKING_DIR / "water_bottle_opened" / "water_bottle_opened.usd")
# The authored bottle mesh is long along local +Y. This init rotation puts
# local +Y roughly on world +Z, so the bottle stands upright at reset.
BOTTLE_ROT_INIT = (0.707, 0.707, 0, 0)
BOTTLE_MASS = 0.3
BOTTLE_HALF_HEIGHT_EST = 0.00  # TODO: measure in viewport
# The OBJ bbox is x/z ~= +/-0.029 m and y = 0..0.20 m; the narrow neck is
# at local +Y. The stage gate rotates this local point by the live bottle
# quaternion before comparing xy against the pot root.
BOTTLE_POUR_AXIS_LOCAL = (0.0, 1.0, 0.0)
BOTTLE_POUR_SPOUT_LOCAL_OFFSET = (0.0, 0.20, 0.0)

POT_USD_PATH = str(SYNTHESIS_COOKING_DIR / "pot002" / "model_pot2.usd")
POT_SCALE = (1.5, 1.5, 1.5)
POT_MASS = 0.4
# 180° rotation about world Z in (w, x, y, z) order, so the pot's
# handles face the opposite direction at spawn.
POT_ROT_INIT = (0.0, 0.0, 0.0, 1.0)
POT_HALF_HEIGHT_EST = 0.03  # TODO: measure in viewport

KITCHEN_CABINET_USD_PATH = str(SYNTHESIS_DIR / "kitchencabinet_01" / "model_Kitchencabinet.usd")
KITCHEN_CABINET_SCALE = (1.0, 1.0, 1.0)
KITCHEN_CABINET_INIT_POS = (-0.8, -4.25, 0.0)
# 90 degree about z axis so the cabinet's visible counter faces the robot
# at spawn. (w, x, y, z) order.
KITCHEN_CABINET_INIT_ROT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)

# Cabinet-local position of the visible counter top centre. The cabinet
# USD authors its visible body ≈ 5.85 m from the asset origin in the
# cabinet's local +X direction, with the counter surface at ≈ 0.85 m
# above the root. Every prop on the counter is positioned relative to
# THIS point in cabinet-local frame, then transformed to world via
# ``cabinet_pos + R(cabinet_quat) * local_offset``. Tune if you swap to
# a different cabinet asset.
CABINET_COUNTER_LOCAL_ORIGIN = (5.85, 0.0, 0.85)

# Invisible cuboid support: kept so the inherited tabletop utilities have
# a stable surface to anchor against. The visible furniture is the
# cabinet. Sits under the cabinet root and follows it horizontally; its
# top surface is held at ``DEFAULT_TABLE_TOP_HEIGHT`` so the base class's
# tabletop math keeps working.
KITCHEN_COUNTER_TOP_Z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
KITCHEN_COUNTER_SUPPORT_SIZE = (1.45, 1.0, 0.04)
KITCHEN_COUNTER_SUPPORT_POS = (
    KITCHEN_CABINET_INIT_POS[0],
    KITCHEN_CABINET_INIT_POS[1],
    KITCHEN_COUNTER_TOP_Z - KITCHEN_COUNTER_SUPPORT_SIZE[2] * 0.5,
)

STOVE_USD_PATH = str(SYNTHESIS_DIR / "cooker007" / "model_cooker_007.usd")
STOVE_SCALE = (1.2, 1.2, 1.2)
# Per-prop offset (in cabinet local frame) from the counter centre to
# this prop's spawn pose. Small intuitive values: y slides the prop
# along the counter, x walks toward/away from the cabinet edge, z
# embeds the root below the counter top (stove only) or stays 0 so the
# base + half-height lands the prop flush.
STOVE_OFFSET_FROM_COUNTER = (-1.825, -0.775, -0.05)
# Stove rotation. World-frame quaternion (w, x, y, z) -- *not* composed
# with the cabinet rotation. Pick the orientation you want the stove to
# physically have in the world; if you rotate the cabinet and want the
# stove to swing with it, also update this value.
STOVE_INIT_ROT = (0.70710678118654758, 0.0, 0.0, -0.7071067811865476)
# Stove-LOCAL offset (in SCALED metres -- the frame the success gate uses:
# ``burner_w = stove.root_pos_w + R(stove.root_quat_w) * offset``) from the
# stove root to the LEFT burner grate. "Left" = the robot's left (world +Y;
# the robot faces +X), which for cooker007 is prim ``E_body_7/E_part4_18``
# (authored local centre ``(-0.1981, 0.0283, 0.0331)``; the right burner
# ``E_part4_9`` sits at +x). Scaled by ``STOVE_SCALE`` because the spawn scale
# bakes into the child-geometry offset relative to the root. The gate checks
# xy only, so z is left at 0. To target the right burner instead, flip the
# x sign.
POT_ON_STOVE_LOCAL_OFFSET = (
    -0.1981 * STOVE_SCALE[0],
    0.0283 * STOVE_SCALE[1],
    0.0,
)

# ---- Stove knobs (the "turn on the stove" final step) ----
# The cooker007 USD exposes two revolute knob joints (axis Z, range
# 0..180 deg). Driven by a *simple friction-based* implicit actuator
# (stiffness=0 so there is no spring-back setpoint; small Coulomb joint
# friction holds the knob wherever the hand leaves it) -- the same
# friction-detent scheme as the ``biamnaul_articulations`` hinges.
STOVE_KNOB_JOINTS = ("RevoluteJoint_cooker007_down", "RevoluteJoint_cooker007_down1")
# The robot's-left knob (world +Y side, nearest the left burner): cooker007
# joint ``RevoluteJoint_cooker007_down1`` (world y ~= -0.152 vs the right
# knob ``..._down`` at y ~= -0.288). Used as the single "turn on" target so
# success requires turning the LEFT knob; both knobs stay actuated below.
STOVE_LEFT_KNOB_JOINT = "RevoluteJoint_cooker007_down1"
STOVE_KNOB_STATIC_FRICTION = 0.1
STOVE_KNOB_DYNAMIC_FRICTION = 0.1
STOVE_KNOB_DAMPING = 0.1
# Rotation (rad) from the knob's init pose that counts as "stove on".
# ~45 deg; the knob can travel a full 180 deg.
STOVE_KNOB_ON_DISP = math.radians(45.0)

# Vertical lift of the pour-goal indicator above the pot's root (so the
# green sphere sits over the pot's mouth, not its base).
POUR_GOAL_Z_ABOVE_POT = 0.15

# Bottle and pot offsets, same convention as STOVE_OFFSET_FROM_COUNTER:
# in cabinet local frame, measured from the counter centre. z entry is
# usually 0 (prop sits flush; runtime adds its half_height). y splits
# the props apart so they don't overlap each other or the stove burner.
# Moved onto the (now-removed) potato bowl's xy spot -- (-1, -0.8) in cabinet
# local -- so the bottle sits in the near ingredient-prep area and is easier to
# reach. z keeps the bottle's flush-on-counter value. See
# ``BOWL_POTATO_OFFSET_FROM_COUNTER`` below for the matching xy.
BOTTLE_OFFSET_FROM_COUNTER = (-1.0, -0.8, -0.05)
POT_OFFSET_FROM_COUNTER = (-2.4, -0.7, -0.07)

# ---- Tomato soup can (second pour object) ----
# Same asset and pour configuration as ``pour_can_cfg``. It deliberately
# keeps can-local pour params separate from the bottle because the authored
# asset frames are different.
CAN_USD_PATH = str(YCB_DIR / "005_tomato_soup_can" / "tomato_soup_can.usd")
CAN_SCALE = (1.0, 1.0, 1.0)
CAN_MASS = 0.3
CAN_HALF_HEIGHT_EST = 0.0
# 90° rotation about -X so the can lies on its side at spawn, spout
# pointing along world -Y. Matches pour_can_cfg.
CAN_ROT_INIT = (0.707107, -0.707107, 0.0, 0.0)
CAN_FORBIDDEN_ZONE_ROT_OFFSET = (
    math.cos(math.pi / 4.0),
    math.sin(math.pi / 4.0),
    0.0,
    0.0,
)
CAN_POUR_AXIS_LOCAL = (0.0, -1.0, 0.0)
# Local-frame spout offset for the can lip.
CAN_POUR_SPOUT_LOCAL_OFFSET = (0.0, -0.0475, 0.0)
# Cabinet-local placement -- right next to the meat bowl
# (``BOWL_OFFSET_FROM_COUNTER`` ~ (-1.25, -0.6, -0.05)), so it sits in
# the same ingredient-prep area on the counter.
CAN_OFFSET_FROM_COUNTER = (-1, -0.6, 0.0)

# Forbidden zone shared between bottle and can: matches pour_can_cfg.
# Cylinder anchored in the object's local frame near the spout rim, so
# the fingertips can't sit inside the spout / lip when the pour gate
# fires (otherwise tipping the can with a finger blocking the opening
# would count as success).
SHARED_POUR_FORBIDDEN_ZONE = ForbiddenZone(
    kind="cylinder",
    center=(0.0, -0.048, 0.0),
    radius=0.03,
    half_height=0.005,
    rotation_offset=CAN_FORBIDDEN_ZONE_ROT_OFFSET,
)

# ---- Bowl + meat cube props (ingredient prep on the counter) ----
BOWL_USD_PATH = str(SYNTHESIS_COOKING_DIR / "CWCMixingBowl1" / "model_CWCMixingBowl1_69323.usd")
BOWL_SCALE = (0.7, 0.7, 0.7)
BOWL_MASS = 0.2
# Half the bowl's outer height *in the asset's authored frame* (before
# ``BOWL_SCALE`` is applied). ``__post_init__`` multiplies by
# ``bowl_scale[2]`` so changing the scale auto-updates the flush-on-
# counter spawn pose and the meat-cube drop height.
BOWL_HALF_HEIGHT_EST = 0.02  # TODO: measure in viewport (unscaled)
BOWL_ROT_INIT = (1.0, 0.0, 0.0, 0.0)
# Bowl offset from the counter centre in cabinet local frame. Same
# convention as the other props; tune so the bowl lands on a free part
# of the counter (clear of the stove, pot, and bottle).
BOWL_OFFSET_FROM_COUNTER = (-1.25, -0.6, -0.05)  # TODO: tune

MEAT_CUBE_USD_PATH = str(SYNTHESIS_COOKING_DIR / "meat_cube" / "meat_cube.usd")
MEAT_CUBE_SCALE = (2.0, 2.0, 2.0)  # TODO: tune to fit the bowl
MEAT_CUBE_MASS = 0.03
# Half-edge of the cube. Used (a) to space the cubes apart on spawn so
# they don't interpenetrate and (b) implicitly via the convex-hull
# collider sized from the mesh.
MEAT_CUBE_HALF_EXTENT_EST = 0.0  # TODO: measure in viewport
NUM_MEAT_CUBES = 4

POTATO_USD_PATH = str(SYNTHESIS_COOKING_DIR / "potato" / "potato.usd")
POTATO_SCALE = (2.0, 2.0, 2.0)  # TODO: tune to fit the bowl
POTATO_MASS = 0.05
POTATO_HALF_EXTENT_EST = 0.0  # TODO: measure in viewport
NUM_POTATOES = 4

CARROT_USD_PATH = str(SYNTHESIS_COOKING_DIR / "carrot" / "carrot.usd")
CARROT_SCALE = (2.0, 2.0, 2.0)  # TODO: tune to fit the bowl
CARROT_MASS = 0.04
CARROT_HALF_EXTENT_EST = 0.0  # TODO: measure in viewport
NUM_CARROTS = 4

# In-plane offsets from the bowl xy for each ingredient piece, in cabinet
# local frame. A 4-arm "+" pattern centred on the bowl, with arm length
# chosen so the pieces have a clear gap between them (≈ 2x half-extent +
# slack) and small enough to fit inside the bowl's interior cavity
# *after* the bowl scale is applied. Bump down if pieces spill over the
# rim, bump up (with care) for fewer collisions during free-fall.
_INGREDIENT_R = 0.0275
INGREDIENT_XY_OFFSETS_4 = (
    (_INGREDIENT_R, 0.0),
    (-_INGREDIENT_R, 0.0),
    (0.0, _INGREDIENT_R),
    (0.0, -_INGREDIENT_R),
)
# Drop height above the bowl's top rim. Pieces spawn at
# bowl_world_z + 2*bowl_half_height + this, then free-fall into the
# cavity under gravity. Raise if pieces land outside the bowl, lower
# if they clip through the rim before settling.
INGREDIENT_DROP_HEIGHT = 0.04

# Bowl positions on the counter (cabinet local). The meat bowl keeps its
# existing offset; potato and carrot bowls cluster around it so all three
# sit in the ingredient-prep area away from the stove + pot + bottle.
# 0.20 m spacing is enough at BOWL_SCALE = 0.7 (visible bowl ≈ 0.15 m
# across) to keep the rims clear of one another.
BOWL_MEAT_OFFSET_FROM_COUNTER = BOWL_OFFSET_FROM_COUNTER  # back-compat alias
BOWL_POTATO_OFFSET_FROM_COUNTER = (-1, -0.8, -0.05)  # +y of meat
BOWL_CARROT_OFFSET_FROM_COUNTER = (-1.25, -0.8, -0.05)  # -y of meat


@dataclass(frozen=True)
class BowlIngredientSpec:
    """Recipe for one bowl on the counter plus the ingredient pieces
    dropped into it.

    The ``name`` field doubles as the scene-attr prefix: the bowl is
    registered as ``scene.bowl_{name}`` and each ingredient piece as
    ``scene.{name}_{i}``. Cabinet-relative positioning is handled by
    ``__post_init__`` via ``_prop_world_pos`` (the same helper the stove,
    pot and bottle use), so moving / rotating the cabinet brings every
    bowl with it.
    """

    name: str
    bowl_usd_path: str
    bowl_scale: tuple[float, float, float]
    bowl_mass: float
    bowl_half_height: float
    bowl_init_rot: tuple[float, float, float, float]
    bowl_offset_from_counter: tuple[float, float, float]
    ingredient_usd_path: str
    ingredient_scale: tuple[float, float, float]
    ingredient_mass: float
    ingredient_half_extent: float
    ingredient_init_rot: tuple[float, float, float, float]
    num_ingredients: int
    ingredient_xy_offsets: tuple[tuple[float, float], ...]
    ingredient_drop_height: float


def _make_bowl_spec(
    *,
    name: str,
    bowl_offset_from_counter: tuple[float, float, float],
    ingredient_usd_path: str,
    ingredient_scale: tuple[float, float, float],
    ingredient_mass: float,
    ingredient_half_extent: float,
    num_ingredients: int,
) -> BowlIngredientSpec:
    """Build a spec using the shared CWCMixingBowl1 bowl asset.

    All three current specs share the same bowl USD / scale / mass /
    init-rot, so this convenience cuts the per-spec boilerplate down to
    the parts that actually vary (name, position, ingredient).
    """
    return BowlIngredientSpec(
        name=name,
        bowl_usd_path=BOWL_USD_PATH,
        bowl_scale=BOWL_SCALE,
        bowl_mass=BOWL_MASS,
        bowl_half_height=BOWL_HALF_HEIGHT_EST,
        bowl_init_rot=BOWL_ROT_INIT,
        bowl_offset_from_counter=bowl_offset_from_counter,
        ingredient_usd_path=ingredient_usd_path,
        ingredient_scale=ingredient_scale,
        ingredient_mass=ingredient_mass,
        ingredient_half_extent=ingredient_half_extent,
        ingredient_init_rot=(1.0, 0.0, 0.0, 0.0),
        num_ingredients=num_ingredients,
        ingredient_xy_offsets=INGREDIENT_XY_OFFSETS_4,
        ingredient_drop_height=INGREDIENT_DROP_HEIGHT,
    )


MEAT_BOWL_SPEC = _make_bowl_spec(
    name="meat",
    bowl_offset_from_counter=BOWL_MEAT_OFFSET_FROM_COUNTER,
    ingredient_usd_path=MEAT_CUBE_USD_PATH,
    ingredient_scale=MEAT_CUBE_SCALE,
    ingredient_mass=MEAT_CUBE_MASS,
    ingredient_half_extent=MEAT_CUBE_HALF_EXTENT_EST,
    num_ingredients=NUM_MEAT_CUBES,
)
POTATO_BOWL_SPEC = _make_bowl_spec(
    name="potato",
    bowl_offset_from_counter=BOWL_POTATO_OFFSET_FROM_COUNTER,
    ingredient_usd_path=POTATO_USD_PATH,
    ingredient_scale=POTATO_SCALE,
    ingredient_mass=POTATO_MASS,
    ingredient_half_extent=POTATO_HALF_EXTENT_EST,
    num_ingredients=NUM_POTATOES,
)
CARROT_BOWL_SPEC = _make_bowl_spec(
    name="carrot",
    bowl_offset_from_counter=BOWL_CARROT_OFFSET_FROM_COUNTER,
    ingredient_usd_path=CARROT_USD_PATH,
    ingredient_scale=CARROT_SCALE,
    ingredient_mass=CARROT_MASS,
    ingredient_half_extent=CARROT_HALF_EXTENT_EST,
    num_ingredients=NUM_CARROTS,
)
# Only the meat bowl is active for now. The potato and carrot specs stay
# defined above so they can be re-added to this tuple later without re-deriving
# their offsets / ingredient configs.
BOWL_INGREDIENT_SPECS = (MEAT_BOWL_SPEC,)


def _cabinet_local_to_world(
    local_xyz: tuple[float, float, float],
    *,
    cabinet_pos: tuple[float, float, float] = KITCHEN_CABINET_INIT_POS,
    cabinet_quat: tuple[float, float, float, float] = KITCHEN_CABINET_INIT_ROT,
) -> tuple[float, float, float]:
    """Apply cabinet rotation + translation to a cabinet-local 3D offset.

    Returns ``cabinet_pos + R(cabinet_quat) * local_xyz`` as a plain
    tuple. Used at configclass init time to convert intuitive
    cabinet-relative prop offsets into world spawn poses, so moving or
    rotating the cabinet brings every counter prop with it.
    """
    local_t = torch.tensor(local_xyz, dtype=torch.float32).unsqueeze(0)
    quat_t = torch.tensor(cabinet_quat, dtype=torch.float32).unsqueeze(0)
    world_off = quat_apply(quat_t, local_t).squeeze(0)
    return (
        cabinet_pos[0] + float(world_off[0]),
        cabinet_pos[1] + float(world_off[1]),
        cabinet_pos[2] + float(world_off[2]),
    )


@clone
def spawn_synthesis_rigid_preserve_collision(prim_path, cfg, *args, **kwargs):
    """Spawn a synthesis USD as a SINGLE rigid body, keeping the collision
    authoring already baked into the USD.

    The pot / bowl USDs ship with the right MeshCollisionAPI on their
    mesh prims (authored by the offline ``scripts`` pipeline, e.g.
    ``author_usd_mesh_collision.py --approximation convexDecomposition``
    or ``--approximation sdf``). Touching those APIs at spawn time --
    even just applying ``CollisionAPI`` to the root xform -- causes
    IsaacLab's ``@apply_nested`` walk in ``modify_collision_properties``
    to stop at the root and let PhysX fall back to convexHull on the
    inner meshes, silently undoing the authored approximation.

    This helper therefore does *only* the hierarchy cleanup needed to
    spawn the asset as a single rigid body:

      1. Strip ``RigidBodyAPI`` / ``PhysxRigidBodyAPI`` /
         ``ArticulationRootAPI`` / ``PhysxArticulationAPI`` from every
         child prim (the root keeps its rigid body).
      2. Delete inner ``UsdPhysics.Joint`` prims (no internal
         articulation; otherwise PhysX fires "missing xformstack reset"
         errors when child bodies have parent rigid bodies).
      3. Re-apply the cfg's root ``rigid_props`` and ``mass_props``.

    No CollisionAPI / MeshCollisionAPI is added, removed, or modified.
    Whatever the USD authored on disk is what PhysX will use.
    """
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import PhysxSchema, UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()

    joint_paths = []
    stack = list(prim.GetChildren())
    while stack:
        sub = stack.pop()
        if sub.IsA(UsdPhysics.Joint) or sub.GetTypeName().endswith("Joint"):
            joint_paths.append(sub.GetPath())
        if sub.HasAPI(UsdPhysics.ArticulationRootAPI):
            sub.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if sub.HasAPI(PhysxSchema.PhysxArticulationAPI):
            sub.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if sub.HasAPI(UsdPhysics.RigidBodyAPI):
            sub.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if sub.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            sub.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        stack.extend(sub.GetChildren())

    for joint_path in joint_paths:
        stage.RemovePrim(joint_path)

    prim_path_resolved = prim.GetPath().pathString
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path_resolved, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path_resolved, cfg.mass_props)

    print(
        f"[cook_foods] {prim_path_resolved}: stripped inner rigid bodies / joints; "
        "USD-authored collision APIs left untouched."
    )
    return prim


def _build_stove_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> ArticulationCfg:
    cleaned_usd = ensure_single_articulation_root(STOVE_USD_PATH)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Stove",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cleaned_usd,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                max_effort=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos={name: 0.0 for name in STOVE_KNOB_JOINTS},
        ),
        actuators={
            # Simple friction-based detent so the hand can rotate a knob and
            # it stays put (stiffness=0 -> no spring-back, small Coulomb
            # friction holds it). Mirrors the ``biamnaul_articulations``
            # hinge actuators (implicit, not IdealPD, so the friction
            # actually reaches the PhysX joint). Required for the "turn on
            # stove" step -- without an actuator the knob won't hold.
            "stove_knobs": ImplicitActuatorCfg(
                joint_names_expr=list(STOVE_KNOB_JOINTS),
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=STOVE_KNOB_DAMPING,
                friction=STOVE_KNOB_STATIC_FRICTION,
                dynamic_friction=STOVE_KNOB_DYNAMIC_FRICTION,
                armature=0.005,
            ),
        },
    )


def _build_kitchen_cabinet_cfg() -> ArticulationCfg:
    cleaned_usd = ensure_single_articulation_root(KITCHEN_CABINET_USD_PATH)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/KitchenCabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cleaned_usd,
            scale=KITCHEN_CABINET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                max_effort=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=KITCHEN_CABINET_INIT_POS,
            rot=KITCHEN_CABINET_INIT_ROT,
        ),
        actuators={},
    )


POT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pot",
    spawn=sim_utils.UsdFileCfg(
        # The pot USD already authors its own MeshCollisionAPI (set
        # offline via the scripts pipeline). Use the
        # preserve-collision spawner so the hierarchy cleanup runs but
        # the on-disk colliders are kept as-is. No collision_props on
        # the cfg -- IsaacLab would otherwise apply CollisionAPI to the
        # root xform and let PhysX fall back to convexHull.
        func=spawn_synthesis_rigid_preserve_collision,
        usd_path=POT_USD_PATH,
        scale=POT_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,  # dynamic -- robot picks it up + places it
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=POT_MASS),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),  # placeholder; overridden in __post_init__
        rot=POT_ROT_INIT,
    ),
)


def _pour_gate_for_object(
    obj: RigidObject,
    pot: RigidObject,
    *,
    min_lift_height: float,
    tilt_threshold_rad: float,
    tilt_axis_local: tuple[float, float, float],
    world_axis: tuple[float, float, float],
    spout_local_offset: tuple[float, float, float],
    goal_xy_threshold: float,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:
    """Per-object pour gate: lift + tilt + spout-over-pot.

    Shared by every pour object via :func:`_stage_pour_object` so the
    bottle, the can, and any future pour object all use exactly the same
    success criterion (and any tuning happens in one place).
    """
    # Same convention as ``mdp.utils.root_height_delta``: env grids are
    # horizontal so z is identical in env-local and world frames -- no
    # need to add env_origins for the lift delta.
    lifted = (obj.data.root_pos_w[:, 2] - obj.data.default_root_state[:, 2]) >= min_lift_height

    tilt = axis_tilt_angle(
        obj.data.root_quat_w,
        axis_local=tilt_axis_local,
        world_axis=world_axis,
    )
    tilted = tilt >= tilt_threshold_rad

    spout_local = torch.tensor(spout_local_offset, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
    spout_w = obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, spout_local)
    d_spout = spout_w[:, :2] - pot.data.root_pos_w[:, :2]
    spout_over_pot = d_spout.pow(2).sum(dim=-1) <= float(goal_xy_threshold) ** 2

    return lifted & tilted & spout_over_pot


def _ingredient_in_pot_gate(
    ingredient: RigidObject,
    pot: RigidObject,
    *,
    zone_center_pot_local: tuple[float, float, float],
    zone_radius: float,
    zone_half_height: float,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:
    """Return True per env if the ingredient's root is inside a cylinder
    fixed in the pot's local frame.

    The zone is described in pot-LOCAL coordinates (it tracks the pot's
    pose, so moving / tilting the pot moves the acceptance region with
    it). Inversely-shaped vs. a forbidden zone: the ingredient must lie
    INSIDE for the gate to fire. Same shape encoding as
    ``ForbiddenZone(kind="cylinder", ...)``.
    """
    rel_world = ingredient.data.root_pos_w - pot.data.root_pos_w
    rel_pot_local = quat_apply_inverse(pot.data.root_quat_w, rel_world)
    center_t = torch.tensor(zone_center_pot_local, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
    rel_zone = rel_pot_local - center_t
    xy_inside = rel_zone[:, :2].pow(2).sum(dim=-1) <= float(zone_radius) ** 2
    z_inside = rel_zone[:, 2].abs() <= float(zone_half_height)
    return xy_inside & z_inside


# Stage-graph key for the cook-foods long-horizon success machine. The graph
# itself is (re)registered per cfg instance in ``__post_init__`` (override=True)
# so the per-instance threshold fields below feed the stage predicates.
COOK_FOODS_STAGE_GRAPH_KEY = "long_horizon.cook_foods"


def _pot_on_stove_gate(
    env: ManagerBasedRLEnv,
    *,
    stove_cfg: SceneEntityCfg,
    pot_cfg: SceneEntityCfg,
    pot_on_stove_local_offset: tuple[float, float, float],
    pot_xy_threshold: float,
    pot_flat_threshold_rad: float,
) -> torch.Tensor:
    """Live gate: pot xy on the burner AND pot lying roughly flat.

    Burner xy = ``stove.root_pos_w + R(stove.root_quat_w) * offset`` so it
    tracks the stove's live world pose.
    """
    pot: RigidObject = env.scene[pot_cfg.name]
    stove = env.scene[stove_cfg.name]
    device = env.device
    num_envs = env.num_envs

    burner_local = (
        torch.tensor(pot_on_stove_local_offset, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
    )
    burner_w = stove.data.root_pos_w + quat_apply(stove.data.root_quat_w, burner_local)
    d_pot = pot.data.root_pos_w[:, :2] - burner_w[:, :2]
    pot_at_burner = d_pot.pow(2).sum(dim=-1) <= float(pot_xy_threshold) ** 2

    pot_tilt = axis_tilt_angle(
        pot.data.root_quat_w,
        axis_local=(0.0, 0.0, 1.0),
        world_axis=(0.0, 0.0, 1.0),
    )
    pot_flat = pot_tilt <= pot_flat_threshold_rad
    return pot_at_burner & pot_flat


def _ingredients_in_pot_gate(
    env: ManagerBasedRLEnv,
    *,
    pot_cfg: SceneEntityCfg,
    ingredient_cfgs: list[SceneEntityCfg],
    zone_center: tuple[float, float, float],
    zone_radius: float,
    zone_half_height: float,
) -> torch.Tensor:
    """Live gate: every ingredient piece inside the pot's interior cylinder."""
    pot: RigidObject = env.scene[pot_cfg.name]
    device = env.device
    num_envs = env.num_envs

    ok: torch.Tensor | None = None
    for ing_cfg in ingredient_cfgs:
        ingredient: RigidObject = env.scene[ing_cfg.name]
        gate = _ingredient_in_pot_gate(
            ingredient,
            pot,
            zone_center_pot_local=zone_center,
            zone_radius=zone_radius,
            zone_half_height=zone_half_height,
            device=device,
            num_envs=num_envs,
        )
        ok = gate if ok is None else (ok & gate)
    if ok is None:
        # No ingredients configured -- treat as satisfied.
        ok = torch.ones(num_envs, dtype=torch.bool, device=device)
    return ok


def _stage_pour_object(
    env: ManagerBasedRLEnv,
    *,
    pour_cfg: SceneEntityCfg,
    pot_cfg: SceneEntityCfg,
    min_lift_height: float,
    tilt_threshold_rad: float,
    tilt_axis_local: tuple[float, float, float],
    world_axis: tuple[float, float, float],
    spout_local_offset: tuple[float, float, float],
    goal_xy_threshold: float,
) -> torch.Tensor:
    """Stage predicate for a single pour object (lift + tilt + spout-over-pot).

    Used as a *latched* stage in the cook-foods graph, so the transient act
    of pouring only has to happen once during the episode.
    """
    obj: RigidObject = env.scene[pour_cfg.name]
    pot: RigidObject = env.scene[pot_cfg.name]
    return _pour_gate_for_object(
        obj,
        pot,
        min_lift_height=min_lift_height,
        tilt_threshold_rad=tilt_threshold_rad,
        tilt_axis_local=tilt_axis_local,
        world_axis=world_axis,
        spout_local_offset=spout_local_offset,
        goal_xy_threshold=goal_xy_threshold,
        device=env.device,
        num_envs=env.num_envs,
    )


def _stage_turn_on_stove(
    env: ManagerBasedRLEnv,
    *,
    stove_cfg: SceneEntityCfg,
    pot_cfg: SceneEntityCfg,
    ingredient_cfgs: list[SceneEntityCfg],
    knob_asset_cfg: SceneEntityCfg,
    knob_threshold_rad: float,
    pot_on_stove_local_offset: tuple[float, float, float],
    pot_xy_threshold: float,
    pot_flat_threshold_rad: float,
    pot_interior_zone_center: tuple[float, float, float],
    pot_interior_zone_radius: float,
    pot_interior_zone_half_height: float,
) -> torch.Tensor:
    """Terminal stage predicate: stove turned on with pot + ingredients still set.

    Returns (per env) the AND of three *live* conditions, evaluated every
    step:
      - the (left) stove knob in ``knob_asset_cfg`` rotated
        >= ``knob_threshold_rad`` from init,
      - the pot is currently on the burner and flat,
      - every ingredient is currently inside the pot.

    The pour prerequisites are NOT checked here -- they are wired as latched
    ``deps`` of this stage in the graph, so this fires only once both pours
    have already happened ("stove on last"). Because the graph is evaluated
    with ``persistent=True``, the latched pour flags gate this terminal stage
    while pot/ingredients stay live.
    """
    knob_on = mdp.joint_relative_move(
        env,
        threshold=knob_threshold_rad,
        asset_cfg=knob_asset_cfg,
        mode="displacement",
        op=">=",
        reduce="any",
    )
    pot_ok = _pot_on_stove_gate(
        env,
        stove_cfg=stove_cfg,
        pot_cfg=pot_cfg,
        pot_on_stove_local_offset=pot_on_stove_local_offset,
        pot_xy_threshold=pot_xy_threshold,
        pot_flat_threshold_rad=pot_flat_threshold_rad,
    )
    ingredients_ok = _ingredients_in_pot_gate(
        env,
        pot_cfg=pot_cfg,
        ingredient_cfgs=ingredient_cfgs,
        zone_center=pot_interior_zone_center,
        zone_radius=pot_interior_zone_radius,
        zone_half_height=pot_interior_zone_half_height,
    )
    return knob_on & pot_ok & ingredients_ok


@configclass
class CookFoodsEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Long-horizon cook task: pot on stove, ingredients + pours into pot, stove on.

    Built on the pour template ( ``object`` = small water bottle ) with an
    additional ``pot`` rigid prop carried onto the stove burner, ingredient
    bowls, a tomato-soup ``can``, and an articulated stove with a turnable
    knob. Success is driven by a strict, persistent :class:`mdp.StageGraphSpec`
    (see ``__post_init__``):

      - ``pour_water`` / ``pour_can`` -- latched prerequisites (the transient
        pour only has to happen once during the episode),
      - ``turn_on_stove`` -- terminal stage that fires once both pours are
        latched AND, *live* at that instant, a stove knob is turned, the pot
        is on the burner, and every ingredient is inside the pot.

    The four prerequisites have no ordering among themselves; turning on the
    stove is always the last step. The pour goal indicator is slaved to the
    pot, so moving the pot first is the natural way to make the pour usable.
    """

    # ---- Pour object: small water bottle ----
    usd_path: str = BOTTLE_USD_PATH
    object_mass: float = BOTTLE_MASS
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    object_half_height: float = BOTTLE_HALF_HEIGHT_EST
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    # Inherited base computes ``scene.object.init_state.pos`` from these
    # offsets + the (invisible) cuboid table pos. We override the bottle's
    # pos in ``__post_init__`` from ``bottle_offset_from_counter`` instead,
    # so leave the inherited fields at 0.
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    # Offset from the cabinet's counter centre to the bottle's spawn pose,
    # expressed in cabinet local frame. ``__post_init__`` rotates it by
    # the cabinet quat and adds the cabinet pos, so moving or rotating
    # the cabinet brings the bottle with it.
    bottle_offset_from_counter: tuple[float, float, float] = BOTTLE_OFFSET_FROM_COUNTER
    object_init_rot: tuple[float, float, float, float] = BOTTLE_ROT_INIT
    object_collision_enabled: bool = False
    # Per-reset random offsets around the bottle's init xy on the prep
    # side of the counter. Zeroed for now (debugging) -- restore e.g.
    # (-0.15, 0.15) / (-0.10, 0.10) once the task is tuned.
    object_reset_x_range: tuple[float, float] = (0.0, 0.0)
    object_reset_y_range: tuple[float, float] = (0.0, 0.0)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    # Forbidden zones for the BOTTLE (the inherited base wires this onto
    # ``scene.object``). Matches the can's pour-can-cfg zone so both
    # pour objects share the same fingertip-clearance requirement.
    forbidden_zones: tuple[ForbiddenZone, ...] = (SHARED_POUR_FORBIDDEN_ZONE,)

    # ---- Pour-template parameters (shared by bottle AND can) ----
    # The ``pour_water`` and ``pour_can`` stages apply these same values
    # (via ``common_pour_params`` in ``__post_init__``), so both the bottle
    # and the can see the same lift / tilt / spout-over-pot thresholds.
    pour_lift_height: float = 0.10
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = BOTTLE_POUR_AXIS_LOCAL
    pour_world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_tilt_ge: bool = True
    pour_goal_xy_threshold: float | None = 0.10
    pour_goal_object_local_offset: tuple[float, float, float] = BOTTLE_POUR_SPOUT_LOCAL_OFFSET
    pour_plane_axis_local: tuple[float, float, float] | None = None
    pour_plane_angle_threshold_rad: float | None = None
    pour_show_progress_marker: bool = False

    # ---- Tomato soup can: second pour object, same pour params ----
    can_usd_path: str = CAN_USD_PATH
    can_scale: tuple[float, float, float] = CAN_SCALE
    can_mass: float = CAN_MASS
    can_half_height: float = CAN_HALF_HEIGHT_EST
    can_init_rot: tuple[float, float, float, float] = CAN_ROT_INIT
    can_offset_from_counter: tuple[float, float, float] = CAN_OFFSET_FROM_COUNTER
    can_pour_axis_local: tuple[float, float, float] = CAN_POUR_AXIS_LOCAL
    can_pour_goal_object_local_offset: tuple[float, float, float] = CAN_POUR_SPOUT_LOCAL_OFFSET
    # Per-reset random offsets around the can's spawn xy. Zeroed for now
    # (debugging) -- restore e.g. (-0.15, 0.15) / (-0.10, 0.10) when tuned.
    can_reset_x_range: tuple[float, float] = (0.0, 0.0)
    can_reset_y_range: tuple[float, float] = (0.0, 0.0)

    # ---- Pot prop ----
    pot_usd_path: str = POT_USD_PATH
    pot_scale: tuple[float, float, float] = POT_SCALE
    pot_mass: float = POT_MASS
    pot_half_height: float = POT_HALF_HEIGHT_EST
    pot_init_rot: tuple[float, float, float, float] = POT_ROT_INIT
    # Offset from the cabinet's counter centre to the pot's spawn pose,
    # in cabinet local frame. Same convention as
    # ``bottle_offset_from_counter``.
    pot_offset_from_counter: tuple[float, float, float] = POT_OFFSET_FROM_COUNTER
    # Zeroed for now (debugging) -- restore e.g. (-0.15, 0.15) / (-0.10, 0.10).
    pot_reset_x_range: tuple[float, float] = (0.0, 0.0)
    pot_reset_y_range: tuple[float, float] = (0.0, 0.0)
    pot_min_bottle_distance: float = 0.15
    # XY tolerance (m) for "pot is on the burner". Tighten for stricter
    # placement, loosen if the pot rim is large.
    pot_xy_threshold: float = 0.08
    # Maximum tilt of pot local +Z from world +Z to count as flat.
    pot_flat_threshold_rad: float = math.radians(15)

    # ---- Pot interior zone ("ingredients must end up inside the pot") ----
    # Cylinder anchored in the pot's local frame; the success function
    # checks that every ingredient's root falls inside it. The zone
    # tracks the pot pose, so moving / tilting the pot moves the
    # acceptance region with it.
    # Defaults assume the pot's local +Z is the cavity-opening direction
    # and the asset origin is near the pot's bottom centre. Tune from
    # the viewport: widen ``radius`` if pieces graze the rim, raise
    # ``half_height`` if the cavity is deeper, shift ``center`` z if
    # the pot's local origin isn't at the base.
    pot_interior_zone_center: tuple[float, float, float] = (0.0, 0.0, 0.05)
    pot_interior_zone_radius: float = 0.12
    pot_interior_zone_half_height: float = 0.08

    # ---- Bowls + ingredient pieces (one bowl per recipe spec) ----
    # Each spec drops N ingredient pieces into one bowl on the counter.
    # ``__post_init__`` iterates this tuple and registers per-bowl scene
    # attrs ``bowl_{spec.name}`` and per-piece attrs
    # ``{spec.name}_{i}``. Add a new spec to add another bowl.
    bowl_ingredient_specs: tuple[BowlIngredientSpec, ...] = BOWL_INGREDIENT_SPECS

    # ---- Stove prop ----
    stove_usd_path: str = STOVE_USD_PATH
    stove_scale: tuple[float, float, float] = STOVE_SCALE
    # Offset from the cabinet's counter centre to the stove's spawn pose,
    # in cabinet local frame. Same convention as the other props; z
    # entry typically negative to embed the stove root under the counter
    # so the visible body lands on top.
    stove_offset_from_counter: tuple[float, float, float] = STOVE_OFFSET_FROM_COUNTER
    # World-frame stove rotation. Not composed with the cabinet rotation
    # -- if you rotate the cabinet and want the stove to follow visually,
    # update this value too.
    stove_init_rot: tuple[float, float, float, float] = STOVE_INIT_ROT
    # Stove-LOCAL offset from the stove root xy to the burner xy. The
    # success function rotates this by the stove's live world quat, so
    # the burner tracks any stove orientation/randomization.
    pot_on_stove_local_offset: tuple[float, float, float] = POT_ON_STOVE_LOCAL_OFFSET
    pour_goal_z_above_pot: float = POUR_GOAL_Z_ABOVE_POT
    # Knob rotation (rad) from init that counts as "stove on" (final step).
    stove_knob_on_disp: float = STOVE_KNOB_ON_DISP
    # Per-reset stove randomization (offsets from spawn pose). Defaults to
    # fixed; widen if you want the burner location to move across episodes.
    stove_reset_x_range: tuple[float, float] = (0.0, 0.0)
    stove_reset_y_range: tuple[float, float] = (0.0, 0.0)
    stove_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    # ---- Robot wrist init (world-frame palm pose) ----
    # Applied via ``set_robot_wrist_init_world_pos`` so the same intent works
    # across embodiments (floating single/bimanual set translation joints;
    # UR10e re-IKs). The cabinet counter surface sits at world z ~= 0.85, so
    # the palm must start above it -- the bimanual default (0.8) spawns the
    # hands inside the counter. Raise ``...world_z`` if fingers still graze
    # the counter at spawn.
    robot_init_palm_world_x: float = -0.40
    robot_init_palm_world_y: float = 0.3
    robot_init_palm_world_z: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.terminations.object_out_of_bound = None

        # Pour-goal marker: kept as a (kinematic, non-colliding) scene entity so
        # the success_marker reset / follow-to-pot events still have a target,
        # but rendered INVISIBLE -- no on-screen pour-goal cue.
        self.scene.success_marker.spawn = sim_utils.SphereCfg(
            radius=0.04,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        # Hide the cuboid table + remove its legs; the kitchen cabinet is
        # the visible furniture. The invisible cuboid still acts as a
        # support surface for the shared tabletop utilities.
        self.scene.table.spawn.visible = False
        self.scene.table.spawn.size = KITCHEN_COUNTER_SUPPORT_SIZE
        self.scene.table.init_state.pos = KITCHEN_COUNTER_SUPPORT_POS
        self.scene.table_leg_front_left = None
        self.scene.table_leg_front_right = None
        self.scene.table_leg_back_left = None
        self.scene.table_leg_back_right = None

        self.scene.kitchen_cabinet = _build_kitchen_cabinet_cfg()

        # All props on the counter are positioned relative to the cabinet's
        # counter centre in cabinet-local frame, then rotated + translated
        # by the cabinet pose. Move or rotate the cabinet and every prop
        # below follows automatically.
        counter_origin = CABINET_COUNTER_LOCAL_ORIGIN

        def _prop_world_pos(
            offset_from_counter: tuple[float, float, float],
            extra_world_z: float = 0.0,
        ) -> tuple[float, float, float]:
            """Cabinet-local (counter_origin + offset) → world position.

            ``extra_world_z`` is added to the resulting z in world frame
            after the cabinet rotation, so it always points "up" (used to
            add a prop's half-height so it lands flush on the counter).
            """
            local = (
                counter_origin[0] + offset_from_counter[0],
                counter_origin[1] + offset_from_counter[1],
                counter_origin[2] + offset_from_counter[2],
            )
            world = _cabinet_local_to_world(local)
            return (world[0], world[1], world[2] + extra_world_z)

        stove_pos = _prop_world_pos(self.stove_offset_from_counter)
        self.scene.stove = _build_stove_cfg(
            scale=self.stove_scale,
            init_rot=self.stove_init_rot,
            init_pos=stove_pos,
        )

        # ---- Bottle (object) init pose on the visible cabinet counter ----
        # The base class places the bottle relative to the (invisible)
        # cuboid table; we override here with a cabinet-relative spawn.
        # ``extra_world_z`` adds the bottle half-height so its base
        # rests on the counter regardless of cabinet z.
        bottle_init = _prop_world_pos(
            self.bottle_offset_from_counter,
            extra_world_z=self.object_half_height + self.table_clearance,
        )
        self.scene.object.init_state.pos = bottle_init

        # ---- Pot prop on the visible cabinet counter ----
        pot_init = _prop_world_pos(
            self.pot_offset_from_counter,
            extra_world_z=self.pot_half_height,
        )
        self.scene.pot = POT_CFG
        self.scene.pot.init_state.pos = pot_init
        self.scene.pot.init_state.rot = self.pot_init_rot
        self.scene.pot.spawn.scale = self.pot_scale
        if self.scene.pot.spawn.mass_props is not None:
            self.scene.pot.spawn.mass_props.mass = self.pot_mass

        # ---- Tomato soup can (second pour object) ----
        # Cabinet-relative placement so the can tracks the cabinet just
        # like the other props. ``collision_enabled=False`` keeps the
        # YCB asset's authored MeshCollisionAPI intact rather than
        # letting PhysX fall back to convexHull at the root xform.
        can_world = _prop_world_pos(
            self.can_offset_from_counter,
            extra_world_z=self.can_half_height,
        )
        can_cfg = build_object_cfg_from_usd(
            self.can_usd_path,
            mass=self.can_mass,
            scale=self.can_scale,
            init_rot=self.can_init_rot,
            collision_enabled=False,
            prim_name="Can",
        )
        can_cfg.init_state.pos = can_world
        self.scene.can = can_cfg

        # ---- Bowls + ingredient pieces ----
        # Iterate the spec list. Each spec spawns one bowl on the counter
        # and N ingredient pieces dropped into it. Bowl pose is cabinet-
        # local (rotates / translates with the cabinet); each piece is
        # placed at (bowl_xy + piece_xy_offset) in cabinet local frame,
        # then elevated above the bowl rim so it free-falls inside.
        # ``bowl_half_height`` and ``ingredient_half_extent`` are in the
        # asset's authored frame, so we scale by [2] of each prop's scale
        # to get the actual visible height.
        for spec in self.bowl_ingredient_specs:
            bowl_half_h_scaled = spec.bowl_half_height * spec.bowl_scale[2]
            ingredient_half_scaled = spec.ingredient_half_extent * spec.ingredient_scale[2]

            bowl_world = _prop_world_pos(
                spec.bowl_offset_from_counter,
                extra_world_z=bowl_half_h_scaled,
            )
            bowl_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Bowl_{spec.name.capitalize()}",
                spawn=sim_utils.UsdFileCfg(
                    # Bowl USD ships with its own MeshCollisionAPI; the
                    # preserve-collision spawner keeps it intact.
                    func=spawn_synthesis_rigid_preserve_collision,
                    usd_path=spec.bowl_usd_path,
                    scale=spec.bowl_scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=spec.bowl_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=bowl_world,
                    rot=spec.bowl_init_rot,
                ),
            )
            setattr(self.scene, f"bowl_{spec.name}", bowl_cfg)

            # Per-piece spawn pose: bowl_offset + piece_xy in cabinet
            # local, transformed through cabinet rotation, then z =
            # bowl_top_z + drop_height + ingredient_half so the base
            # clears the rim.
            ingredient_z_above_counter = 2.0 * bowl_half_h_scaled + spec.ingredient_drop_height + ingredient_half_scaled
            n_pieces = min(spec.num_ingredients, len(spec.ingredient_xy_offsets))
            for i in range(n_pieces):
                dx, dy = spec.ingredient_xy_offsets[i]
                piece_local = (
                    spec.bowl_offset_from_counter[0] + dx,
                    spec.bowl_offset_from_counter[1] + dy,
                    spec.bowl_offset_from_counter[2],
                )
                piece_world = _prop_world_pos(
                    piece_local,
                    extra_world_z=ingredient_z_above_counter,
                )
                # collision_enabled=False keeps the USD's authored
                # MeshCollisionAPI intact (the ingredient USDs ship with
                # convex-hull authored offline by the scripts
                # pipeline). Switch to True if you want PhysX's runtime
                # convex-hull fallback instead.
                piece_cfg = build_object_cfg_from_usd(
                    spec.ingredient_usd_path,
                    mass=spec.ingredient_mass,
                    scale=spec.ingredient_scale,
                    init_rot=spec.ingredient_init_rot,
                    collision_enabled=False,
                    prim_name=f"{spec.name.capitalize()}{i}",
                )
                piece_cfg.init_state.pos = piece_world
                setattr(self.scene, f"{spec.name}_{i}", piece_cfg)

        # ---- Reset events ----
        # Required ordering (event manager iterates dict insertion order):
        #   1. reset_object              -- bottle gets random xy
        #   2. reset_stove               -- small offset from spawn pose
        #   3. reset_pot                 -- random xy, kept away from bottle
        #   4. reset_success_marker      -- slaved to pot xy
        # Delete the inherited marker reset so we can re-add it last.
        if "reset_success_marker" in self.events.__dict__:
            del self.events.reset_success_marker
        self.events.reset_stove = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.stove_reset_x_range),
                    "y": list(self.stove_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": list(self.stove_reset_yaw_range),
                },
                "asset_cfg": SceneEntityCfg("stove"),
            },
        )
        # Return the stove knobs to their init (0) each episode so the
        # "turn on stove" displacement baseline is correct (otherwise a
        # knob left turned from the previous episode reads as already-on).
        self.events.reset_stove_joints = EventTerm(
            func=mdp.reset_joints_to_init,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("stove", joint_names=".*")},
        )
        self.events.reset_pot = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.pot_reset_x_range),
                    "y": list(self.pot_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
                "asset_cfg": SceneEntityCfg("pot"),
                "reference_asset_cfg": SceneEntityCfg("object"),
                "min_xy_distance": self.pot_min_bottle_distance,
            },
        )
        # The can (second pour object) had no reset event, so it persisted
        # across episodes -- reset it back to its spawn pose each reset.
        self.events.reset_can = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.can_reset_x_range),
                    "y": list(self.can_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "asset_cfg": SceneEntityCfg("can"),
            },
        )
        self.events.reset_success_marker = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("success_marker"),
                "source_cfg": SceneEntityCfg("pot"),
                "z_offset": self.pour_goal_z_above_pot,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )
        # The pour target (success_marker, and the pour-progress sphere drawn
        # on it) must track the pot as it's carried onto the stove -- not just
        # sit at the reset pose. ``interval`` with a zero range + per-env time
        # is this codebase's "every physics step" hook (cf. trash_drawer_sort
        # update_push_toggled_trash_can_lid), so re-sync the marker each step.
        self.events.follow_success_marker_to_pot = EventTerm(
            func=mdp.sync_object,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=False,
            params={
                "target_cfg": SceneEntityCfg("success_marker"),
                "source_cfg": SceneEntityCfg("pot"),
                "z_offset": self.pour_goal_z_above_pot,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )

        # ---- Reset the bowls + their ingredient pieces each episode ----
        # These are registered as scene rigid objects above but had no reset
        # event, so they persisted wherever they ended up (e.g. poured into
        # the pot) across resets. Reset each back to its spawn pose: empty
        # ``pose_range`` => zero offset from the init pose, and an omitted
        # ``velocity_range`` => zero velocity, so the pieces free-fall and
        # settle into the bowls exactly as on first spawn. One event per
        # asset (SceneEntityCfg targets a single scene entity), matching the
        # per-object reset pattern used by trash_drawer_sort.
        for spec in self.bowl_ingredient_specs:
            setattr(
                self.events,
                f"reset_bowl_{spec.name}",
                EventTerm(
                    func=mdp.reset_root_pose_uniform,
                    mode="reset",
                    params={
                        "pose_range": {},
                        "asset_cfg": SceneEntityCfg(f"bowl_{spec.name}"),
                    },
                ),
            )
            n_pieces = min(spec.num_ingredients, len(spec.ingredient_xy_offsets))
            for i in range(n_pieces):
                setattr(
                    self.events,
                    f"reset_{spec.name}_{i}",
                    EventTerm(
                        func=mdp.reset_root_pose_uniform,
                        mode="reset",
                        params={
                            "pose_range": {},
                            "asset_cfg": SceneEntityCfg(f"{spec.name}_{i}"),
                        },
                    ),
                )

        # ---- Long-horizon success: latched stage graph ----
        # Four prerequisites (any order, "can happen simultaneously"):
        #   pour_water, pour_can  -- LATCHED (the transient pour only has to
        #                            happen once during the episode), and
        #   pot-on-stove, ingredients-in-pot -- checked LIVE inside the
        #                            terminal stage.
        # The terminal ``turn_on_stove`` stage fires only once both pours are
        # latched (its deps) AND the knob is turned with the pot + ingredients
        # still in place -- i.e. turning on the stove is the last step.
        # The ingredient list is built from the bowl specs so adding a bowl
        # spec automatically expands the required ingredients.
        ingredient_cfgs = [
            SceneEntityCfg(f"{spec.name}_{i}")
            for spec in self.bowl_ingredient_specs
            for i in range(min(spec.num_ingredients, len(spec.ingredient_xy_offsets)))
        ]
        pour_goal_xy = self.pour_goal_xy_threshold if self.pour_goal_xy_threshold is not None else 1e6
        shared_pour_params = {
            "pot_cfg": SceneEntityCfg("pot"),
            "min_lift_height": self.pour_lift_height,
            "tilt_threshold_rad": self.pour_angle_rad,
            "world_axis": self.pour_world_axis,
            "goal_xy_threshold": pour_goal_xy,
        }
        cook_foods_stage_graph = mdp.StageGraphSpec(
            stages=(
                mdp.StageSpec(
                    name="pour_water",
                    func=_stage_pour_object,
                    params={
                        # the bottle (inherited pour-template ``object``)
                        "pour_cfg": SceneEntityCfg("object"),
                        "tilt_axis_local": self.pour_axis_local,
                        "spout_local_offset": self.pour_goal_object_local_offset,
                        **shared_pour_params,
                    },
                ),
                mdp.StageSpec(
                    name="pour_can",
                    func=_stage_pour_object,
                    params={
                        "pour_cfg": SceneEntityCfg("can"),  # the tomato soup can
                        "tilt_axis_local": self.can_pour_axis_local,
                        "spout_local_offset": self.can_pour_goal_object_local_offset,
                        **shared_pour_params,
                    },
                ),
                mdp.StageSpec(
                    name="turn_on_stove",
                    func=_stage_turn_on_stove,
                    params={
                        "stove_cfg": SceneEntityCfg("stove"),
                        "pot_cfg": SceneEntityCfg("pot"),
                        "ingredient_cfgs": ingredient_cfgs,
                        "knob_asset_cfg": SceneEntityCfg("stove", joint_names=[STOVE_LEFT_KNOB_JOINT]),
                        "knob_threshold_rad": self.stove_knob_on_disp,
                        "pot_on_stove_local_offset": self.pot_on_stove_local_offset,
                        "pot_xy_threshold": self.pot_xy_threshold,
                        "pot_flat_threshold_rad": self.pot_flat_threshold_rad,
                        "pot_interior_zone_center": self.pot_interior_zone_center,
                        "pot_interior_zone_radius": self.pot_interior_zone_radius,
                        "pot_interior_zone_half_height": self.pot_interior_zone_half_height,
                    },
                    deps=("pour_water", "pour_can"),
                ),
            ),
            terminal_stage="turn_on_stove",
            ordering_mode="strict",
            success_mode="substage",
        )
        mdp.register_stage_graph(COOK_FOODS_STAGE_GRAPH_KEY, cook_foods_stage_graph, override=True)
        self.terminations.success = DoneTerm(
            func=mdp.stage_success,
            params={"task_key": COOK_FOODS_STAGE_GRAPH_KEY, "persistent": True},
        )

        # Pull the wrist forward so the visible counter sits in the natural
        # reach zone, and lift it above the counter (z) so the hands don't
        # spawn inside it. World coords so the same intent works for armed
        # robots (UR10e re-IKs to the target).
        set_robot_wrist_init_world_pos(
            self,
            x=self.robot_init_palm_world_x,
            z=self.robot_init_palm_world_z,
        )
        self.xr = XrCfg(
            anchor_pos=[-0.65, 0.0, 0.1],
            anchor_rot=DEFAULT_FLOATING_SHADOW_XR_CFG.anchor_rot,
        )
        align_retargeter_wrist_origin_to_init(self)
