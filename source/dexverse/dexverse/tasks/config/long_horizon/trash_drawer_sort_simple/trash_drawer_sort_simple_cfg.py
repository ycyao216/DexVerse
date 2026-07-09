# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified long-horizon trash-can / drawer scene.

Strip-down of ``trash_drawer_sort``:
    * Drawer on the right side of the table (+y), trash bin on the left (-y).
    * One cube intended for the drawer, one cylinder intended for the trash;
      both random-spawn on the tabletop each reset.
    * Simple PD (stiffness + damping) on the drawer and trash-can joints.
      No press-to-toggle lid state machine, no contact sensors, no rewards.

Follows a scene-only structure and the simpler
``functional_manipulation/grasp_*_cfg`` configs for the env layout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from dexverse.assets import LONG_HORIZON_EXTRA_TABLE_CLEANING_DIR, SYNTHESIS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ....mdp.utils import resolve_env_ids, resolve_joint_ids
from ...floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop

# =====================================================================
# USD assets (same source as ``trash_drawer_sort``)
# =====================================================================
BEDSIDE_TABLE_USD_PATH = str(SYNTHESIS_DIR / "bedside table014" / "model_drawer.usd")
TRASH_CAN_USD_PATH = str(SYNTHESIS_DIR / "trash can006" / "model_pressthetrashcan1.usd")


# =====================================================================
# Layout (env-local, z is added in __post_init__)
# =====================================================================
DRAWER_INIT_X = 0.10
DRAWER_INIT_Y = 0.35  # +y = right side
DRAWER_INIT_ROT = (0.70710678, 0.0, 0.0, -0.70710678)
DRAWER_TABLETOP_Z_OFFSET = 0.0
BEDSIDE_TABLE_SCALE = (1.0, 1.0, 1.0)

TRASH_CAN_INIT_X = 0.00
TRASH_CAN_INIT_Y = -0.30  # -y = left side
TRASH_CAN_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
TRASH_CAN_TABLETOP_Z_OFFSET = 0.005
TRASH_CAN_SCALE = (2.0, 2.0, 2.0)

# Simple PD: small stiffness gives a soft spring back to rest; damping kills
# oscillation. Replaces the press-to-toggle state machine in the original.
DRAWER_JOINT_NAMES = ".*"
TRASH_CAN_JOINT_NAMES = ".*"
# Trash-can USD has two joints: the lid hinge and a prismatic "press-to-
# open" button with travel ~1 mm (``PrismaticJoint_pressthetrashcan1_up1``).
# The button must keep its USD default; only the lid gets the PD target and
# stiffness/damping.
TRASH_CAN_LID_JOINT_NAMES = "RevoluteJoint_pressthetrashcan1_up"
# Drawer: *simple friction-based* slide (same scheme as the
# ``biamnaul_articulations`` hinges / the cook-foods stove knob). stiffness=0
# => no spring-back target, so the drawer stays wherever the hand leaves it;
# small Coulomb joint friction holds it; ``armature`` regularizes the light
# slider so it doesn't jitter. The hand opens/closes it and the success
# check requires it pushed back closed.
DRAWER_STIFFNESS = 0.0
DRAWER_DAMPING = 1.0
DRAWER_FRICTION = 0.1
DRAWER_DYNAMIC_FRICTION = 0.1
DRAWER_ARMATURE = 0.005
# Lid PD: stiff enough that gravity + light hand contact can't move it off
# the open target, with a high enough effort cap that the drive can actually
# pull the lid back from a deliberate push. Friction adds dry resistance
# so the lid doesn't drift on casual contact.
TRASH_CAN_STIFFNESS = 50.0
TRASH_CAN_DAMPING = 3.0
TRASH_CAN_FRICTION = 0.0
TRASH_CAN_DYNAMIC_FRICTION = 0.0
TRASH_CAN_EFFORT_LIMIT = 20.0

# Spring-loaded lid + latch state machine.
# Physical travel is constrained to [OPEN, UPPER_LIMIT] = [-90°, 0°].
# We use one toggle threshold with hysteresis-by-crossing:
#   * Reset starts latched at RESET_POSITION (slightly open).
#   * Crossing TOGGLE_THRESHOLD toward 0° flips latch.
#   * latched=True  -> PD target = CLOSED_POSITION
#   * latched=False -> PD target = OPEN_POSITION
TRASH_CAN_LID_CLOSED_POSITION = 0.0
TRASH_CAN_LID_RESET_POSITION = math.radians(-2.5)
TRASH_CAN_LID_OPEN_POSITION = math.radians(-90.0)
TRASH_CAN_LID_UPPER_LIMIT = 0.0
# Must satisfy: 0 > threshold > init > -90.
TRASH_CAN_LID_TOGGLE_THRESHOLD = math.radians(-1.0)
TRASH_CAN_LID_TOGGLE_COOLDOWN_S = 0.5

# Per-reset random offsets applied to each articulation's *default* root
# pose (env-local meters / radians). Keep these small so the props don't
# wander into each other or off the table.
DRAWER_RESET_X_RANGE = (0.2, 0.4)
DRAWER_RESET_Y_RANGE = (-0.8, -0.5)
DRAWER_RESET_YAW_RANGE = (-0.3, 0.3)
TRASH_CAN_RESET_X_RANGE = (0.2, 0.5)
TRASH_CAN_RESET_Y_RANGE = (0.4, 0.6)
TRASH_CAN_RESET_YAW_RANGE = (-1.57 - 0.20, -1.57 + 0.20)

# =====================================================================
# Sortable USD objects (from ``assets/long_horizon_extra/table_cleaning``)
# =====================================================================
TABLE_CLEANING_DIR = LONG_HORIZON_EXTRA_TABLE_CLEANING_DIR
# Specs: (scene_name, prim_name, subdir, usd_filename, mass_kg, scale).
# Scene name doubles as the SceneEntityCfg key used by reset events + the
# success-condition helpers. ``scale`` is applied via ``UsdFileCfg.scale`` —
# editing the USD's own root scale is *overridden* by Isaac Lab's spawn
# pipeline (it writes its own xformOp:scale on the wrapper prim), so this
# is the only place that actually changes object size at spawn time.
_TABLE_CLEANING_DRAWER_SPECS: tuple[tuple[str, str, str, str, float, float], ...] = (
    ("advil", "Advil", "Advil", "model_Advil_69323.usd", 0.05, 1.0),
    ("blue_stapler", "BlueStapler", "blue_stapler", "stapler.usd", 0.30, 1.0),
    ("charger", "Charger", "charger001", "model_charger.usd", 0.10, 1.0),
    ("headphone", "Headphone", "headphone", "headphone.usd", 0.20, 1.0),
    ("marker", "Marker", "marker", "model_Office_Marker_B293D6D4_BlackDryEraseMarker_1_69323.usd", 0.03, 1.0),
)
_TABLE_CLEANING_TRASH_SPECS: tuple[tuple[str, str, str, str, float, float], ...] = (
    ("broken_twix", "BrokenTwix", "broken_twix", "broken_twix.usd", 0.05, 1.0),
    ("crumbled_paper", "CrumbledPaper", "crumbled_paper", "crumbled_paper.usd", 0.01, 0.5),
    ("crushed_bottle", "CrushedBottle", "crushed_bottle", "water_bottle.usd", 0.03, 0.5),
    ("paper_cup", "PaperCup", "cup", "model_papercup.usd", 0.01, 1.0),
    ("soda_can", "SodaCan", "soda_can", "soda_can.usd", 0.02, 0.5),
)

# Per-reset random pose offsets (env-local meters). Widened to spread 10
# objects across the tabletop without overlap on most resets. Yaw is also
# fully random per object.
OBJECT_SPAWN_X_RANGE = (-0.3, -0.05)
OBJECT_SPAWN_Y_RANGE = (-0.4, 0.4)
# Drop height above the tabletop — objects fall a few cm and settle on the
# table. Keeps spawning robust to mesh bounding-box differences across
# assets so we don't need per-asset z-tuning.
OBJECT_SPAWN_HEIGHT = 0.01

# Rejection-sampling thresholds for the per-reset placement (env-local
# meters). All pair distances are now derived as
# ``radius_a + radius_b + OBJECT_OBJECT_SEPARATION_MARGIN`` where ``radius``
# comes from each asset's ``<stem>.aabb.json``. The receptacles use the same
# formula with their AABBs (see ``_DRAWER_RADIUS`` / ``_TRASH_RADIUS``).
# Bump the margin if you want extra empty space; lower it for tighter pack.
OBJECT_OBJECT_SEPARATION_MARGIN: float = 0.02
SPAWN_REJECTION_ATTEMPTS: int = 1000
# Fallback per-object radius / bottom-offset used when an asset has no
# ``<stem>.aabb.json`` sibling. Generate the JSONs via
# ``scripts/asset_tools/generate_aabb_json.py`` to replace the fallback with the
# asset's real footprint.
DEFAULT_OBJECT_RADIUS: float = 0.05
DEFAULT_OBJECT_BOTTOM_OFFSET: float = 0.05
DEFAULT_CONTAINER_RADIUS: float = 0.20

# Per-object emergency radius inflation. The radius formula
# (``_aabb_horizontal_radius_about_origin``) is now origin-correct, so
# this dict should normally be empty. Add an entry only as a last-resort
# hand tuning knob — e.g. if an asset's collision proxy extends past its
# visible mesh and you don't want to re-author the USD.
OBJECT_RADIUS_MULTIPLIER: dict[str, float] = {}

# =====================================================================
# Success conditions
# =====================================================================
# Object-to-receptacle assignment derived from the table_cleaning specs.
DRAWER_OBJECT_NAMES: tuple[str, ...] = tuple(spec[0] for spec in _TABLE_CLEANING_DRAWER_SPECS)
TRASH_OBJECT_NAMES: tuple[str, ...] = tuple(spec[0] for spec in _TABLE_CLEANING_TRASH_SPECS)
ALL_TABLE_CLEANING_NAMES: tuple[str, ...] = (*DRAWER_OBJECT_NAMES, *TRASH_OBJECT_NAMES)


# ---------------------------------------------------------------------
# Per-object metadata loaded from ``<stem>.aabb.json`` next to each USD.
# Drives (a) per-object rejection-sampling radii so larger items get more
# clearance, and (b) per-object spawn z so each item's bottom sits just
# above the tabletop regardless of the asset's authored frame.
# Regenerate the JSONs with ``scripts/asset_tools/generate_aabb_json.py`` after any
# rescale / mesh edit.
# ---------------------------------------------------------------------
def _load_aabb_json(usd_path: Path) -> dict | None:
    json_path = usd_path.parent / f"{usd_path.stem}.aabb.json"
    if not json_path.is_file():
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _aabb_horizontal_radius_about_origin(aabb: dict, scale_x: float, scale_y: float) -> float:
    """Smallest circle around the asset's USD origin (0, 0) that contains
    all four horizontal AABB corners.

    Use this instead of "half the AABB diagonal" because the rejection
    sampler / debug disc / yaw rotation all happen around the asset's
    *origin*, not its centroid. For assets whose origin sits at a corner
    or edge of the geometry (very common for Meshy/Synthesis exports),
    the centroid-based formula underestimates how far the geometry
    extends from the origin — which is exactly the bug that lets
    visibly-overlapping placements through.
    """
    max_x = max(abs(aabb["min"][0]), abs(aabb["max"][0])) * scale_x
    max_y = max(abs(aabb["min"][1]), abs(aabb["max"][1])) * scale_y
    return math.sqrt(max_x**2 + max_y**2)


def _build_object_metadata() -> dict[str, dict]:
    """Build a per-scene-name metadata dict {radius, bottom_offset, height}.

    Radius is the rotation-invariant diagonal radius; bottom_offset is the
    distance from the asset origin to its lowest authored point (so spawn
    z places the bottom on the table regardless of where the USD origin
    sits). Falls back to ``DEFAULT_*`` values with a printed warning for
    any asset missing ``<stem>.aabb.json``.
    """
    metadata: dict[str, dict] = {}
    for spec_list, subdir_root in (
        (_TABLE_CLEANING_DRAWER_SPECS, "to_drawer"),
        (_TABLE_CLEANING_TRASH_SPECS, "to_trash"),
    ):
        for scene_name, _, subdir, usd_filename, _, scale in spec_list:
            usd_path = TABLE_CLEANING_DIR / subdir_root / subdir / usd_filename
            data = _load_aabb_json(usd_path)
            if data is None:
                print(
                    "[trash_drawer_sort_simple] WARN: no aabb.json for "
                    f"{usd_path}; falling back to defaults. Run "
                    "scripts/asset_tools/generate_aabb_json.py to fix."
                )
                metadata[scene_name] = {
                    "radius": DEFAULT_OBJECT_RADIUS * scale * OBJECT_RADIUS_MULTIPLIER.get(scene_name, 1.0),
                    "bottom_offset": DEFAULT_OBJECT_BOTTOM_OFFSET * scale,
                    "height": 2.0 * DEFAULT_OBJECT_BOTTOM_OFFSET * scale,
                }
                continue
            aabb = data["local_aabb"]
            # Origin-based radius (correct for off-centre USD origins). The
            # ``OBJECT_RADIUS_MULTIPLIER`` dict is now redundant for the
            # known assets — kept as an emergency hand-tuning knob but
            # defaults to 1.0.
            base_radius = _aabb_horizontal_radius_about_origin(aabb, scale, scale)
            radius_mult = OBJECT_RADIUS_MULTIPLIER.get(scene_name, 1.0)
            metadata[scene_name] = {
                "radius": base_radius * radius_mult,
                "bottom_offset": max(0.0, -aabb["min"][2]) * scale,
                "height": aabb["size"][2] * scale,
            }
    return metadata


def _container_radius(usd_path_str: str, scale_tuple: tuple[float, ...]) -> float:
    """Diagonal horizontal radius for a receptacle, using its aabb.json.

    Falls back to ``DEFAULT_CONTAINER_RADIUS`` if the JSON is missing
    (with a warning). The drawer / trash xy reset offsets are added in
    Isaac Lab's reset pipeline before this radius gates the sampler, so
    this radius only needs to cover the container's *own* footprint.
    """
    data = _load_aabb_json(Path(usd_path_str))
    if data is None:
        print(
            "[trash_drawer_sort_simple] WARN: no aabb.json for container "
            f"{usd_path_str}; using DEFAULT_CONTAINER_RADIUS. Run "
            "scripts/asset_tools/generate_aabb_json.py on the synthesis dir to fix."
        )
        return DEFAULT_CONTAINER_RADIUS * max(scale_tuple[:2])
    aabb = data["local_aabb"]
    return _aabb_horizontal_radius_about_origin(aabb, scale_tuple[0], scale_tuple[1])


_OBJECT_METADATA: dict[str, dict] = _build_object_metadata()
# Per-object horizontal radii in the order of ``ALL_TABLE_CLEANING_NAMES``.
# Used as a torch tensor inside the rejection sampler.
_OBJECT_RADII: list[float] = [_OBJECT_METADATA[name]["radius"] for name in ALL_TABLE_CLEANING_NAMES]
_DRAWER_RADIUS: float = _container_radius(BEDSIDE_TABLE_USD_PATH, BEDSIDE_TABLE_SCALE)
_TRASH_RADIUS: float = _container_radius(TRASH_CAN_USD_PATH, TRASH_CAN_SCALE)

# "Inside the receptacle" AABBs in the receptacle's own root frame
# (computed via ``quat_apply_inverse`` of the world-frame offset).
#
# Derived from per-link AABBs measured in the articulation ROOT frame via
# ``scripts/asset_tools/inspect_usd_bbox.py`` (per-link, root-frame):
#   Drawer tray ``E_drawer_59`` at the closed rest pose (joint=0, the
#     success state): x in [-0.187, 0.187], y in [-0.180, 0.138],
#     z in [0.010, 0.190] (front panel at -y). Inset ~2.5 cm for the walls,
#     start above the floor, stop just below the rim (table top at z=0.20):
OBJECT_IN_DRAWER_LOCAL_MIN: tuple[float, float, float] = (-0.16, -0.15, 0.02)
OBJECT_IN_DRAWER_LOCAL_MAX: tuple[float, float, float] = (0.16, 0.12, 0.18)
#   Trash bin: opening (lid ``E_lid_43``) radius ~0.116, cavity below the
#     closed lid at z~0.256, bin scaled by 2.0 -> x,y in [-0.137, 0.137],
#     z in [0, 0.274]. Inset to the opening, floor to just below the lid:
OBJECT_IN_TRASH_LOCAL_MIN: tuple[float, float, float] = (-0.11, -0.11, 0.0)
OBJECT_IN_TRASH_LOCAL_MAX: tuple[float, float, float] = (0.11, 0.11, 0.25)


def _aabb_to_box_zone(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> list[float]:
    """Convert an (min, max) AABB to the ``forbidden_zones_vis`` box encoding
    ``[cx, cy, cz, hx, hy, hz]`` (center + half-extents). Same frame as the
    containment check (:func:`_object_in_aabb_mask`), so the rendered box is
    exactly the success acceptance region.
    """
    center = [(lo + hi) * 0.5 for lo, hi in zip(lower, upper)]
    half = [(hi - lo) * 0.5 for lo, hi in zip(lower, upper)]
    return [*center, *half]


# "Joint at closed position" thresholds (m for the drawer slide, rad for the
# trash-can lid hinge). A sub-task only counts as success when the
# receptacle is closed AND every assigned object is inside.
DRAWER_CLOSED_THRESHOLD: float = 0.02
TRASH_LID_CLOSED_THRESHOLD: float = math.radians(2.0)

# Failure: episode terminates if any sortable object drops more than this
# many meters below the tabletop (env-local frame). Catches "fell off the
# table" cases — once an object goes over an edge gravity pulls it past
# this margin within a step or two.
OBJECT_FALLEN_Z_MARGIN: float = 0.10

# ---------------------------------------------------------------------
# Per-episode active-object subset
# ---------------------------------------------------------------------
# Isaac Lab requires a fixed scene graph, so we spawn all 10 sortable
# objects every episode and "deactivate" the unused ones by parking them
# far below the workspace (env-local park z) at a unique x offset so they
# don't stack on each other in physics-land. Active counts are sampled
# per env per reset, then the active mask is consulted by:
#   * the rejection-sampling placement (only active objects get a real xy)
#   * the success helpers (only active objects must be in their receptacle)
#   * the object-fell termination (parked objects naturally sit below the
#     fall threshold; we ignore them)
ACTIVE_DRAWER_OBJECTS_MIN: int = 1
ACTIVE_DRAWER_OBJECTS_MAX: int = 5
ACTIVE_TRASH_OBJECTS_MIN: int = 1
ACTIVE_TRASH_OBJECTS_MAX: int = 5

INACTIVE_OBJECT_PARK_Z: float = -10.0  # env-local meters below origin
INACTIVE_OBJECT_PARK_X_SPACING: float = 10.0  # unique x per parked object


# =====================================================================
# USD spawner (single-rigid-body convex-decomposition collision).
# Copied verbatim from ``trash_drawer_sort_cfg`` to keep this file self-
# contained — the synthesis USDs need this to spawn without articulation
# errors.
# =====================================================================
SYNTHESIS_CONVEX_DECOMPOSITION_CFG = sim_utils.ConvexDecompositionPropertiesCfg(
    max_convex_hulls=64,
    hull_vertex_limit=64,
    voxel_resolution=300000,
)


def _iter_mesh_prims(root_prim):
    stack = list(root_prim.GetChildren())
    while stack:
        prim = stack.pop()
        if prim.GetTypeName() == "Mesh":
            yield prim
        stack.extend(prim.GetChildren())


@clone
def spawn_synthesis_with_convex_decomposition(prim_path, cfg, *args, **kwargs):
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    for mesh_prim in _iter_mesh_prims(prim):
        mesh_path = mesh_prim.GetPath().pathString
        schemas.define_collision_properties(mesh_path, collision_cfg, stage=stage)
        schemas.define_mesh_collision_properties(
            mesh_path,
            SYNTHESIS_CONVEX_DECOMPOSITION_CFG,
            stage=stage,
        )
    return prim


@clone
def spawn_table_cleaning_rigid(prim_path, cfg, *args, **kwargs):
    """Spawn a table_cleaning USD as a *single* rigid body.

    Same pattern as ``cook_foods_cfg.spawn_synthesis_rigid_with_collision``:
    Meshy / Synthesis USDs often author ``RigidBodyAPI`` on the root *and*
    on inner xforms (plus internal Joints), which makes Isaac Lab's
    ``RigidObject`` wrapper fail with ``Found multiple`` and trips PhysX
    "missing xformstack reset" errors. This helper strips child rigid /
    articulation APIs, removes any inner joints, re-applies the cfg's root
    rigid / mass props, and authors convex-decomposition collision on
    every visible mesh.
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

    collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    for mesh_prim in _iter_mesh_prims(prim):
        mesh_path = mesh_prim.GetPath().pathString
        schemas.define_collision_properties(mesh_path, collision_cfg, stage=stage)
        schemas.define_mesh_collision_properties(
            mesh_path,
            SYNTHESIS_CONVEX_DECOMPOSITION_CFG,
            stage=stage,
        )
    return prim


# =====================================================================
# Asset cfgs
# =====================================================================
BEDSIDE_TABLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/BedsideTable014",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_synthesis_with_convex_decomposition,
        usd_path=BEDSIDE_TABLE_USD_PATH,
        scale=BEDSIDE_TABLE_SCALE,
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
        pos=(DRAWER_INIT_X, DRAWER_INIT_Y, DRAWER_TABLETOP_Z_OFFSET),
        rot=DRAWER_INIT_ROT,
        joint_pos={DRAWER_JOINT_NAMES: 0.0},
    ),
    actuators={
        # Simple friction-based detent (mirrors biamnaul_articulations /
        # the cook-foods stove knob): stiffness=0 so there's no spring-back,
        # small Coulomb friction holds the slide where the hand leaves it,
        # and armature regularizes the light joint so it stays stable.
        "drawer_friction": ImplicitActuatorCfg(
            joint_names_expr=[DRAWER_JOINT_NAMES],
            effort_limit_sim={DRAWER_JOINT_NAMES: 100.0},
            velocity_limit_sim={DRAWER_JOINT_NAMES: 100.0},
            stiffness={DRAWER_JOINT_NAMES: DRAWER_STIFFNESS},
            damping={DRAWER_JOINT_NAMES: DRAWER_DAMPING},
            friction={DRAWER_JOINT_NAMES: DRAWER_FRICTION},
            dynamic_friction={DRAWER_JOINT_NAMES: DRAWER_DYNAMIC_FRICTION},
            armature={DRAWER_JOINT_NAMES: DRAWER_ARMATURE},
        ),
    },
)


TRASH_CAN_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/TrashCan006",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_synthesis_with_convex_decomposition,
        usd_path=TRASH_CAN_USD_PATH,
        scale=TRASH_CAN_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            max_effort=100.0,
            stiffness=10.0,
            damping=0.1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(TRASH_CAN_INIT_X, TRASH_CAN_INIT_Y, TRASH_CAN_TABLETOP_Z_OFFSET),
        rot=TRASH_CAN_INIT_ROT,
        joint_pos={TRASH_CAN_LID_JOINT_NAMES: TRASH_CAN_LID_RESET_POSITION},
    ),
    actuators={
        "trash_can_lid_pd": ImplicitActuatorCfg(
            joint_names_expr=[TRASH_CAN_LID_JOINT_NAMES],
            effort_limit_sim=TRASH_CAN_EFFORT_LIMIT,
            velocity_limit_sim=100.0,
            stiffness=TRASH_CAN_STIFFNESS,
            damping=TRASH_CAN_DAMPING,
            friction=TRASH_CAN_FRICTION,
            dynamic_friction=TRASH_CAN_DYNAMIC_FRICTION,
        ),
    },
)


def _make_table_cleaning_cfg(
    prim_name: str,
    usd_path: str,
    mass: float,
    scale: float,
) -> RigidObjectCfg:
    """Build a single-rigid-body RigidObjectCfg for a table_cleaning USD."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_table_cleaning_rigid,
            usd_path=str(usd_path),
            scale=(scale, scale, scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Placeholder pose — z is patched in env ``__post_init__`` and the
            # per-reset event re-randomizes xy + yaw each episode.
            pos=(0.0, 0.0, 0.7),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _build_table_cleaning_cfgs(
    specs: tuple[tuple[str, str, str, str, float, float], ...],
    subdir_root: str,
) -> dict[str, RigidObjectCfg]:
    return {
        scene_name: _make_table_cleaning_cfg(
            prim_name,
            TABLE_CLEANING_DIR / subdir_root / subdir / usd_filename,
            mass,
            scale,
        )
        for scene_name, prim_name, subdir, usd_filename, mass, scale in specs
    }


_DRAWER_OBJECT_CFGS: dict[str, RigidObjectCfg] = _build_table_cleaning_cfgs(_TABLE_CLEANING_DRAWER_SPECS, "to_drawer")
_TRASH_OBJECT_CFGS: dict[str, RigidObjectCfg] = _build_table_cleaning_cfgs(_TABLE_CLEANING_TRASH_SPECS, "to_trash")


# =====================================================================
# Event helpers
# =====================================================================
def set_joint_position_target(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    target_position: float,
):
    """Set the PD position target on the resolved joints to a constant.

    Kept for ad-hoc use; the latch state machine below now drives the lid
    target itself, so this is unused by default.
    """
    asset = env.scene[asset_cfg.name]
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    joint_ids = resolve_joint_ids(env, asset_cfg)
    target = torch.full(
        (env_ids_t.shape[0], len(joint_ids)),
        float(target_position),
        device=env.device,
    )
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids_t)


# ---------------------------------------------------------------------
# Active-object subset selection (per-episode)
# ---------------------------------------------------------------------
def _ensure_active_object_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazy-allocate per-env active masks. Default: every object active.

    Returns ``(active_drawer, active_trash)`` — each ``(num_envs, len(...))``
    bool tensors stored on ``env`` for downstream consumers (success
    helpers, fell-off termination, demo recording).
    """
    n_drawer = len(DRAWER_OBJECT_NAMES)
    n_trash = len(TRASH_OBJECT_NAMES)
    if not hasattr(env, "_active_drawer_objects"):
        env._active_drawer_objects = torch.ones(env.num_envs, n_drawer, device=env.device, dtype=torch.bool)
    if not hasattr(env, "_active_trash_objects"):
        env._active_trash_objects = torch.ones(env.num_envs, n_trash, device=env.device, dtype=torch.bool)
    return env._active_drawer_objects, env._active_trash_objects


def _sample_active_mask(
    n_envs_local: int,
    n_objects: int,
    min_count: int,
    max_count: int,
    device,
) -> torch.Tensor:
    """Per-env active mask: sample k_i ∈ [min, max] active objects per env."""
    upper = min(max_count, n_objects)
    lower = max(1, min(min_count, upper))
    counts = torch.randint(lower, upper + 1, (n_envs_local,), device=device)
    # argsort(argsort(scores)) gives each object's rank within its env. The
    # first ``counts[env]`` ranks are active.
    scores = torch.rand(n_envs_local, n_objects, device=device)
    rank = scores.argsort(dim=1).argsort(dim=1)
    return rank < counts.unsqueeze(1)


def _install_active_object_metadata_getter(env) -> None:
    def get_active_object_metadata(env_index: int = 0) -> dict:
        active_drawer, active_trash = _ensure_active_object_state(env)
        env_index_int = int(env_index)
        return {
            "groups": {
                "drawer": {
                    "object_names": list(DRAWER_OBJECT_NAMES),
                    "active_mask": active_drawer[env_index_int].detach().cpu().to(torch.bool).tolist(),
                },
                "trash": {
                    "object_names": list(TRASH_OBJECT_NAMES),
                    "active_mask": active_trash[env_index_int].detach().cpu().to(torch.bool).tolist(),
                },
            },
        }

    env.get_active_object_metadata = get_active_object_metadata


def reset_active_object_subset(env, env_ids):
    """Resample which objects are active this episode. Must run *before*
    ``reset_sortable_objects_random`` so the placement event reads the
    fresh mask.
    """
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    active_drawer, active_trash = _ensure_active_object_state(env)
    n_local = env_ids_t.shape[0]
    active_drawer[env_ids_t] = _sample_active_mask(
        n_local,
        len(DRAWER_OBJECT_NAMES),
        ACTIVE_DRAWER_OBJECTS_MIN,
        ACTIVE_DRAWER_OBJECTS_MAX,
        env.device,
    )
    active_trash[env_ids_t] = _sample_active_mask(
        n_local,
        len(TRASH_OBJECT_NAMES),
        ACTIVE_TRASH_OBJECTS_MIN,
        ACTIVE_TRASH_OBJECTS_MAX,
        env.device,
    )
    _install_active_object_metadata_getter(env)


def _active_mask_all(env, env_ids_t: torch.Tensor) -> torch.Tensor:
    """Concatenated active mask over ``ALL_TABLE_CLEANING_NAMES`` for the
    given env subset. Shape: ``(len(env_ids_t), num_objects)``."""
    active_drawer, active_trash = _ensure_active_object_state(env)
    return torch.cat([active_drawer[env_ids_t], active_trash[env_ids_t]], dim=1)


# ---------------------------------------------------------------------
# Sortable-object rejection-sampling placement
# ---------------------------------------------------------------------
def _push_outside_circle(
    candidate_xy: torch.Tensor,
    excl_xy: torch.Tensor,
    min_dist,
    push_mask: torch.Tensor,
) -> torch.Tensor:
    """Active-avoidance helper: for each env where ``push_mask[i] = True``
    and ``candidate_xy[i]`` is inside the disc of radius ``min_dist`` around
    ``excl_xy[i]``, project the candidate radially outward to land exactly
    on the boundary. Other envs are left unchanged.

    ``min_dist`` may be a scalar or a per-env tensor of shape ``(n,)``.
    Handles the degenerate ``candidate == excl_xy`` case by picking an
    arbitrary +x direction so the push has a defined target.
    """
    direction = candidate_xy - excl_xy
    dist = direction.norm(dim=1)
    zero = dist < 1e-9
    if zero.any():
        direction = direction.clone()
        direction[zero, 0] = 1.0
        direction[zero, 1] = 0.0
        dist = direction.norm(dim=1)
    if not isinstance(min_dist, torch.Tensor):
        min_dist = torch.full_like(dist, float(min_dist))
    needs_push = (dist < min_dist) & push_mask
    if not bool(needs_push.any()):
        return candidate_xy
    scale = min_dist / dist
    pushed = excl_xy + direction * scale.unsqueeze(-1)
    return torch.where(needs_push.unsqueeze(-1), pushed, candidate_xy)


# Number of "push + reclamp" passes per object. Each pass walks every
# exclusion once (drawer + trash + every prior active object), so a tight
# spawn area may need multiple passes to converge as the clamp keeps
# nudging the candidate back into a previously-cleared exclusion.
# Upper bound = (max active priors) + 2 containers. With 5 drawer + 5
# trash active per env, the last object faces up to 9 + 2 = 11
# exclusions; we round up to 12 for headroom.
ACTIVE_AVOIDANCE_ITERATIONS: int = 12


def _sortable_spawn_valid(
    candidate_xy_local: torch.Tensor,
    placed_xy_local: torch.Tensor,
    drawer_xy_local: torch.Tensor,
    trash_xy_local: torch.Tensor,
    pairwise_min_dist: torch.Tensor,
    drawer_min_dist: float,
    trash_min_dist: float,
) -> torch.Tensor:
    """Per-env bool mask: candidate xy clears every prior object + drawer + trash.

    ``candidate_xy_local``: (n, 2). ``placed_xy_local``: (n, k, 2) with NaN
    sentinels for any prior slot that hasn't been placed yet (treated as
    "no obstacle"). ``pairwise_min_dist``: (k,) per-prior-object minimum
    distance, derived from ``radius_current + radius_prior + margin``.
    """
    n = candidate_xy_local.shape[0]
    valid = torch.ones(n, device=candidate_xy_local.device, dtype=torch.bool)

    if placed_xy_local.shape[1] > 0:
        diffs = candidate_xy_local.unsqueeze(1) - placed_xy_local  # (n, k, 2)
        dists = diffs.norm(dim=2)  # (n, k)
        far_enough = dists >= pairwise_min_dist.unsqueeze(0)  # broadcast (1, k)
        # Slots still at NaN (unplaced) shouldn't fail the check.
        has_prior = ~torch.isnan(placed_xy_local[..., 0])
        far_enough = far_enough | ~has_prior
        valid = valid & far_enough.all(dim=1)

    drawer_dist = (candidate_xy_local - drawer_xy_local).norm(dim=1)
    valid = valid & (drawer_dist >= drawer_min_dist)

    trash_dist = (candidate_xy_local - trash_xy_local).norm(dim=1)
    valid = valid & (trash_dist >= trash_min_dist)

    return valid


def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about world z) from a (..., 4) wxyz quaternion.

    Assumes the input represents a rotation about z only (roll = pitch = 0),
    which is the case for our drawer / trash default rotations. For a pure
    z-rotation, ``yaw = 2 * atan2(qz, qw)``.
    """
    qw = quat_wxyz[..., 0]
    qz = quat_wxyz[..., 3]
    return 2.0 * torch.atan2(qz, qw)


def reset_containers_random(env, env_ids):
    """Reset drawer + trash root poses with rejection sampling between them.

    The previous design used two independent ``mdp.reset_root_state_uniform``
    events, which could pick poses where ``|drawer_xy - trash_xy| <
    r_drawer + r_trash + margin``. This event places the drawer freely
    within its range, then rejection-samples the trash against it. Yaw is
    sampled as an offset from each asset's default rotation and applied
    as a pure-z quaternion (the only axis the original event varied).
    """
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    n = env_ids_t.shape[0]
    device = env.device

    drawer = env.scene["drawer"]
    trash = env.scene["trash_can"]
    env_origins = env.scene.env_origins[env_ids_t]
    env_origins_xy = env_origins[:, 0:2]
    env_origins_z = env_origins[:, 2]

    def _sample_uniform(low: float, high: float, shape) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.float).uniform_(low, high)

    def _apply(asset, base_root: torch.Tensor, offset_xy: torch.Tensor, yaw: torch.Tensor):
        root_state = base_root.clone()
        root_state[:, 0:2] = env_origins_xy + base_root[:, 0:2] + offset_xy
        root_state[:, 2] = env_origins_z + base_root[:, 2]
        half_yaw = yaw * 0.5
        root_state[:, 3] = torch.cos(half_yaw)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(half_yaw)
        root_state[:, 7:13] = 0.0
        asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
        asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)

    # Drawer: unconstrained uniform sample.
    drawer_base = drawer.data.default_root_state[env_ids_t].clone()
    drawer_offset_xy = torch.stack(
        [_sample_uniform(*DRAWER_RESET_X_RANGE, (n,)), _sample_uniform(*DRAWER_RESET_Y_RANGE, (n,))],
        dim=1,
    )
    default_drawer_yaw = _quat_to_yaw(drawer_base[:, 3:7])
    drawer_yaw = default_drawer_yaw + _sample_uniform(*DRAWER_RESET_YAW_RANGE, (n,))
    _apply(drawer, drawer_base, drawer_offset_xy, drawer_yaw)
    drawer_xy_local = drawer_base[:, 0:2] + drawer_offset_xy

    # Trash: rejection-sample so the two bounding circles stay clear.
    trash_base = trash.data.default_root_state[env_ids_t].clone()
    trash_offset_xy = torch.stack(
        [_sample_uniform(*TRASH_CAN_RESET_X_RANGE, (n,)), _sample_uniform(*TRASH_CAN_RESET_Y_RANGE, (n,))],
        dim=1,
    )
    min_dist = _DRAWER_RADIUS + _TRASH_RADIUS + OBJECT_OBJECT_SEPARATION_MARGIN
    trash_xy_local = trash_base[:, 0:2] + trash_offset_xy
    # Active avoidance: push the trash xy radially out of the drawer's
    # exclusion zone, then reclamp back into its own range. Iterate so
    # a clamp-back-into-an-exclusion gets corrected on the next pass.
    clamp_x_lo = trash_base[:, 0] + TRASH_CAN_RESET_X_RANGE[0]
    clamp_x_hi = trash_base[:, 0] + TRASH_CAN_RESET_X_RANGE[1]
    clamp_y_lo = trash_base[:, 1] + TRASH_CAN_RESET_Y_RANGE[0]
    clamp_y_hi = trash_base[:, 1] + TRASH_CAN_RESET_Y_RANGE[1]
    always_active = torch.ones(n, device=device, dtype=torch.bool)
    for _ in range(ACTIVE_AVOIDANCE_ITERATIONS):
        before = trash_xy_local.clone()
        trash_xy_local = _push_outside_circle(trash_xy_local, drawer_xy_local, min_dist, always_active)
        trash_xy_local = torch.stack(
            [
                torch.clamp(trash_xy_local[:, 0], clamp_x_lo, clamp_x_hi),
                torch.clamp(trash_xy_local[:, 1], clamp_y_lo, clamp_y_hi),
            ],
            dim=1,
        )
        if torch.allclose(trash_xy_local, before, atol=1e-6):
            break
    trash_offset_xy = trash_xy_local - trash_base[:, 0:2]
    default_trash_yaw = _quat_to_yaw(trash_base[:, 3:7])
    trash_yaw = default_trash_yaw + _sample_uniform(*TRASH_CAN_RESET_YAW_RANGE, (n,))
    _apply(trash, trash_base, trash_offset_xy, trash_yaw)


def reset_sortable_objects_random(env, env_ids):
    """Place active sortable objects at non-overlapping random poses on the
    tabletop. Inactive objects are parked far below the workspace at unique
    x offsets so they don't stack on each other.

    Each active object is rejection-tested against previously-placed *active*
    objects + drawer + trash. Inactive envs skip validation (they're going
    to the park location anyway, which lives nowhere near the validator's
    workspace).
    """
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    n = env_ids_t.shape[0]
    device = env.device

    env_origins = env.scene.env_origins[env_ids_t]  # (n, 3), world frame
    env_origins_xy = env_origins[:, 0:2]

    drawer = env.scene["drawer"]
    trash = env.scene["trash_can"]
    drawer_xy_local = drawer.data.root_pos_w[env_ids_t, 0:2] - env_origins_xy
    trash_xy_local = trash.data.root_pos_w[env_ids_t, 0:2] - env_origins_xy

    active_all = _active_mask_all(env, env_ids_t)  # (n, num_objects) bool

    placed_xy_local = torch.full((n, len(ALL_TABLE_CLEANING_NAMES), 2), float("nan"), device=device)
    radii_tensor = torch.tensor(_OBJECT_RADII, device=device)

    for i, object_name in enumerate(ALL_TABLE_CLEANING_NAMES):
        obj = env.scene[object_name]
        base_root = obj.data.default_root_state[env_ids_t].clone()  # (n, 13), env-local xy
        base_xy_local = base_root[:, 0:2]
        is_active = active_all[:, i]  # (n,) bool

        current_radius = float(_OBJECT_RADII[i])
        # Per-prior-object required distance = r_current + r_prior + margin.
        prior_radii = radii_tensor[:i]
        pairwise_min_dist = prior_radii + current_radius + OBJECT_OBJECT_SEPARATION_MARGIN
        # Container distances use the receptacle's diagonal AABB radius
        # plus the current object's radius. This is the true minimum gap
        # needed to guarantee no horizontal overlap at any yaw.
        drawer_min_dist = _DRAWER_RADIUS + current_radius + OBJECT_OBJECT_SEPARATION_MARGIN
        trash_min_dist = _TRASH_RADIUS + current_radius + OBJECT_OBJECT_SEPARATION_MARGIN

        # Initial uniform sample inside the spawn box.
        offset_xy = torch.empty((n, 2), device=device, dtype=base_root.dtype)
        offset_xy[:, 0].uniform_(OBJECT_SPAWN_X_RANGE[0], OBJECT_SPAWN_X_RANGE[1])
        offset_xy[:, 1].uniform_(OBJECT_SPAWN_Y_RANGE[0], OBJECT_SPAWN_Y_RANGE[1])
        candidate_xy_local = base_xy_local + offset_xy

        # Active avoidance: instead of rejection-sampling, push the
        # candidate radially out of every exclusion zone (drawer, trash,
        # already-placed active priors). Iterate so that a push that lands
        # the candidate in a different exclusion gets corrected. Inactive
        # envs are skipped (their candidate is overwritten by the park
        # location below).
        clamp_x_lo = base_xy_local[:, 0] + OBJECT_SPAWN_X_RANGE[0]
        clamp_x_hi = base_xy_local[:, 0] + OBJECT_SPAWN_X_RANGE[1]
        clamp_y_lo = base_xy_local[:, 1] + OBJECT_SPAWN_Y_RANGE[0]
        clamp_y_hi = base_xy_local[:, 1] + OBJECT_SPAWN_Y_RANGE[1]
        for _ in range(ACTIVE_AVOIDANCE_ITERATIONS):
            before = candidate_xy_local.clone()
            candidate_xy_local = _push_outside_circle(candidate_xy_local, drawer_xy_local, drawer_min_dist, is_active)
            candidate_xy_local = _push_outside_circle(candidate_xy_local, trash_xy_local, trash_min_dist, is_active)
            for j in range(i):
                prior_xy = placed_xy_local[:, j, :]
                prior_active = ~torch.isnan(prior_xy[:, 0])
                if not bool(prior_active.any()):
                    continue
                safe_prior_xy = torch.where(prior_active.unsqueeze(-1), prior_xy, torch.zeros_like(prior_xy))
                candidate_xy_local = _push_outside_circle(
                    candidate_xy_local,
                    safe_prior_xy,
                    float(pairwise_min_dist[j].item()),
                    is_active & prior_active,
                )
            # Reclamp into the spawn box after the push. The clamp can
            # nudge the candidate back into an exclusion — the next
            # iteration's push handles that.
            candidate_xy_local = torch.stack(
                [
                    torch.clamp(candidate_xy_local[:, 0], clamp_x_lo, clamp_x_hi),
                    torch.clamp(candidate_xy_local[:, 1], clamp_y_lo, clamp_y_hi),
                ],
                dim=1,
            )
            if torch.allclose(candidate_xy_local, before, atol=1e-6):
                break
        # Inactive slots get NaN so they don't contribute to subsequent
        # rejection checks.
        nan_xy = torch.full_like(candidate_xy_local, float("nan"))
        placed_xy_local[:, i, :] = torch.where(is_active.unsqueeze(1), candidate_xy_local, nan_xy)

        # Park location for inactive envs: unique x offset per object so two
        # parked objects don't share the same spot.
        park_xy_local = torch.zeros_like(candidate_xy_local)
        park_xy_local[:, 0] = i * INACTIVE_OBJECT_PARK_X_SPACING
        park_xy_local[:, 1] = 0.0

        # Build root state — branch xy/z/yaw on is_active.
        root_state = base_root.clone()
        active_xy_world = env_origins_xy + candidate_xy_local
        park_xy_world = env_origins_xy + park_xy_local
        root_state[:, 0:2] = torch.where(is_active.unsqueeze(1), active_xy_world, park_xy_world)

        active_z = env_origins[:, 2] + base_root[:, 2]
        park_z = env_origins[:, 2] + INACTIVE_OBJECT_PARK_Z
        root_state[:, 2] = torch.where(is_active, active_z, park_z)

        yaw = torch.empty(n, device=device, dtype=base_root.dtype).uniform_(-math.pi, math.pi)
        half_yaw = yaw * 0.5
        active_qw = torch.cos(half_yaw)
        active_qz = torch.sin(half_yaw)
        identity_qw = torch.ones_like(active_qw)
        identity_qz = torch.zeros_like(active_qz)
        root_state[:, 3] = torch.where(is_active, active_qw, identity_qw)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.where(is_active, active_qz, identity_qz)
        root_state[:, 7:13] = 0.0

        obj.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
        obj.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)

        # Per IsaacLab discussion #2834 the canonical "deactivate" trick is
        # a USD visibility toggle on the per-env prim. Without it, parked
        # objects are still rendered (and shadow the camera even if they're
        # 10 m below the workspace). We do per-env-list batching: pass the
        # subset of env_ids that are active/inactive for this object.
        active_env_ids = env_ids_t[is_active]
        inactive_env_ids = env_ids_t[~is_active]
        if active_env_ids.numel() > 0:
            obj.set_visibility(True, env_ids=active_env_ids)
        if inactive_env_ids.numel() > 0:
            obj.set_visibility(False, env_ids=inactive_env_ids)


# ---------------------------------------------------------------------
# Trash-can lid latch state machine
# ---------------------------------------------------------------------
def _ensure_trash_can_latch_state(env):
    """Allocate per-env latch buffers on first access."""
    if not hasattr(env, "_trash_can_latched"):
        env._trash_can_latched = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if not hasattr(env, "_trash_can_prev_q"):
        env._trash_can_prev_q = torch.full(
            (env.num_envs,),
            TRASH_CAN_LID_RESET_POSITION,
            device=env.device,
        )
    if not hasattr(env, "_trash_can_cooldown_s"):
        env._trash_can_cooldown_s = torch.zeros(env.num_envs, device=env.device)
    return env._trash_can_latched, env._trash_can_prev_q, env._trash_can_cooldown_s


def _apply_trash_can_latch_state(
    env,
    env_ids_t: torch.Tensor,
    lid_joint_ids,
    latched_local: torch.Tensor,
    latched_target_position: float = TRASH_CAN_LID_CLOSED_POSITION,
):
    """Write joint position limits + PD target to match each env's latch state."""
    trash = env.scene["trash_can"]
    n = env_ids_t.shape[0]
    n_joints = len(lid_joint_ids)
    dtype = trash.data.joint_pos.dtype

    # Keep physical travel available in both states; latch behavior is carried
    # by the PD target, not by collapsing the lower limit to CLOSED.
    lower_value = torch.full((n,), TRASH_CAN_LID_OPEN_POSITION, device=env.device, dtype=dtype)
    upper_value = torch.full((n,), TRASH_CAN_LID_UPPER_LIMIT, device=env.device, dtype=dtype)
    limits = torch.stack([lower_value, upper_value], dim=-1)  # (n, 2)
    limits = limits.unsqueeze(1).expand(n, n_joints, 2).contiguous()
    trash.write_joint_position_limit_to_sim(
        limits, joint_ids=lid_joint_ids, env_ids=env_ids_t, warn_limit_violation=False
    )

    # PD target mirrors the limit: closed when latched, open when free.
    target_value = torch.where(
        latched_local,
        torch.full((n,), latched_target_position, device=env.device, dtype=dtype),
        torch.full((n,), TRASH_CAN_LID_OPEN_POSITION, device=env.device, dtype=dtype),
    )
    trash.set_joint_position_target(
        target_value.unsqueeze(-1).expand(n, n_joints),
        joint_ids=lid_joint_ids,
        env_ids=env_ids_t,
    )


def reset_trash_can_latch(env, env_ids, asset_cfg: SceneEntityCfg):
    """Reset each env to ``latched`` at ``TRASH_CAN_LID_RESET_POSITION``."""
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    latched, prev_q, cooldown = _ensure_trash_can_latch_state(env)
    latched[env_ids_t] = True
    prev_q[env_ids_t] = TRASH_CAN_LID_RESET_POSITION
    cooldown[env_ids_t] = 0.0
    lid_joint_ids = resolve_joint_ids(env, asset_cfg)
    _apply_trash_can_latch_state(
        env,
        env_ids_t,
        lid_joint_ids,
        latched[env_ids_t],
        latched_target_position=TRASH_CAN_LID_RESET_POSITION,
    )


def update_trash_can_latch(env, env_ids, asset_cfg: SceneEntityCfg):
    """Toggle latch on threshold crossing toward 0° using one shared threshold."""
    env_ids_t = resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    trash = env.scene[asset_cfg.name]
    lid_joint_ids = resolve_joint_ids(env, asset_cfg)
    latched, prev_q, cooldown = _ensure_trash_can_latch_state(env)

    # Mean across hinge joints (there's only one — averaging is just defensive).
    joint_pos = trash.data.joint_pos[env_ids_t][:, lid_joint_ids].mean(dim=1)
    prev_local = prev_q[env_ids_t]

    step_dt = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)
    cooldown[env_ids_t] = torch.clamp(cooldown[env_ids_t] - step_dt, min=0.0)
    local_cooldown = cooldown[env_ids_t]

    # Single-threshold crossing toward 0° toggles latch.
    trigger = (
        (prev_local < TRASH_CAN_LID_TOGGLE_THRESHOLD)
        & (joint_pos >= TRASH_CAN_LID_TOGGLE_THRESHOLD)
        & (local_cooldown <= 0.0)
    )
    if not trigger.any():
        prev_q[env_ids_t] = joint_pos
        return

    trigger_env_ids = env_ids_t[trigger]
    new_latched = ~latched[trigger_env_ids]
    latched[trigger_env_ids] = new_latched
    cooldown[trigger_env_ids] = TRASH_CAN_LID_TOGGLE_COOLDOWN_S
    _apply_trash_can_latch_state(env, trigger_env_ids, lid_joint_ids, new_latched)
    prev_q[env_ids_t] = joint_pos


# ---------------------------------------------------------------------
# Success-condition helpers
# ---------------------------------------------------------------------
def _object_local_pos(env, container_name: str, object_name: str) -> torch.Tensor:
    """Object root pos expressed in the container's root frame."""
    container = env.scene[container_name]
    obj = env.scene[object_name]
    return quat_apply_inverse(
        container.data.root_quat_w,
        obj.data.root_pos_w - container.data.root_pos_w,
    )


def _object_in_aabb_mask(
    env,
    container_name: str,
    object_name: str,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> torch.Tensor:
    pos_b = _object_local_pos(env, container_name, object_name)
    lower_t = torch.tensor(lower, device=env.device, dtype=pos_b.dtype)
    upper_t = torch.tensor(upper, device=env.device, dtype=pos_b.dtype)
    return ((pos_b >= lower_t) & (pos_b <= upper_t)).all(dim=1)


def _cached_joint_ids(env, asset_name: str, attr: str, joint_regex: str) -> list[int]:
    """Resolve ``joint_regex`` once and cache the joint ids on the env."""
    asset = env.scene[asset_name]
    if not hasattr(env, attr):
        ids, _ = asset.find_joints([joint_regex])
        setattr(env, attr, ids)
    return getattr(env, attr)


def _drawer_closed_mask(env) -> torch.Tensor:
    drawer = env.scene["drawer"]
    joint_ids = _cached_joint_ids(env, "drawer", "_drawer_success_joint_ids", DRAWER_JOINT_NAMES)
    joint_pos = drawer.data.joint_pos[:, joint_ids]
    return joint_pos.abs().max(dim=1).values < DRAWER_CLOSED_THRESHOLD


def _trash_lid_closed_mask(env) -> torch.Tensor:
    """Closed = lid joint near 0° AND the latch state machine reports latched.

    Joint position alone is not enough — when the lid is mid-swing after an
    unlatch press it briefly crosses 0° on its way to -90°, which would
    otherwise false-positive. Requiring ``latched`` ensures the lid was
    actually pressed shut, not just transiting through 0°.
    """
    trash = env.scene["trash_can"]
    joint_ids = _cached_joint_ids(env, "trash_can", "_trash_lid_success_joint_ids", TRASH_CAN_LID_JOINT_NAMES)
    joint_pos = trash.data.joint_pos[:, joint_ids]
    near_zero = joint_pos.abs().max(dim=1).values < TRASH_LID_CLOSED_THRESHOLD
    latched, _, _ = _ensure_trash_can_latch_state(env)
    return near_zero & latched


def drawer_subtask_success(env) -> torch.Tensor:
    """Drawer closed AND every *active* drawer object inside the AABB.

    Inactive objects are parked far below the workspace; we don't require
    them to be inside the drawer.
    """
    ok = _drawer_closed_mask(env)
    active_drawer, _ = _ensure_active_object_state(env)
    for i, name in enumerate(DRAWER_OBJECT_NAMES):
        inside = _object_in_aabb_mask(env, "drawer", name, OBJECT_IN_DRAWER_LOCAL_MIN, OBJECT_IN_DRAWER_LOCAL_MAX)
        ok = ok & (~active_drawer[:, i] | inside)
    return ok


def trash_subtask_success(env) -> torch.Tensor:
    """Trash lid closed AND every *active* trash object inside the AABB."""
    ok = _trash_lid_closed_mask(env)
    _, active_trash = _ensure_active_object_state(env)
    for i, name in enumerate(TRASH_OBJECT_NAMES):
        inside = _object_in_aabb_mask(env, "trash_can", name, OBJECT_IN_TRASH_LOCAL_MIN, OBJECT_IN_TRASH_LOCAL_MAX)
        ok = ok & (~active_trash[:, i] | inside)
    return ok


# Debug: periodically print per-object containment + closed state for one
# env, so you can see *why* success isn't firing (object out of the AABB vs
# receptacle not closed vs object parked/inactive). Set DEBUG_CONTAINMENT
# False to silence.
DEBUG_CONTAINMENT = True
DEBUG_CONTAINMENT_EVERY_N_STEPS = 30
DEBUG_CONTAINMENT_ENV = 0


def _debug_print_containment(env) -> None:
    if not DEBUG_CONTAINMENT:
        return
    step = int(getattr(env, "common_step_counter", 0))
    if DEBUG_CONTAINMENT_EVERY_N_STEPS > 0 and step % DEBUG_CONTAINMENT_EVERY_N_STEPS != 0:
        return
    e = DEBUG_CONTAINMENT_ENV
    if e >= env.num_envs:
        return

    active_drawer, active_trash = _ensure_active_object_state(env)
    drawer_closed = bool(_drawer_closed_mask(env)[e].item())
    lid_closed = bool(_trash_lid_closed_mask(env)[e].item())

    def fmt(v):
        return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"

    lines = [f"[trash_drawer DEBUG env{e}] step={step}  drawer_closed={drawer_closed}  lid_closed={lid_closed}"]
    lines.append(f"  DRAWER zone xyz in {OBJECT_IN_DRAWER_LOCAL_MIN} .. {OBJECT_IN_DRAWER_LOCAL_MAX}")
    for i, name in enumerate(DRAWER_OBJECT_NAMES):
        if not bool(active_drawer[e, i].item()):
            lines.append(f"    {name:<14} parked (inactive)")
            continue
        pos_b = _object_local_pos(env, "drawer", name)[e]
        inside = bool(
            _object_in_aabb_mask(env, "drawer", name, OBJECT_IN_DRAWER_LOCAL_MIN, OBJECT_IN_DRAWER_LOCAL_MAX)[e].item()
        )
        lines.append(f"    {name:<14} in_drawer={str(inside):<5} pos_b={fmt(pos_b.tolist())}")
    lines.append(f"  TRASH  zone xyz in {OBJECT_IN_TRASH_LOCAL_MIN} .. {OBJECT_IN_TRASH_LOCAL_MAX}")
    for i, name in enumerate(TRASH_OBJECT_NAMES):
        if not bool(active_trash[e, i].item()):
            lines.append(f"    {name:<14} parked (inactive)")
            continue
        pos_b = _object_local_pos(env, "trash_can", name)[e]
        inside = bool(
            _object_in_aabb_mask(env, "trash_can", name, OBJECT_IN_TRASH_LOCAL_MIN, OBJECT_IN_TRASH_LOCAL_MAX)[e].item()
        )
        lines.append(f"    {name:<14} in_trash={str(inside):<5} pos_b={fmt(pos_b.tolist())}")
    lines.append(
        f"  -> drawer_subtask={bool(drawer_subtask_success(env)[e].item())}  "
        f"trash_subtask={bool(trash_subtask_success(env)[e].item())}"
    )
    print("\n".join(lines), flush=True)


def task_success(env) -> torch.Tensor:
    """Final task success: both sub-tasks pass simultaneously."""
    # _debug_print_containment(env)
    return drawer_subtask_success(env) & trash_subtask_success(env)


def any_object_fell(env, fallen_z_threshold: float) -> torch.Tensor:
    """Per-env mask: True if any *active* sortable object's z (env-local) is
    below ``fallen_z_threshold``. Inactive (parked) objects sit well below
    the threshold by design — they're excluded from this check.
    """
    fell = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env_origin_z = env.scene.env_origins[:, 2]
    active_drawer, active_trash = _ensure_active_object_state(env)
    for i, name in enumerate(DRAWER_OBJECT_NAMES):
        obj = env.scene[name]
        z_local = obj.data.root_pos_w[:, 2] - env_origin_z
        fell = fell | (active_drawer[:, i] & (z_local < fallen_z_threshold))
    for i, name in enumerate(TRASH_OBJECT_NAMES):
        obj = env.scene[name]
        z_local = obj.data.root_pos_w[:, 2] - env_origin_z
        fell = fell | (active_trash[:, i] & (z_local < fallen_z_threshold))
    return fell


# =====================================================================
# Scene / Event / Env configclasses
# =====================================================================
@configclass
class TrashDrawerSortSimpleSceneCfg(dexverse_base_env.SceneCfg):
    drawer: ArticulationCfg = BEDSIDE_TABLE_CFG
    trash_can: ArticulationCfg = TRASH_CAN_CFG

    # Drawer-bound objects.
    advil: RigidObjectCfg = _DRAWER_OBJECT_CFGS["advil"]
    blue_stapler: RigidObjectCfg = _DRAWER_OBJECT_CFGS["blue_stapler"]
    charger: RigidObjectCfg = _DRAWER_OBJECT_CFGS["charger"]
    headphone: RigidObjectCfg = _DRAWER_OBJECT_CFGS["headphone"]
    marker: RigidObjectCfg = _DRAWER_OBJECT_CFGS["marker"]

    # Trash-bound objects.
    broken_twix: RigidObjectCfg = _TRASH_OBJECT_CFGS["broken_twix"]
    crumbled_paper: RigidObjectCfg = _TRASH_OBJECT_CFGS["crumbled_paper"]
    crushed_bottle: RigidObjectCfg = _TRASH_OBJECT_CFGS["crushed_bottle"]
    paper_cup: RigidObjectCfg = _TRASH_OBJECT_CFGS["paper_cup"]
    soda_can: RigidObjectCfg = _TRASH_OBJECT_CFGS["soda_can"]

    # ``configclass`` requires an annotation here, otherwise the None default
    # is treated as a class attribute and the inherited fields after
    # ``object`` (cameras, etc.) silently drop out.
    object: RigidObjectCfg | ArticulationCfg | None = None


@configclass
class TrashDrawerSortSimpleEventCfg(dexverse_base_env.EventCfg):
    """Reset events: drawer / trash root pose + joints to init, sortable
    items to a uniform-random pose on the table.

    Order matters here — pose first, then joints — so the joint-init pass
    sees the post-randomization root pose.
    """

    # Joint reset of drawer + trash poses, with rejection sampling between
    # them so their bounding circles don't overlap. Replaces the two
    # independent ``mdp.reset_root_state_uniform`` events that could pick
    # poses where the trash bin clips into the drawer.
    reset_containers = EventTerm(
        func=reset_containers_random,
        mode="reset",
        params={},
    )
    reset_drawer_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("drawer", joint_names=DRAWER_JOINT_NAMES)},
    )
    reset_trash_can_joints = EventTerm(
        # Physical state reset: lid back to ``TRASH_CAN_LID_RESET_POSITION``
        # (slightly open), button back to USD default. Latch/target is applied
        # separately below.
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("trash_can", joint_names=TRASH_CAN_JOINT_NAMES)},
    )
    reset_trash_can_latch = EventTerm(
        # Start each episode with latch engaged at RESET_POSITION. Physical
        # travel remains [-90°, 0°]; latch behavior is via PD target only.
        func=reset_trash_can_latch,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("trash_can", joint_names=TRASH_CAN_LID_JOINT_NAMES)},
    )
    update_trash_can_latch = EventTerm(
        # Per-step latch update. A single threshold crossing toward 0° toggles
        # latched/unlatched (with cooldown to avoid double-trigger).
        func=update_trash_can_latch,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=False,
        params={"asset_cfg": SceneEntityCfg("trash_can", joint_names=TRASH_CAN_LID_JOINT_NAMES)},
    )
    # Re-sample which objects are active this episode. Must run BEFORE
    # ``reset_sortable_objects`` so the placement event reads the fresh
    # mask and parks the inactive ones at the graveyard location.
    reset_active_object_subset = EventTerm(
        func=reset_active_object_subset,
        mode="reset",
        params={},
    )
    # Single batched event that places every sortable object: active ones
    # via rejection-sampled random xy + yaw on the table, inactive ones
    # at a far-below-workspace park location.
    reset_sortable_objects = EventTerm(
        func=reset_sortable_objects_random,
        mode="reset",
        params={},
    )


@configclass
class TrashDrawerSortSimpleTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Only the combined task success terminates the episode.

    ``drawer_subtask_success`` / ``trash_subtask_success`` remain callable
    for observations or per-stage rewards, but are intentionally not wired
    as terminations — finishing one half early should not stop the episode
    before the other half is also done.
    """

    success = DoneTerm(func=task_success)
    object_fell = DoneTerm(
        func=any_object_fell,
        params={
            "fallen_z_threshold": dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT - OBJECT_FALLEN_Z_MARGIN,
        },
    )


@configclass
class TrashDrawerSortSimpleEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Trash + drawer scene with 10 ``table_cleaning`` USD objects to sort.

    Drawer-bound: ``advil``, ``blue_stapler``, ``charger``, ``headphone``,
    ``marker``. Trash-bound: ``broken_twix``, ``crumbled_paper``,
    ``crushed_bottle``, ``paper_cup``, ``soda_can``. Edit the
    ``_TABLE_CLEANING_*_SPECS`` tuples at the top of the file to change.

    Default robot is the floating right Shadow hand. Pass
    ``--robot_type <key>`` (any key in ``SIMPLE_RETARGETER_LAYOUT_SOURCES``)
    to ``teleop_agent.py`` to swap.
    """

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    scene: TrashDrawerSortSimpleSceneCfg = TrashDrawerSortSimpleSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=False,
    )
    events: TrashDrawerSortSimpleEventCfg = TrashDrawerSortSimpleEventCfg()
    terminations: TrashDrawerSortSimpleTerminationsCfg = TrashDrawerSortSimpleTerminationsCfg()

    # ---- Debug viz: the two "contained in" acceptance boxes ----
    # Semi-transparent boxes tracking the drawer / trash-can roots, matching
    # the AABBs the success helpers test. Gated by ``enable_debug_vis``.
    drawer_zone_color: tuple[float, float, float] = (0.1, 0.6, 1.0)  # cyan
    trash_zone_color: tuple[float, float, float] = (1.0, 0.55, 0.1)  # orange
    containment_zone_opacity: float = 0.25

    def __post_init__(self):
        # Alias the first table_cleaning object into the canonical "object"
        # slot so the base __post_init__ leaves observation/camera hooks
        # alone; we drop the alias immediately after super() runs.
        first_object_name = ALL_TABLE_CLEANING_NAMES[0]
        self.scene.object = getattr(self.scene, first_object_name)

        super().__post_init__()

        self.scene.object = None
        self.commands.object_pose = None
        self.observations.perception = None
        # Inherited ``observations.contact`` group declares ``contact: ObsTerm
        # = MISSING`` and nothing fills it on this simplified scene, so
        # configclass validation rejects the env. Null the whole group out
        # — we don't use contact obs here.
        self.observations.contact = None
        self.terminations.object_out_of_bound = None
        for event_name in ("reset_object", "object_physics_material", "object_scale_mass"):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)
        self.events.reset_environment_background = None
        self.events.reset_table_texture = None

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        dx, dy, _ = self.scene.drawer.init_state.pos
        tx, ty, _ = self.scene.trash_can.init_state.pos
        self.scene.drawer.init_state.pos = (dx, dy, table_top_z + DRAWER_TABLETOP_Z_OFFSET)
        self.scene.trash_can.init_state.pos = (tx, ty, table_top_z + TRASH_CAN_TABLETOP_Z_OFFSET)

        # Per-object spawn z: lift each item so its *bottom* sits
        # ``OBJECT_SPAWN_HEIGHT`` above the tabletop. ``bottom_offset`` comes
        # from the asset's aabb.json (distance from the asset's origin to
        # its lowest authored point, with cfg scale applied), so assets
        # with non-centred origins still land flush with the table.
        for object_name in ALL_TABLE_CLEANING_NAMES:
            obj_cfg = getattr(self.scene, object_name)
            ox, oy, _ = obj_cfg.init_state.pos
            bottom_offset = _OBJECT_METADATA[object_name]["bottom_offset"]
            obj_cfg.init_state.pos = (ox, oy, table_top_z + bottom_offset + OBJECT_SPAWN_HEIGHT)

        # Per-object resets are now handled by the single batched event
        # ``reset_sortable_objects`` declared on the EventCfg, which does
        # rejection sampling against the drawer + trash + already-placed
        # objects each reset.

        self.episode_length_s = 40.0

        # Inherited ``privileged.hand_tips_state_b`` defaults to LEAP-hand
        # body names (``["base", "thumb_fingertip", ...]``). Swap to the
        # active robot's fingertip body names — Shadow-hand names like
        # ``rh_palm`` / ``rh_thtip`` / ``rh_fftip`` don't match the LEAP
        # defaults and the manager raises before sim starts.
        if getattr(self.observations, "privileged", None) is not None and hasattr(
            self.observations.privileged, "hand_tips_state_b"
        ):
            self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
                self.robot_config.hand_tips_body_names
            )

        setup_floating_teleop(self)
        self.configure_debug_vis()

    def configure_debug_vis(self) -> None:
        """Populate the drawer / trash containment-zone markers in
        ``observations.scene_vis`` when ``self.enable_debug_vis`` is True.

        Renders the drawer / trash containment AABBs as boxes that track each
        container's root (same frame + extents the success helpers test, so
        what you see is exactly the acceptance region).
        """
        super().configure_debug_vis()
        if not self.enable_debug_vis:
            return
        if self.observations.scene_vis is None:
            self.observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()
        self.observations.scene_vis.drawer_zone_vis = ObsTerm(
            func=mdp.forbidden_zones_vis,
            params={
                "sphere_zones": [],
                "box_zones": [_aabb_to_box_zone(OBJECT_IN_DRAWER_LOCAL_MIN, OBJECT_IN_DRAWER_LOCAL_MAX)],
                "cylinder_zones": [],
                "object_cfg": SceneEntityCfg("drawer"),
                "color": self.drawer_zone_color,
                "opacity": self.containment_zone_opacity,
                "prim_path_prefix": "/Visuals/DrawerContainmentZone",
            },
        )
        self.observations.scene_vis.trash_zone_vis = ObsTerm(
            func=mdp.forbidden_zones_vis,
            params={
                "sphere_zones": [],
                "box_zones": [_aabb_to_box_zone(OBJECT_IN_TRASH_LOCAL_MIN, OBJECT_IN_TRASH_LOCAL_MAX)],
                "cylinder_zones": [],
                "object_cfg": SceneEntityCfg("trash_can"),
                "color": self.trash_zone_color,
                "opacity": self.containment_zone_opacity,
                "prim_path_prefix": "/Visuals/TrashContainmentZone",
            },
        )
