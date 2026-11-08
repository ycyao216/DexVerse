# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon task config: load food into a microwave, shut it, turn a dial.

Task intent
~~~~~~~~~~~
A microwave (closed) and a small fridge both sit on the table; a plate of food
spawns INSIDE the closed fridge. The fridge door faces -Y and, when open, fouls
the microwave door, so the fridge must be SHUT before the microwave can be
opened. The sequence the policy / teleoperator must perform, in order, is::

    1. open_fridge  -- swing the fridge door open         (revolute door joint_1)
    2. close_fridge -- retrieve the food, then shut it    (revolute door joint_1)
    3. open         -- swing the microwave door open      (revolute door joint_0)
    4. place        -- set the food inside the cavity     (food at the in-cavity goal)
    5. close        -- swing the microwave door shut      (revolute door joint_0)
    6. rotate_knob  -- turn EITHER control dial by +/-18 deg (joint_1 or joint_2)

Success is driven by a strict, persistent :class:`mdp.StageGraphSpec` (one
sub-stage per step above); the terminal stage is ``rotate_knob``. The strict
ordering forces ``close_fridge`` before the microwave ``open`` (the challenge:
the robot must know to shut the fridge first), and the knob turn to come after
the microwave is closed. Either dial counts. Success also needs the finished state
true right now: the dish upright inside, the door shut, a dial turned, and the dish
never tipped over. See ``checks.py`` for the thresholds and how they were measured.

Asset notes
~~~~~~~~~~~
- Microwave (``partnet_mobility/.../microwave/7167.usd``) has three revolute
  joints: ``joint_0`` (the door, about Y, 0..90 deg) and ``joint_1``/``joint_2``,
  two stacked control dials (about Z). All three are driven with a
  *friction-detent* :class:`ImplicitActuatorCfg` (``stiffness=0`` -> no setpoint
  spring; static + dynamic ``friction`` and ``damping`` hold each joint wherever
  the hand leaves it), so the door stays where placed and the dials hold their
  angle. BOTH dials are range-limited to ``[-pi/2, pi/2]`` by the asset prep (they start
  at 0, mid-range) so they can be turned either way; turning EITHER one by
  +/-18 deg from its start completes the task. Tune ``DOOR_STATIC_FRICTION``
  first: too low and the door sags open, too high and the hand cannot move it.
- Food is a round dish of stew with a flat side handle, built by ``dish.py`` so a
  single hand can carry it. It is the canonical ``scene.object`` and spawns inside
  the fridge with its handle toward the door.
- Minifridge (``partnet_mobility/.../fridge/10797/mobility.usd``) is a small
  fixed-base fridge that sits ON the table. Its body is one revolute door
  (``joint_1``, ``link_1``) on a fixed body (``base`` + ``link_0``). PartNet
  authors the body as a single open-box collision mesh whose convex hull is the
  SOLID box, so any single hull (or convex decomposition of it) SEALS the
  interior -- the hand can't reach in. :func:`_open_fridge_cavity` bakes a copy
  that rebuilds the cavity: it drops the sealing body hull (keeping the thin
  shelf/floor) and re-authors back/sides/top wall colliders, leaving the front
  (door) face open so food can sit inside and be pulled out -- the food-cavity
  analogue of the microwave's :func:`_open_microwave_cavity_mouth`. It is scaled
  down (``FRIDGE_SCALE``), seated
  with its bottom flush on the table top, and yawed +90 deg about Z so the door
  faces world ``-Y``. The door is driven by a *friction-detent*
  :class:`ImplicitActuatorCfg` (``stiffness=0`` -> no setpoint spring; low
  static/dynamic ``friction`` + ``armature`` hold it where the hand leaves it but
  let it swing easily). The food is placed on the interior floor as an offset
  from the fridge base.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR, SYNTHESIS_DIR
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
from ..check_utils import clear_stage_latches
from ..handling import (
    grip_material_event,
    show_by_state,
    tune_hand_for_carrying,
    tune_teleop_for_carrying,
    tune_wrist_camera_bodies,
)
from .checks import (
    FRIDGE_OPEN_RAD,
    FRIDGE_SHUT_RAD,
    MICROWAVE_OPEN_RAD,
    MICROWAVE_SHUT_RAD,
    dial_turned,
    door_at_least,
    door_at_most,
    finished,
    food_placed,
    microwave_running,
    report_status,
)
from .dish import build_handled_dish_usd

logger = logging.getLogger(__name__)


def _open_microwave_cavity_mouth(
    usd_path,
    *,
    door_joint_name: str = "joint_0",
    front_margin: float = 0.03,
    min_block_area: float = 0.08,
    # Invisible box collider (base-local units; scaled with the spawn) that
    # re-closes the plinth BELOW the cavity floor, so a tray resting on the table
    # cannot slide under the cavity. Spans the cavity width at the front, from the
    # body bottom up to the cavity floor; the cavity mouth ABOVE it stays open.
    sill_center_local: tuple[float, float, float] = (-0.395, 0.175, -0.42),
    sill_size_local: tuple[float, float, float] = (0.06, 0.97, 0.154),
    # Give the (mass=0) knob links a small real mass so their joints have enough
    # inertia to be turned by fingertip contact (see KNOB_LINK_MASS).
    knob_link_names: tuple[str, ...] = ("link_1", "link_2"),
    knob_link_mass: float = 0.05,
    # The dial joints are authored unlimited. PhysX ignores limits written at runtime
    # to an unlimited joint, so their stops have to be authored here.
    dial_joint_names: tuple[str, ...] = ("joint_1", "joint_2"),
    dial_limit_deg: float = 90.0,
) -> str:
    """Cache a microwave USD that stays fully solid except for the food-cavity mouth.

    PartNet 7167's ``base`` link is a hollow shell decomposed into convex-hull
    panels. The body front is authored as a few large *solid* slabs (no door hole
    in the mesh), so their convex hulls seal the cavity -> an "invisible wall" no
    object can be placed through. We keep the whole body collidable and deactivate
    only the slabs that actually cover the food-cavity mouth, identified as base
    collider pieces that are ALL of:

    * at/ahead of the door-hinge plane in the base frame
      (max-x <= ``hinge_x + front_margin``) -- i.e. a front-face panel, and
    * a large slab (y-z area > ``min_block_area``) -- not a thin frame/edge lip, and
    * reaching into the cavity (food) side rather than the control-panel side
      (max-y past the body's y-centre).

    Everything else stays solid -- outer skin, inner liner, control-panel front,
    top / bottom / sides / back, the door (``link_0``) and dials
    (``link_1``/``link_2``) -- so the collision covers the whole microwave, only
    the cavity mouth is open, and food rests on the inner floor.

    Removing the front slab also opens the plinth *below* the cavity floor, which
    would let a tray resting on the table slide under the cavity. To prevent that
    we add one invisible box collider (``sill_*_local``) spanning the cavity
    width at the front, from the body bottom up to the cavity floor -- the cavity
    mouth above it stays open for placing food.

    PartNet authors mass=0 on every link; the tiny knob links then have near-zero
    inertia and their joints will not turn under fingertip contact, so we also set
    a small real mass (``knob_link_mass``) on the knob links. The dials are authored
    with no limits, so they get stops at +/- ``dial_limit_deg``.

    The colliders live behind an **instanceable** reference, so the collisions
    group is de-instanced first to make the individual pieces authorable. Result
    is cached next to the source as ``<stem>__cavityprep_v4.usd`` and rebuilt only
    when the source is newer (bump the suffix when this logic changes so stale
    caches are not reused). Returns the source path unchanged on any error /
    missing ``pxr`` / nothing to remove, so it never blocks task load.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__cavityprep_v4.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            return str(src)
        root = default_prim.GetPath().pathString

        # De-instance recursively so every link's collider pieces become
        # authorable. ``stage.Traverse()`` does not recurse into instance proxies,
        # and calling ``SetInstanceable(False)`` mid-iteration invalidates the
        # iterator (PartNet's per-link /collisions Xforms would otherwise stay
        # instanceable -> their Mesh children stay unreachable, so the mesh
        # approximation we author below silently misses them).
        while True:
            paths = [p.GetPath() for p in stage.Traverse() if p.IsInstanceable()]
            if not paths:
                break
            for p in paths:
                stage.GetPrimAtPath(p).SetInstanceable(False)

        # Opening plane = the door hinge x in the base frame (joint body0=base).
        hinge_x = None
        for prim in stage.Traverse():
            if prim.GetName() == door_joint_name and UsdPhysics.Joint(prim):
                local_pos0 = prim.GetAttribute("physics:localPos0").Get()
                if local_pos0 is not None:
                    hinge_x = float(local_pos0[0])
                break
        if hinge_x is None:
            return str(src)
        front_thresh = hinge_x + front_margin

        base_collisions = stage.GetPrimAtPath(f"{root}/base/collisions")
        if not base_collisions.IsValid():
            return str(src)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )

        # First pass: gather collider shapes + the body's overall y-extent (the
        # food cavity is on one y-half, the control panel/knobs on the other).
        shapes = []  # (prim, world_range)
        y_min, y_max = float("inf"), float("-inf")
        for prim in Usd.PrimRange(base_collisions):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if world_range.IsEmpty():
                continue
            shapes.append((prim, world_range))
            y_min = min(y_min, world_range.GetMin()[1])
            y_max = max(y_max, world_range.GetMax()[1])
        if not shapes:
            return str(src)
        y_center = 0.5 * (y_min + y_max)

        # Second pass: deactivate only the large front slabs over the cavity mouth.
        removed = 0
        for prim, world_range in shapes:
            box_min, box_max = world_range.GetMin(), world_range.GetMax()
            yz_area = (box_max[1] - box_min[1]) * (box_max[2] - box_min[2])
            is_front = box_max[0] <= front_thresh
            is_large = yz_area > min_block_area
            on_cavity_side = box_max[1] > y_center
            if is_front and is_large and on_cavity_side:
                prim.SetActive(False)
                removed += 1

        if removed == 0:
            return str(src)

        # Re-close the plinth below the cavity floor with an invisible box
        # collider, so a tray on the table can't slide under the open cavity
        # mouth (the mouth above the floor stays clear for placing food).
        sill = UsdGeom.Cube.Define(stage, f"{root}/base/cavity_front_sill")
        sill.GetSizeAttr().Set(1.0)
        sill.AddTranslateOp().Set(Gf.Vec3d(*sill_center_local))
        sill.AddScaleOp().Set(Gf.Vec3f(*sill_size_local))
        UsdGeom.Imageable(sill).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(sill.GetPrim())

        # Give the (mass=0) knob links a real mass so their near-zero-inertia
        # joints can actually be turned by fingertip contact.
        for knob in knob_link_names:
            knob_prim = stage.GetPrimAtPath(f"{root}/{knob}")
            if not knob_prim.IsValid():
                continue
            mass_api = UsdPhysics.MassAPI.Apply(knob_prim)
            mass_attr = mass_api.GetMassAttr() or mass_api.CreateMassAttr()
            mass_attr.Set(float(knob_link_mass))

        for prim in stage.Traverse():
            if prim.GetName() in dial_joint_names and prim.IsA(UsdPhysics.RevoluteJoint):
                dial = UsdPhysics.RevoluteJoint(prim)
                dial.GetLowerLimitAttr().Set(-float(dial_limit_deg))
                dial.GetUpperLimitAttr().Set(float(dial_limit_deg))

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Opened microwave cavity mouth in %s -> %s (removed %d front slab(s); "
            "added plinth sill; set knob mass=%s; rest of body kept solid).",
            src.name,
            cleaned.name,
            removed,
            knob_link_mass,
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Microwave cavity-mouth prep failed for %s: %s", src, exc)
        return str(src)


def _open_fridge_cavity(
    usd_path,
    *,
    body_link: str = "link_0",
    wall_thickness: float = 0.03,
    full_height_frac: float = 0.6,
) -> str:
    """Cache a fridge USD whose body cavity is hollow and reachable from the front.

    The same "solid box" problem as the raw microwave, but it cannot be solved the
    same way. PartNet 10797's body (``link_0``) collision is a single *open-box*
    mesh, and the convex hull of an open box is the SOLID box -- so a single
    ``convexHull`` collider, OR any ``convexDecomposition`` of it (the vertex cloud
    is already almost convex, so VHACD returns ~one hull), seals the interior. A
    hand then can't reach in to grab the food. Unlike the microwave -- whose
    PartNet body was pre-split into separate convex front panels we could simply
    deactivate (see :func:`_open_microwave_cavity_mouth`) -- the fridge body is one
    inseparable mesh, so we rebuild the cavity instead.

    Mirroring the microwave fix in spirit (keep the body solid everywhere EXCEPT
    the food opening), we:

    * deactivate the full-height sealing collider on ``link_0`` (the thin
      shelf/floor piece, z-span < ``full_height_frac`` of the body height, is kept
      so food still rests on the real shelf), and
    * re-author the back (+X), both sides (+/-Y) and top (+Z) as thin invisible
      box colliders flush with the body's outer shell, leaving the -X front face
      (the door opening) OPEN.

    The walls are authored under ``base`` (which has an identity world transform,
    so the measured world-space extents can be used directly; ``link_0`` itself is
    rotated). ``base`` is fixed to ``link_0`` by a fixed joint and self-collisions
    are disabled, so the walls form one static cavity with the shelf while the food
    (a separate body) is contained by them. The door (``link_1``) keeps its own
    colliders, so a CLOSED door still seals the front and an OPEN door exposes the
    mouth.

    The link geometry lives behind **instanceable** references, so every
    instanceable prim is de-instanced first to make the collider pieces
    authorable. Result is cached next to the source as ``<stem>__cavity_v1.usd``
    and rebuilt only when the source is newer (bump the suffix when this logic
    changes). Returns the source path unchanged on any error / missing ``pxr`` /
    nothing to remove so it never blocks task load.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__cavity_v1.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            return str(src)
        root = default_prim.GetPath().pathString

        # De-instance recursively (same reason as the microwave prep): each link's
        # /collisions Xform is instanceable, so its collider pieces are not
        # authorable until flipped, and flipping mid-Traverse invalidates the
        # iterator. Snapshot + repeat until no instanceable prims remain.
        while True:
            paths = [p.GetPath() for p in stage.Traverse() if p.IsInstanceable()]
            if not paths:
                break
            for p in paths:
                stage.GetPrimAtPath(p).SetInstanceable(False)

        body_collisions = stage.GetPrimAtPath(f"{root}/{body_link}/collisions")
        if not body_collisions.IsValid():
            return str(src)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )

        # Body outer bbox (world) + per-piece z-spans. The full-height piece is the
        # sealing cabinet hull; thin pieces are the shelf/floor we keep.
        shapes = []  # (prim, world_range)
        x0 = y0 = z0 = float("inf")
        x1 = y1 = z1 = float("-inf")
        for prim in Usd.PrimRange(body_collisions):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if world_range.IsEmpty():
                continue
            shapes.append((prim, world_range))
            mn, mx = world_range.GetMin(), world_range.GetMax()
            x0, y0, z0 = min(x0, mn[0]), min(y0, mn[1]), min(z0, mn[2])
            x1, y1, z1 = max(x1, mx[0]), max(y1, mx[1]), max(z1, mx[2])
        if not shapes:
            return str(src)
        height = z1 - z0

        # Deactivate the full-height sealing collider(s); keep the thin shelf/floor.
        removed = 0
        for prim, world_range in shapes:
            mn, mx = world_range.GetMin(), world_range.GetMax()
            if (mx[2] - mn[2]) >= full_height_frac * height:
                prim.SetActive(False)
                removed += 1
        if removed == 0:
            return str(src)

        # Re-author the solid faces as thin invisible box colliders flush with the
        # outer shell, leaving the -X (front / door) face open so the hand can
        # reach into the cavity. ``x0`` is the front (door) face, ``x1`` the back.
        t = wall_thickness
        xc, yc, zc = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        xs, ys, zs = (x1 - x0), (y1 - y0), (z1 - z0)
        walls = {
            "cavity_back": ((x1 - 0.5 * t, yc, zc), (t, ys, zs)),  # +X back
            "cavity_left": ((xc, y0 + 0.5 * t, zc), (xs, t, zs)),  # -Y side
            "cavity_right": ((xc, y1 - 0.5 * t, zc), (xs, t, zs)),  # +Y side
            "cavity_top": ((xc, yc, z1 - 0.5 * t), (xs, ys, t)),  # +Z top
        }
        for name, (center, size) in walls.items():
            cube = UsdGeom.Cube.Define(stage, f"{root}/base/{name}")
            cube.GetSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdGeom.Imageable(cube).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Opened fridge cavity in %s -> %s (deactivated %d sealing collider(s); "
            "added back/sides/top wall colliders; -X front left open for the hand).",
            src.name,
            cleaned.name,
            removed,
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Fridge cavity prep failed for %s: %s", src, exc)
        return str(src)


# =====================================================================
# USD asset paths (the food is prepped into a rigid, collidable copy)
# =====================================================================

# PartNet 7167's body front is authored as solid slabs whose convex hulls seal
# the food cavity (an "invisible wall"). Bake a copy that removes ONLY the slabs
# over the cavity mouth: the mouth opens so food can be placed inside, while the
# whole rest of the body stays solid -- outer skin, inner liner, control-panel
# front, top / bottom / sides / back -- plus the door / dials keep their
# colliders and the body stays visible.
MICROWAVE_USD_PATH = _open_microwave_cavity_mouth(PARTNET_MOBILITY_ARTICULATIONS_DIR / "microwave" / "7167.usd")
# A dish of stew with a side handle (see dish.py). The original food, a flat Meshy
# tray of stew, had nothing a single hand could hold and was tipped over in teleop.
FOOD_USD_PATH = build_handled_dish_usd(SYNTHESIS_DIR / "micro_food")
# Minifridge (PartNet 10797): a small fridge that sits ON the table. Its body
# (``link_0``) is one open-box collision mesh whose convex hull is the SOLID box,
# so a single hull (or any convex decomposition of it) seals the interior and the
# hand can't reach in. Bake a copy that rebuilds the cavity: drop the sealing hull
# and re-author back/sides/top wall colliders, leaving the front (door) face open
# -- the food-cavity equivalent of the microwave cavity-mouth fix. The door is
# ``joint_1`` (revolute); ``base``/``link_0`` form the body, ``link_1`` the door.
FRIDGE_USD_PATH = _open_fridge_cavity(PARTNET_MOBILITY_ARTICULATIONS_DIR / "fridge" / "10797" / "mobility.usd")


# =====================================================================
# Microwave geometry, joints, and the friction-detent joint drive
# =====================================================================

MICROWAVE_SCALE = (0.35, 0.35, 0.35)
MICROWAVE_HALF_HEIGHT_EST = 0.17
MICROWAVE_CENTER_X = 0.3
# Shifted toward the fridge (-Y) so the retrieve->place hop is short on the
# smaller table (the -Y table edge is now at y=-0.50).
MICROWAVE_CENTER_Y = -0.15
MICROWAVE_Z_OFFSET = 0.0

# PartNet 7167 joints: joint_0 door (revolute Y, 0..90 deg); joint_1/joint_2 are
# the two stacked control dials (revolute Z). Turning EITHER dial completes the
# task (joint_1 = lower, joint_2 = upper).
DOOR_JOINT = "joint_0"
LOWER_KNOB_JOINT = "joint_1"
KNOB_JOINT = "joint_2"

# Door spawns CLOSED; the policy must open it to load the food, then shut it.
MICROWAVE_DOOR_INIT_RAD = 0.0

# Friction-detent drive applied to all three revolute joints (stiffness=0 -> no
# spring-back setpoint; static/dynamic friction + damping hold each joint where
# the hand leaves it). TUNE DOOR_STATIC_FRICTION FIRST if the door sags open or
# the hand can't open it.
DOOR_STATIC_FRICTION = 0.2
DOOR_DYNAMIC_FRICTION = 0.1
DOOR_DAMPING = 2.0
# Small dials (~2.7 cm lever arm): keep friction/damping LOW so a fingertip can
# actually turn them -- even modest joint friction is unrotatable at that radius.
# The dials rotate about the vertical Z axis, so gravity exerts no torque about
# it and they still hold where the hand leaves them. RAISE these if a dial spins
# too freely / gets knocked; LOWER them further if it is still hard to turn.
KNOB_STATIC_FRICTION = 0.02
KNOB_DYNAMIC_FRICTION = 0.01
KNOB_DAMPING = 0.05
# PartNet authors mass=0 on every link. The big door still gets a usable mass
# from its large colliders, but the tiny knob colliders give a near-zero inertia,
# so the articulation solve can't turn the dial from fingertip contact (it reads
# as "not rotatable" no matter how low the friction). Armature adds effective
# rotor inertia to condition that solve; the prep step also gives the knob LINKS
# a small real mass (see KNOB_LINK_MASS / the USD prep). RAISE armature toward
# 0.01 if a dial jitters; LOWER it if the dial feels too sluggish to turn.
KNOB_ARMATURE = 0.005
KNOB_LINK_MASS = 0.05

# Both dials stop at +/-90 deg, authored by the asset prep (they start at 0,
# mid-range) so they can be turned EITHER way. Success is turning either dial by
# ``checks.DIAL_ON_RAD`` from its start, in either direction (+/- 18 deg).


# =====================================================================
# Minifridge geometry, joint, and the friction-detent door drive
# =====================================================================

# PartNet 10797 minifridge: authored ~1.07 x 0.89 x 1.52 m with the body min at
# local z=-0.746 and the door (link_1) on the local -X face. It sits ON the table
# (bottom flush with the table top) at FRIDGE_SCALE, yawed +90 deg about Z so the
# door faces world -Y. The door is revolute ``joint_1`` (0..180 deg);
# ``base``+``link_0`` are the fixed body, ``link_1`` the door.
FRIDGE_SCALE_S = 0.35
FRIDGE_SCALE = (FRIDGE_SCALE_S, FRIDGE_SCALE_S, FRIDGE_SCALE_S)
FRIDGE_YAW = math.pi / 2.0  # local -X door -> world -Y
FRIDGE_ROT = R.from_euler("z", FRIDGE_YAW).as_quat(scalar_first=True)  # (w, x, y, z)
FRIDGE_BOTTOM_LOCAL_Z = -0.746  # asset min-z; bottom sits on the table
# On-table xy (ROUGH -- tune to taste). Placed +Y (behind) the microwave: the
# fridge door (faces -Y, hinge on its +X edge) swings its free edge toward -Y/+X,
# down into the microwave door's swing region, so the OPEN fridge door fouls the
# microwave door -- the fridge must be shut before the microwave can be opened.
# The two closed bodies stay clear of each other in y.
FRIDGE_CENTER_X = -0.2
FRIDGE_CENTER_Y = 0.35

FRIDGE_DOOR_JOINT = "joint_1"
FRIDGE_DOOR_LIMIT = math.radians(180.0)  # authored door range 0..180 deg

# Friction-detent door drive: stiffness=0 (no spring-back setpoint), LOW static /
# dynamic friction + damping so the door is easy to swing, small armature to
# condition the solve. RAISE friction if the door swings too freely / gets
# knocked; LOWER it if the hand cannot move the door easily.
FRIDGE_STATIC_FRICTION = 0.1
FRIDGE_DYNAMIC_FRICTION = 0.05
FRIDGE_ARMATURE = 0.001
FRIDGE_DAMPING = 0.2


# =====================================================================
# Food geometry / placement
# =====================================================================

# The dish's root is at the centre of its base, +Z up, handle along local +X. A
# ceramic dish of stew weighs about half a kilo. It is yawed -90 deg so the handle
# points at the fridge door (world -Y) and is the first thing the hand reaches.
FOOD_SCALE = (1.0, 1.0, 1.0)
FOOD_MASS = 0.5
FOOD_HALF_HEIGHT_EST = 0.0
FOOD_INIT_ROT = (math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4))
# Placeholder spawn xy (overwritten in __post_init__ to the in-fridge location).
FOOD_INIT_X = 0.0
FOOD_INIT_Y = FRIDGE_CENTER_Y
FOOD_INIT_CLEARANCE_FROM_TABLE = 0.02

# Food spawns INSIDE the minifridge as an offset from the fridge base (asset
# local frame: -X is the door/front). It is dropped just above the interior floor
# and settles onto it. The dish centre sits at world y 0.40, so its handle, which
# reaches 15.9 cm toward the door, ends clear of the closed door's inner face
# (world y 0.213); any closer and the handle spawns inside the door.
FRIDGE_FOOD_FLOOR_LOCAL_Z = -0.746  # interior floor == asset bottom (food rests near table top)
FRIDGE_FOOD_LOCAL_X = 0.143  # toward the back (+X face); the handle reaches the door
FRIDGE_FOOD_LOCAL_Y = 0.0


def _yaw_rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Rotate a planar (x, y) offset by ``yaw`` about +Z (local -> world)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (x * c - y * s, x * s + y * c)


# In-cavity goal for the place-tracking reward (height above the table top; xy is
# the microwave centre). Success uses the cavity box in ``checks.py`` instead.
FOOD_PLACE_Z_OFFSET = 0.04

# Window light shown while the microwave runs (door shut, a dial turned): a thin
# warm panel just in front of the door's glass, which measures y -0.257-0.063 and
# z 0.668-0.869 at x 0.123 in the world with the door shut. Offsets are from the
# microwave root. Visual only, and fixed in place like the microwave itself.
MICROWAVE_LIGHT_OFFSET = (-0.1785, 0.053, -0.0015)
MICROWAVE_LIGHT_SIZE = (0.001, 0.316, 0.197)
MICROWAVE_LIGHT_COLOR = (1.0, 0.55, 0.1)
MICROWAVE_LIT_PRIMS = {1: ("MicrowaveLight",)}

LONG_HORIZON_STAGE_GRAPH_KEY = "long_horizon.microwave_retrieve_place"

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.6


# =====================================================================
# Stage graph: strict, persistent -- one sub-stage per step of the task
# =====================================================================


def stage_progress_reward(env, task_key: str, persistent: bool = True):
    """Dense long-horizon shaping: the per-env stage-completion ratio in [0, 1]."""
    return mdp.stage_progress(env, task_key=task_key, persistent=persistent).squeeze(-1)


DIAL_JOINTS = (LOWER_KNOB_JOINT, KNOB_JOINT)

MICROWAVE_STAGE_GRAPH = mdp.StageGraphSpec(
    stages=(
        mdp.StageSpec(
            name="open_fridge",
            func=door_at_least,
            params={
                "asset_name": "fridge",
                "joint_name": FRIDGE_DOOR_JOINT,
                "angle_rad": FRIDGE_OPEN_RAD,
                "key": "open_fridge",
                "message": "Fridge open. Next: take the food out and shut the fridge.",
            },
        ),
        mdp.StageSpec(
            name="close_fridge",
            func=door_at_most,
            params={
                "asset_name": "fridge",
                "joint_name": FRIDGE_DOOR_JOINT,
                "angle_rad": FRIDGE_SHUT_RAD,
                "key": "close_fridge",
                "message": "Fridge shut. Next: open the microwave.",
            },
            deps=("open_fridge",),
        ),
        mdp.StageSpec(
            name="open",
            func=door_at_least,
            params={
                "asset_name": "microwave",
                "joint_name": DOOR_JOINT,
                "angle_rad": MICROWAVE_OPEN_RAD,
                "key": "open",
                "message": "Microwave open. Next: set the food down inside.",
            },
            # Microwave may only be opened after the fridge has been shut.
            deps=("close_fridge",),
        ),
        mdp.StageSpec(name="place", func=food_placed, deps=("open",)),
        mdp.StageSpec(
            name="close",
            func=door_at_most,
            params={
                "asset_name": "microwave",
                "joint_name": DOOR_JOINT,
                "angle_rad": MICROWAVE_SHUT_RAD,
                "key": "close",
                "message": "Microwave shut. Next: turn either dial.",
            },
            deps=("place",),
        ),
        mdp.StageSpec(
            name="rotate_knob",
            func=dial_turned,
            params={"dial_joints": DIAL_JOINTS},
            deps=("close",),
        ),
    ),
    # Strict order: open_fridge -> close_fridge -> open (microwave) -> place ->
    # close (microwave) -> rotate_knob. The robot must open the fridge to reach the
    # food, SHUT the fridge before opening the microwave (their doors foul each
    # other), then load/close the microwave and turn a dial. Terminal = rotate_knob.
    terminal_stage="rotate_knob",
    ordering_mode="strict",
    success_mode="substage",
)
mdp.register_stage_graph(LONG_HORIZON_STAGE_GRAPH_KEY, MICROWAVE_STAGE_GRAPH, override=True)


def microwave_success(env, task_key: str) -> torch.Tensor:
    """Every stage done, and the finished state still true right now.

    The stage flags latch. Checking the end state live as well means a dish taken out
    again, a door left open or a dial turned back does not count, and makes the
    consecutive steps counted by ``--num_success_steps`` mean something.
    """
    stages_done = mdp.stage_success(env, task_key=task_key, persistent=True)
    finished_now = finished(env, DOOR_JOINT, DIAL_JOINTS)
    flags = mdp.evaluate_stage_graph(env, task_key=task_key, persistent=True)
    report_status(env, ~(stages_done & finished_now), flags, FRIDGE_DOOR_JOINT, DOOR_JOINT, DIAL_JOINTS)
    show_by_state(env, "microwave_light", MICROWAVE_LIT_PRIMS, microwave_running(env, DOOR_JOINT, DIAL_JOINTS).long())
    return stages_done & finished_now


# =====================================================================
# Asset configs
# =====================================================================

MICROWAVE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Microwave",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MICROWAVE_USD_PATH,
        scale=MICROWAVE_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        # Free baseline drive; the friction-detent actuators below own the joints.
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
        pos=(MICROWAVE_CENTER_X, MICROWAVE_CENTER_Y, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={DOOR_JOINT: MICROWAVE_DOOR_INIT_RAD, LOWER_KNOB_JOINT: 0.0, KNOB_JOINT: 0.0},
    ),
    actuators={
        # Door hinge: damped + frictional so it holds open/closed where left.
        "microwave_door_hinge": ImplicitActuatorCfg(
            joint_names_expr=[DOOR_JOINT],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=DOOR_DAMPING,
            friction=DOOR_STATIC_FRICTION,
            dynamic_friction=DOOR_DYNAMIC_FRICTION,
        ),
        # Both control dials: light friction/damping + armature so a fingertip
        # can turn them. Armature conditions the near-zero-inertia knob solve so
        # the joint actually responds to contact (see KNOB_ARMATURE).
        "microwave_knob_dials": ImplicitActuatorCfg(
            joint_names_expr=[LOWER_KNOB_JOINT, KNOB_JOINT],
            effort_limit_sim=50.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=KNOB_DAMPING,
            friction=KNOB_STATIC_FRICTION,
            dynamic_friction=KNOB_DYNAMIC_FRICTION,
            armature=KNOB_ARMATURE,
        ),
    },
)

FRIDGE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Fridge",
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRIDGE_USD_PATH,
        scale=FRIDGE_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        # Free baseline drive; the friction-detent actuator below owns the door.
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
        # z is overwritten in __post_init__ to sit the bottom on the table top.
        pos=(FRIDGE_CENTER_X, FRIDGE_CENTER_Y, 1.0),
        rot=FRIDGE_ROT,
        joint_pos={FRIDGE_DOOR_JOINT: 0.0},  # door starts closed
    ),
    actuators={
        # Door hinge: friction-detent drive (stiffness=0, low friction + damping,
        # small armature) so the door holds where the hand leaves it but opens
        # easily.
        "fridge_door": ImplicitActuatorCfg(
            joint_names_expr=[FRIDGE_DOOR_JOINT],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=FRIDGE_DAMPING,
            friction=FRIDGE_STATIC_FRICTION,
            dynamic_friction=FRIDGE_DYNAMIC_FRICTION,
            armature=FRIDGE_ARMATURE,
        ),
    },
)

FOOD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=FOOD_USD_PATH,
        scale=FOOD_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            disable_gravity=False,
        ),
        # collision_props left None: keep the dish's authored colliders (passing
        # CollisionPropertiesCfg would clobber them).
        mass_props=sim_utils.MassPropertiesCfg(mass=FOOD_MASS),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(FOOD_INIT_X, FOOD_INIT_Y, 0.0),
        rot=FOOD_INIT_ROT,
    ),
)


# =====================================================================
# Manager configclasses
# =====================================================================


@configclass
class LongHorizonMicrowaveRetrievePlaceCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for long-horizon microwave retrieve-place."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="object",
        resampling_time_range=(20.0, 20.0),
        # No in-cavity placement goal marker rendered.
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(MICROWAVE_CENTER_X, MICROWAVE_CENTER_X),
            pos_y=(MICROWAVE_CENTER_Y, MICROWAVE_CENTER_Y),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        position_only=True,
    )


@configclass
class LongHorizonMicrowaveRetrievePlaceObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation terms for long-horizon microwave retrieve-place.

    ``state`` (observable, no velocities): object pose, microwave door/dial
    joint positions, fridge door joint position, and stage progress.
    ``privileged``: object linear and angular velocities (+ inherited robot
    ``joint_vel`` / ``hand_tips_state_b``). ``policy`` stays as the base's
    last-action-only group; ``proprio`` stays as the base's joint-pos-only
    group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        # Door (joint_0) and the rotated upper dial (joint_2) angles.
        microwave_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("microwave", joint_names=[DOOR_JOINT, KNOB_JOINT])},
        )
        # Fridge door (joint_1) angle.
        fridge_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("fridge", joint_names=[FRIDGE_DOOR_JOINT])},
        )
        stage_progress = ObsTerm(
            func=mdp.stage_progress,
            params={
                "task_key": LONG_HORIZON_STAGE_GRAPH_KEY,
                "persistent": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class LongHorizonMicrowaveRetrievePlaceRewardsCfg(dexverse_base_env.RewardsCfg):
    """Per-step regularization + behaviour shaping + per-stage dense/sparse credit.

    Note the door_open shaping mildly opposes the later ``close`` stage; the
    dense ``stage_progress`` (weight 3.0) and sparse ``success`` (weight 10.0)
    rewards dominate once it is time to shut the door and turn the dial.
    """

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.0005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0005)

    # Bootstrap opening the door (range progress of joint_0 over its 0..pi/2 span).
    door_open = RewTerm(
        func=mdp.joint_range_progress,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("microwave", joint_names=[DOOR_JOINT])},
    )

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

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    # Drive the food toward the in-cavity placement goal.
    place_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=2.0,
        params={"std": 0.12, "command_name": "object_pose"},
    )

    # Reward turning EITHER dial away from its start, in either direction
    # (joint_range_progress_from_init = |angle - start| / reachable, max over the
    # listed joints).
    knob_progress = RewTerm(
        func=mdp.joint_range_progress_from_init,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("microwave", joint_names=[LOWER_KNOB_JOINT, KNOB_JOINT])},
    )

    stage_progress = RewTerm(
        func=stage_progress_reward,
        weight=3.0,
        params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY, "persistent": True},
    )

    success = RewTerm(
        func=mdp.stage_success_reward,
        weight=10.0,
        params={
            "task_key": LONG_HORIZON_STAGE_GRAPH_KEY,
            "persistent": True,
        },
    )


@configclass
class LongHorizonMicrowaveRetrievePlaceTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Time-out (base) + food out-of-bound + terminal-stage (rotate_knob) success."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.60, 0.60), "y": (-0.85, 0.85), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.stage_success,
        params={
            "task_key": LONG_HORIZON_STAGE_GRAPH_KEY,
            "persistent": True,
        },
    )


@configclass
class LongHorizonMicrowaveRetrievePlaceEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for long-horizon microwave retrieve-place."""

    reset_microwave = EventTerm(
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
            "asset_cfg": SceneEntityCfg("microwave"),
        },
    )

    reset_microwave_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("microwave", joint_names=".*")},
    )

    reset_fridge = EventTerm(
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
            "asset_cfg": SceneEntityCfg("fridge"),
        },
    )

    reset_fridge_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("fridge", joint_names=".*")},
    )

    # Note: the dials' [-90, 90] deg stops are authored at asset-prep time (see
    # _open_microwave_cavity_mouth). A limit event cannot add them: the dials are
    # authored unlimited, and PhysX ignores limits written to an unlimited joint.
    # Note: the cavity-mouth colliders are dropped at asset-prep time (see
    # MICROWAVE_USD_PATH / _open_microwave_cavity_mouth) so the cavity is
    # reachable while the rest of the body stays solid; no runtime
    # collision-toggling event is needed here.

    # Grip, re-applied on every reset for the same reason as the dial limits.
    robot_physics_material = grip_material_event("robot")
    food_physics_material = grip_material_event("object")


@configclass
class LongHorizonMicrowaveRetrievePlaceSceneCfg(dexverse_base_env.SceneCfg):
    """Scene with both the microwave articulation and the food object."""

    microwave: ArticulationCfg = MICROWAVE_CFG
    fridge: ArticulationCfg = FRIDGE_CFG
    object: RigidObjectCfg = FOOD_CFG


@configclass
class LongHorizonMicrowaveRetrievePlaceEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Long-horizon task: open microwave, load food, shut the door, set the dial."""

    supports_object_pose_command: bool = True

    scene: LongHorizonMicrowaveRetrievePlaceSceneCfg = LongHorizonMicrowaveRetrievePlaceSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        microwave=MICROWAVE_CFG,
        fridge=FRIDGE_CFG,
        object=FOOD_CFG,
    )
    commands: LongHorizonMicrowaveRetrievePlaceCommandsCfg = LongHorizonMicrowaveRetrievePlaceCommandsCfg()
    observations: LongHorizonMicrowaveRetrievePlaceObservationsCfg = LongHorizonMicrowaveRetrievePlaceObservationsCfg()
    rewards: LongHorizonMicrowaveRetrievePlaceRewardsCfg = LongHorizonMicrowaveRetrievePlaceRewardsCfg()
    events: LongHorizonMicrowaveRetrievePlaceEventCfg = LongHorizonMicrowaveRetrievePlaceEventCfg()
    terminations: LongHorizonMicrowaveRetrievePlaceTerminationsCfg = LongHorizonMicrowaveRetrievePlaceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Four sub-stages (open -> place -> close -> rotate_knob) -- give room.
        self.episode_length_s = 30.0
        self.is_finite_horizon = True
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        microwave_z = table_top_z + MICROWAVE_HALF_HEIGHT_EST + MICROWAVE_Z_OFFSET
        self.scene.microwave.init_state.pos = (MICROWAVE_CENTER_X, MICROWAVE_CENTER_Y, microwave_z)

        # Minifridge sits ON the table (its authored min-z flush with the table
        # top), door facing -Y.
        fridge_z = table_top_z + (-FRIDGE_BOTTOM_LOCAL_Z) * FRIDGE_SCALE_S
        self.scene.fridge.init_state.pos = (FRIDGE_CENTER_X, FRIDGE_CENTER_Y, fridge_z)
        self.scene.fridge.init_state.rot = FRIDGE_ROT

        # Food spawns INSIDE the minifridge as an offset from the fridge base: the
        # local xy is rotated by the fridge yaw and added to the fridge xy; its z
        # is just above the interior floor so it settles directly onto it.
        fx, fy, fz = self.scene.fridge.init_state.pos
        food_ox, food_oy = _yaw_rotate_xy(
            FRIDGE_FOOD_LOCAL_X * FRIDGE_SCALE_S,
            FRIDGE_FOOD_LOCAL_Y * FRIDGE_SCALE_S,
            FRIDGE_YAW,
        )
        food_x = fx + food_ox
        food_y = fy + food_oy
        floor_z = fz + FRIDGE_FOOD_FLOOR_LOCAL_Z * FRIDGE_SCALE_S
        food_z = floor_z + FOOD_HALF_HEIGHT_EST + FOOD_INIT_CLEARANCE_FROM_TABLE
        self.scene.object.init_state.pos = (food_x, food_y, food_z)
        self.scene.object.init_state.rot = FOOD_INIT_ROT

        # Stage "place" goal: inside the microwave cavity (xy at the microwave
        # centre, z just above the cavity floor / table top).
        mx, my, mz = self.scene.microwave.init_state.pos
        light = tuple(base + offset for base, offset in zip((mx, my, mz), MICROWAVE_LIGHT_OFFSET))
        self.scene.microwave_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MicrowaveLight",
            spawn=sim_utils.CuboidCfg(
                size=MICROWAVE_LIGHT_SIZE,
                visible=False,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=MICROWAVE_LIGHT_COLOR, emissive_color=MICROWAVE_LIGHT_COLOR
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=light),
        )
        self.terminations.success = DoneTerm(
            func=microwave_success, params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY}
        )
        # The state observation evaluates the stage graph as an episode restarts, but
        # presets without it would otherwise carry one attempt's stages into the next.
        self.events.clear_stage_latches = EventTerm(
            func=clear_stage_latches, mode="reset", params={"task_key": LONG_HORIZON_STAGE_GRAPH_KEY}
        )
        goal_z = table_top_z + FOOD_PLACE_Z_OFFSET
        self.commands.object_pose.ranges.pos_x = (mx, mx)
        self.commands.object_pose.ranges.pos_y = (my, my)
        self.commands.object_pose.ranges.pos_z = (goal_z, goal_z)

        if self.events.reset_object is not None:
            self.events.reset_object.func = mdp.reset_articulation_with_supports_uniform
            self.events.reset_object.params.pop("velocity_range", None)
            # The dish rests directly on the fridge shelf. Keep the spawn
            # randomization small so the perturbed/rotated dish and its handle never
            # overlap the closed door or slide off the shelf.
            self.events.reset_object.params["pose_range"] = {
                "x": [-0.015, 0.015],
                "y": [-0.015, 0.015],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi * 0.05, math.pi * 0.05],
            }
            self.events.reset_object.params["support_cfgs"] = []

        # Out-of-bound tracks the table footprint (everything -- fridge, microwave,
        # food path -- is on the table now), with generous margins.
        half_x = table_size[0] * 0.5 + 0.10
        half_y = table_size[1] * 0.5 + 0.10
        self.terminations.object_out_of_bound.params["in_bound_range"] = {
            "x": (-half_x, half_x),
            "y": (-half_y, half_y),
            "z": (BOUND_Z_MIN, BOUND_Z_MAX),
        }

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

        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class LongHorizonMicrowaveRetrievePlaceEnvFloatingDexHandRightCfg(LongHorizonMicrowaveRetrievePlaceEnvCfg):
    """Long-horizon microwave-retrieve-place configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()

        tune_hand_for_carrying(self)
        tune_wrist_camera_bodies(self)
        setup_floating_teleop(self)
        table = self.scene.table
        tune_teleop_for_carrying(self, surface_z=table.init_state.pos[2] + 0.5 * table.spawn.size[2])
