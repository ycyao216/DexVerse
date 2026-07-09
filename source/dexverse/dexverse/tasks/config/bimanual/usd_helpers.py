# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-time USD cleanup utilities for bimanual lift tasks.

Some synthesis / scanned USDs ship as fully authored articulations: a root
prim carries :class:`UsdPhysics.ArticulationRootAPI`, several child meshes
carry :class:`UsdPhysics.RigidBodyAPI`, and joint prims (Revolute / Prismatic /
Fixed) glue them together.  For tasks that just want to *lift* the asset as
a single rigid body, all of that internal structure is wrong in three ways
when IsaacLab spawns the USD under a ``prim_path`` and applies its own
top-level ``RigidBodyAPI`` via ``spawn_usd_with_rigid_properties``:

* The :class:`RigidObject` resolver aborts with
  ``Failed to find a single rigid body when resolving '<prim>'. Found
  multiple '[...]' under '<prim>'`` (multiple authored rigid bodies).
* The resolver also aborts with
  ``Found an articulation root when resolving '<prim>' for rigid objects``
  (authored :class:`ArticulationRootAPI`).
* PhysX logs
  ``CreateJoint - you cannot create a joint between a body and itself``
  because, after stripping the internal RigidBodyAPIs, both endpoints of
  each authored joint collapse onto the single top-level rigid body.

:func:`ensure_single_rigid_body` rewrites a cached copy of the USD that has
all three classes of offending authoring removed, so IsaacLab can spawn it
cleanly as one rigid body.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Bumped from ``__rb_cleaned`` when joint/articulation stripping was added,
# so stale single-purpose caches are regenerated with the broader cleanup.
_CLEANED_SUFFIX = "__rigid_collapsed.usd"


def ensure_single_rigid_body(usd_path: str | Path) -> str:
    """Return a USD path cleaned for use as a single rigid body.

    Strips, in place on a cached copy of the source USD:

    * every authored :class:`UsdPhysics.RigidBodyAPI` (so only the
      spawn-applied top-level rigid body survives),
    * every authored :class:`UsdPhysics.ArticulationRootAPI` (so the
      RigidObject resolver doesn't see an articulation under the prim),
    * every authored physics joint prim (Revolute / Prismatic / Fixed /
      Spherical / Distance / generic ``Joint``), by deactivating it —
      deactivation leaves the prim in the layer for diffing but keeps the
      physics plugin from trying to realise it.

    If ``pxr`` is unavailable or the source has no authored physics
    schemas to clean, the original path is returned unchanged.  Otherwise
    the cleaned copy is cached next to the original as
    ``<name>__rigid_collapsed.usd`` and regenerated whenever the source is
    newer than the cache.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}{_CLEANED_SUFFIX}")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        return str(src)

    # Joint prim types we want to silence. ``Joint`` is the base type; the
    # typed subclasses get caught by ``IsA(UsdPhysics.Joint)`` too, but
    # iterating a concrete list keeps the intent explicit and makes logs
    # easier to read.
    joint_types: tuple[type, ...] = tuple(
        t
        for t in (
            getattr(UsdPhysics, "Joint", None),
            getattr(UsdPhysics, "RevoluteJoint", None),
            getattr(UsdPhysics, "PrismaticJoint", None),
            getattr(UsdPhysics, "SphericalJoint", None),
            getattr(UsdPhysics, "DistanceJoint", None),
            getattr(UsdPhysics, "FixedJoint", None),
        )
        if t is not None
    )

    removed_rigid: list[str] = []
    removed_art_root: list[str] = []
    deactivated_joints: list[str] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_rigid.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed_art_root.append(str(prim.GetPath()))
        if joint_types and any(prim.IsA(t) for t in joint_types):
            # Deactivating is preferable to deleting: it survives USD
            # composition edge cases (referenced layers, variant sets)
            # and is trivially reversible if a subsequent task needs the
            # joints back.
            prim.SetActive(False)
            deactivated_joints.append(str(prim.GetPath()))

    if not (removed_rigid or removed_art_root or deactivated_joints):
        return str(src)

    stage.GetRootLayer().Export(str(cleaned))
    logger.info(
        "Collapsed %s -> %s (rigid APIs: %d, articulation roots: %d, joints: %d).",
        src.name,
        cleaned.name,
        len(removed_rigid),
        len(removed_art_root),
        len(deactivated_joints),
    )
    if removed_rigid:
        logger.debug("  removed RigidBodyAPI from: %s", removed_rigid)
    if removed_art_root:
        logger.debug("  removed ArticulationRootAPI from: %s", removed_art_root)
    if deactivated_joints:
        logger.debug("  deactivated joints: %s", deactivated_joints)
    return str(cleaned)
