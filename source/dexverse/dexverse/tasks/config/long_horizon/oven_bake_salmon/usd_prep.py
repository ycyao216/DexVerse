# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-time USD preparation for the oven-bake-salmon long-horizon task.

Two of this task's assets are *not* physics-ready as shipped:

* ``salmon/salmon.usd`` is a raw Meshy export -- a single high-poly mesh held
  behind an **instanceable** reference, with **no** ``CollisionAPI`` and **no**
  ``RigidBodyAPI``. As-is it cannot be grasped or rested in the tray (it has no
  collider) and physics cannot be authored on it (the geometry is instanceable).
* ``tray001/model_redtray__rb_cleaned.usd`` authors its collider as ``sdf``,
  which PhysX cooks at a 256^3 default that routinely stalls scene load, and it
  carries no ``RigidBodyAPI``.

:func:`prepare_rigid_mesh_usd` produces a cached, physics-ready copy of either:
it de-instances any instanceable prims, (re)authors a chosen mesh-collision
approximation on every ``Mesh``, and applies a single top-level
``RigidBodyAPI`` on the default prim so the asset resolves as exactly one rigid
body (and a body exists when IsaacLab activates contact sensors at spawn, which
happens *before* it re-applies its own rigid-body properties). The concrete
dynamic/kinematic flags and mass are still (re)applied at spawn time from the
task config -- this only fixes the authored USD structure.

The result is cached next to the source as ``<stem>__<approx>_rigidprep.usd``
and regenerated only when the source is newer. If ``pxr`` is unavailable or the
source has nothing to fix, the original path is returned unchanged, so this
never makes the task fail to load.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_rigid_mesh_usd(
    usd_path: str | Path,
    *,
    approximation: str = "convexHull",
    add_rigid_body: bool = True,
) -> str:
    """Return a path to a physics-ready single-rigid-body copy of ``usd_path``.

    Args:
        usd_path: Source USD.
        approximation: Mesh-collision approximation authored on every ``Mesh``
            (e.g. ``"convexHull"`` for a graspable object, ``"convexDecomposition"``
            to preserve a concave cavity like a tray).
        add_rigid_body: When ``True`` a single ``RigidBodyAPI`` is applied to the
            default prim if one is not already present.

    Returns:
        Path (as ``str``) to the cleaned/cached USD, or the original path when no
        changes are needed or ``pxr`` is unavailable.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__{approximation}_rigidprep.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)

        # 1. De-instance so physics APIs can be authored on the (now unique)
        #    descendants of the instanceable reference.
        deinstanced: list[str] = []
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                deinstanced.append(str(prim.GetPath()))

        # 2. (Re)author collision + the chosen approximation on every mesh.
        meshes: list[str] = []
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approx_attr = mca.GetApproximationAttr() or mca.CreateApproximationAttr()
            approx_attr.Set(approximation)
            meshes.append(str(prim.GetPath()))

        # 3. One top-level rigid body on the default prim.
        added_rigid = False
        default_prim = stage.GetDefaultPrim()
        if (
            add_rigid_body
            and default_prim
            and default_prim.IsValid()
            and not default_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            UsdPhysics.RigidBodyAPI.Apply(default_prim)
            added_rigid = True

        if not (deinstanced or meshes or added_rigid):
            return str(src)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Prepared rigid mesh %s -> %s (de-instanced: %d, meshes: %d -> %s, top-level rigid added: %s).",
            src.name,
            cleaned.name,
            len(deinstanced),
            len(meshes),
            approximation,
            added_rigid,
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Rigid-mesh preparation failed for %s: %s", src, exc)
        return str(src)
