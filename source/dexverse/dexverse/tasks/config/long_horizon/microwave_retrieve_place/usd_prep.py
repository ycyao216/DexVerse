# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-time USD preparation for the microwave-retrieve-place long-horizon task.

The food asset (``synthesis/micro_food/Meshy_AI_Beef_stew_with_vegeta_...usd``)
is a raw Meshy export: a single high-poly mesh held behind an **instanceable**
reference, with **no** ``CollisionAPI`` and **no** ``RigidBodyAPI``. As shipped
it cannot be grasped or rested on the table (it has no collider), and physics
cannot be authored on it because the geometry is instanceable.

:func:`prepare_rigid_mesh_usd` produces a cached, physics-ready copy: it
de-instances any instanceable prims, (re)authors a chosen mesh-collision
approximation on every ``Mesh``, and applies a single top-level
``RigidBodyAPI`` on the default prim so the asset resolves as exactly one rigid
body. The concrete dynamic/kinematic flags and mass are still (re)applied at
spawn time from the task config -- this only fixes the authored USD structure.

The result is cached next to the source as ``<stem>__<approx>_rigidprep.usd``
and regenerated only when the source is newer. If ``pxr`` is unavailable or the
source has nothing to fix, the original path is returned unchanged, so this
never makes the task fail to load. (This mirrors the helper used by the
oven-bake-salmon task, which preps an analogous raw Meshy food export.)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def disable_link_collisions_usd(
    usd_path: str | Path,
    link_names: tuple[str, ...] = ("base",),
    *,
    suffix: str = "cavityopen",
) -> str:
    """Return a copy of ``usd_path`` with the given links' colliders removed.

    PartNet-Mobility containers (e.g. microwave ``7167``) author every link's
    collider as a set of per-mesh ``convexHull`` pieces. For a hollow body the
    union of those hulls fills the cavity, so an object cannot be placed inside
    -- it hits an "invisible wall". These colliders live behind **instanceable**
    references, so they cannot be toggled at runtime (``stage.TraverseAll()``
    does not descend into instance prototypes, and instance proxies are
    read-only).

    This deactivates the ``<link>/collisions`` group prim for each name in
    ``link_names`` (default: the fixed ``base`` body that forms the cavity),
    dropping those colliders from composition while leaving the link's
    **visuals** and every **other** link's colliders intact (so the door /
    knobs stay collidable and the body stays visible). The result is cached
    next to the source as ``<stem>__<suffix>.usd`` and regenerated only when the
    source is newer. On any error, missing ``pxr``, or nothing to disable, the
    input path is returned unchanged so this never makes the task fail to load.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__{suffix}.usd")
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

        disabled: list[str] = []
        for link in link_names:
            collisions_prim = stage.GetPrimAtPath(f"{root}/{link}/collisions")
            if collisions_prim.IsValid():
                collisions_prim.SetActive(False)  # drop from composition
                disabled.append(link)

        if not disabled:
            return str(src)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Opened cavity in %s -> %s (disabled collisions on links: %s).",
            src.name,
            cleaned.name,
            ", ".join(disabled),
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Cavity-open preparation failed for %s: %s", src, exc)
        return str(src)


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
            to preserve a concave cavity).
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
