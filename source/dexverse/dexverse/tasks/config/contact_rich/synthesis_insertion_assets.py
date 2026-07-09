# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD preparation for synthesis-asset insertion tasks.

The ``synthesis/*`` assets used by the pipette/glassware and pen/pen-holder
insertion tasks ship as multi-body articulations: each part carries its own
:class:`UsdPhysics.RigidBodyAPI` and the parts are glued together with fixed
joints (the pipette tube + bulb, the pen body + cap, …). IsaacLab's
``RigidObject`` resolver needs **one** rigid body under the spawn prim, so we
strip that authored physics structure — exactly like
``grasping/bimanual_lift/usd_helpers.ensure_single_rigid_body``.

On top of the collapse, this helper can re-author the **mesh collision
approximation**. The source assets all ship with ``convexDecomposition``, which
is fine for a held object (it stays a dynamic convex collider) but can seal the
narrow opening of a receptacle so nothing can be inserted. For the fixed
glassware / pen-holder (spawned kinematic) we therefore switch the colliders to
``"none"`` (exact triangle mesh), which keeps the cavity hollow and is a valid
collider type for static/kinematic actors.

After stripping the per-part rigid bodies, a single top-level
:class:`UsdPhysics.RigidBodyAPI` is authored on the default prim. This makes the
asset resolve as exactly one rigid body **and** ensures a rigid body already
exists when IsaacLab activates contact sensors at spawn (that step runs *before*
IsaacLab re-applies its own rigid-body properties, so a body must already be
present). The concrete dynamic/kinematic settings are still (re)applied at spawn
time from the task config.

The cleaned stage is cached next to the source as
``<stem>__<tag>__insertion_prepared_v2.usd`` and regenerated when the source is
newer than the cache. If ``pxr`` is unavailable or the source has nothing to
clean, the original path is returned unchanged.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SUFFIX = "__insertion_prepared_v2.usd"
_VISIBLE_SUFFIX = "__visible_material_v1.usd"


def _safe_usd_identifier(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"Material_{safe}"
    return safe


def prepare_insertion_usd(
    usd_path: str | Path,
    collision_approximation: str | None = None,
) -> str:
    """Collapse a synthesis USD to a single rigid body for insertion tasks.

    Args:
        usd_path: Source USD.
        collision_approximation: When given, every authored
            ``UsdPhysics.MeshCollisionAPI`` approximation is rewritten to this
            token (e.g. ``"none"`` for an exact triangle-mesh receptacle cavity).
            When ``None`` the authored approximation is left untouched.

    Returns:
        Path (as ``str``) to the cleaned/cached USD, or the original path when no
        cleanup is needed or ``pxr`` is unavailable.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    tag = (collision_approximation or "collapsed").replace(" ", "_")
    cleaned = src.with_name(f"{src.stem}__{tag}{_SUFFIX}")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        return str(src)

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
    reapprox_meshes: list[str] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_rigid.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed_art_root.append(str(prim.GetPath()))
        if joint_types and any(prim.IsA(t) for t in joint_types):
            # Deactivate (not delete): survives composition edge cases and is
            # trivially reversible.
            prim.SetActive(False)
            deactivated_joints.append(str(prim.GetPath()))
        if collision_approximation is not None and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set(collision_approximation)
            reapprox_meshes.append(str(prim.GetPath()))

    # Author a single top-level rigid body on the default prim so the asset
    # resolves as exactly one rigid body and contact-sensor activation finds it
    # at spawn time. The concrete dynamic/kinematic flags are (re)applied later
    # by the task's spawn cfg.
    added_rigid = False
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid() and not default_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(default_prim)
        added_rigid = True

    if not (removed_rigid or removed_art_root or deactivated_joints or reapprox_meshes or added_rigid):
        return str(src)

    stage.GetRootLayer().Export(str(cleaned))
    logger.info(
        "Prepared %s -> %s (stripped rigid APIs: %d, art roots: %d, joints: %d, "
        "re-approx: %d -> %s, top-level rigid added: %s).",
        src.name,
        cleaned.name,
        len(removed_rigid),
        len(removed_art_root),
        len(deactivated_joints),
        len(reapprox_meshes),
        collision_approximation,
        added_rigid,
    )
    return str(cleaned)


def prepare_visible_usd(
    usd_path: str | Path,
    *,
    material_name: str = "Visible_Material",
    diffuse_color: tuple[float, float, float] = (0.2, 0.75, 1.0),
    opacity: float = 1.0,
    roughness: float = 0.35,
    metallic: float = 0.0,
    cache_tag: str | None = None,
    target_prim_paths: tuple[str, ...] | None = None,
) -> str:
    """Bind a simple PreviewSurface material directly on every mesh in a USD.

    This is for imported assets whose authored descendant mesh bindings are too
    transparent or otherwise hard to see. A spawn-time visual material may not
    override those bindings in all IsaacLab versions, so this authors a cached
    USD with the mesh bindings changed directly and also writes
    ``displayColor`` / ``displayOpacity`` primvars as a fallback. When
    ``target_prim_paths`` is given, only meshes at or under those prim paths are
    rebound.

    Args:
        usd_path: Source USD.
        material_name: Material prim name under ``<defaultPrim>/materials``.
        diffuse_color: RGB color in ``[0, 1]``.
        opacity: PreviewSurface opacity.
        roughness: PreviewSurface roughness.
        metallic: PreviewSurface metallic value.
        cache_tag: Optional filename tag. Bump this when changing material
            parameters for an already cached asset.
        target_prim_paths: Optional prim paths whose descendant meshes should be
            rebound. When ``None``, every mesh in the USD is rebound.

    Returns:
        Path (as ``str``) to the visible/cached USD, or the original path when
        no mesh exists, the source cannot be opened, or ``pxr`` is unavailable.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    safe_material_name = _safe_usd_identifier(material_name)
    tag = _safe_usd_identifier(cache_tag or safe_material_name).lower()
    visible = src.with_name(f"{src.stem}__{tag}{_VISIBLE_SUFFIX}")
    if visible.exists() and visible.stat().st_mtime >= src.stat().st_mtime:
        return str(visible)

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # type: ignore
    except ImportError:
        return str(src)

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        return str(src)

    target_paths = tuple(Sdf.Path(path) for path in target_prim_paths or ())

    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        material_parent_path = default_prim.GetPath().AppendChild("materials")
    else:
        material_parent_path = Sdf.Path("/materials")
    UsdGeom.Scope.Define(stage, material_parent_path)

    material_path = material_parent_path.AppendChild(safe_material_name)
    shader_path = material_path.AppendChild("PreviewSurface")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = prim.GetPath()
        if target_paths and not any(
            prim_path == target_path or prim_path.HasPrefix(target_path) for target_path in target_paths
        ):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*diffuse_color)])
        gprim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])
        mesh_count += 1

    if mesh_count == 0:
        return str(src)

    stage.GetRootLayer().Export(str(visible))
    logger.info("Prepared visible USD %s -> %s (meshes rebound: %d).", src.name, visible.name, mesh_count)
    return str(visible)
