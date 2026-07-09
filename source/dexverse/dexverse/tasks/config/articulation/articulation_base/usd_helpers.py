# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-time USD utilities for articulation tasks.

Provides:

* :func:`ensure_single_articulation_root` -- repair USDs that author
  :class:`UsdPhysics.ArticulationRootAPI` on more than one prim (some
  synthesis / PartNet-style assets do this and IsaacLab aborts with
  ``Failed to find a single articulation when resolving '<...>'``).
* :func:`collect_articulation_usds_from_dir` -- discover the
  ``<id>/mobility.usd`` tree produced by ``partnet_mobility_to_usd.py``,
  optionally filtering by asset id substring or by ``meta.json`` target_slot.

Cleaned USDs are cached next to the original as ``<name>__cleaned.usd`` and
regenerated only when the source's mtime changes.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_USD_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")


def ensure_single_articulation_root(usd_path: str | Path) -> str:
    """Return a USD path that has at most one ``ArticulationRootAPI`` prim.

    If the source is already valid (or pxr is unavailable), the original path is
    returned unchanged.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__cleaned.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        return str(src)

    root_prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(root_prims) <= 1:
        return str(src)

    # Prefer roots that are also rigid bodies (the proper articulation root).
    body_roots = [p for p in root_prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    keepers = body_roots if body_roots else root_prims[:1]
    keep_paths = {p.GetPath() for p in keepers}

    removed = []
    for prim in root_prims:
        if prim.GetPath() in keep_paths:
            continue
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        removed.append(str(prim.GetPath()))

    if not removed:
        return str(src)

    stage.GetRootLayer().Export(str(cleaned))
    logger.info(
        "Cleaned articulation roots in %s -> %s (removed %d spurious roots: %s)",
        src.name,
        cleaned.name,
        len(removed),
        removed,
    )
    return str(cleaned)


def simplify_collision_approximation(
    usd_path: str | Path,
    *,
    replace_with: str | None = "convexDecomposition",
    expensive: tuple[str, ...] = ("sdf",),
) -> str:
    """Return a USD path whose mesh colliders avoid expensive approximations.

    Synthesis / PartNet assets frequently author *every* collider as ``sdf``
    (signed distance field). With no ``sdfResolution`` authored, PhysX cooks
    these at its default 256^3 resolution, which is extremely slow and -- on a
    cold cooking cache, before the first physics step -- routinely manifests as
    a multi-minute stall or a hard hang at scene load. IsaacLab's
    ``UsdFileCfg`` exposes no way to override the approximation at spawn time,
    so we rewrite the USD up-front (mirroring
    :func:`ensure_single_articulation_root`).

    Any prim whose ``UsdPhysics.MeshCollisionAPI`` approximation is in
    ``expensive`` (default: just ``sdf``) is switched to ``replace_with``
    (default ``convexDecomposition`` -- fast to cook and preserves concavity
    well enough for these tabletop objects). The result is cached next to the
    source as ``<stem>__collsimple.usd`` and regenerated only when the source
    changes.

    Pass ``replace_with=None`` to disable (returns the input unchanged). On any
    error (or if pxr is unavailable) the original path is returned, so this
    never makes a task fail to load.
    """
    if replace_with is None:
        return str(usd_path)

    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__collsimple.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)

        changed = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            mca = UsdPhysics.MeshCollisionAPI(prim)
            attr = mca.GetApproximationAttr()
            current = attr.Get() if attr else None
            if current in expensive:
                if not attr:
                    attr = mca.CreateApproximationAttr()
                attr.Set(replace_with)
                changed.append(str(prim.GetPath()))

        if not changed:
            return str(src)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Simplified %d collider(s) in %s -> %s (%s -> %s)",
            len(changed),
            src.name,
            cleaned.name,
            "/".join(expensive),
            replace_with,
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Collision-approximation simplification failed for %s: %s", src, exc)
        return str(src)


def normalize_joint_limits(usd_path: str | Path) -> str:
    """Return a USD path whose revolute/prismatic joints have lower <= upper.

    Some synthesis assets author a joint's ``physics:lowerLimit`` /
    ``physics:upperLimit`` in the wrong order (e.g. ``scissors004``'s right hinge
    is ``[50, 0]`` instead of ``[0, 50]``). PhysX / Isaac Lab keep them as
    authored, which breaks anything that assumes an ascending range -- notably
    the ``progress`` success metric, which divides by
    ``reachable = max(q_init - lower, upper - q_init)``. For an inverted range
    that collapses to ~0, so the task "succeeds" on the tiniest joint motion.

    Any joint whose lower limit exceeds its upper limit is rewritten with the
    two swapped. The result is cached next to the source as ``<stem>__limfix.usd``
    and regenerated only when the source changes. On any error (or if pxr is
    unavailable) the original path is returned, so this never blocks loading.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__limfix.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)

        fixed = []
        for prim in stage.Traverse():
            lo_attr = prim.GetAttribute("physics:lowerLimit")
            hi_attr = prim.GetAttribute("physics:upperLimit")
            if not (lo_attr and hi_attr):
                continue
            lo, hi = lo_attr.Get(), hi_attr.Get()
            if lo is None or hi is None:
                continue
            if lo > hi:
                lo_attr.Set(hi)
                hi_attr.Set(lo)
                fixed.append(f"{prim.GetName()}: [{lo}, {hi}] -> [{hi}, {lo}]")

        if not fixed:
            return str(src)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Normalized %d inverted joint limit(s) in %s -> %s: %s",
            len(fixed),
            src.name,
            cleaned.name,
            "; ".join(fixed),
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Joint-limit normalization failed for %s: %s", src, exc)
        return str(src)


def make_floating_articulation_root(usd_path: str | Path) -> str:
    """Return a USD path whose articulation root is on the root *rigid body*.

    Many PartNet-Mobility assets are authored ``fix_base=true``: the converter
    pins the root link to the world with a global ``root_joint``
    (:class:`UsdPhysics.FixedJoint` with one empty body target) and applies the
    :class:`UsdPhysics.ArticulationRootAPI` *to that joint prim*. PhysX accepts
    this only for a **fixed-base** articulation. For a **floating** articulation
    (``fix_root_link=False``, e.g. a liftable object) PhysX requires the
    articulation root to live on the root rigid body. With the root API stranded
    on the world-fixed joint, the articulation fails to parse entirely::

        Pattern '/World/envs/env_*/Articulation/root_joint'
            did not match any articulations
        AttributeError: 'NoneType' object has no attribute 'shared_metatype'

    This helper rewrites the USD into floating-base form:

    * the global fixed joint(s) (one empty body target — they attach a link to
      the world) are deactivated, freeing the root link, and
    * the ``ArticulationRootAPI`` is moved off of every other prim and applied to
      the root rigid body (the body the global fixed joint attached to the
      world).

    The result is cached next to the source as ``<stem>__floatroot.usd`` and
    regenerated only when the source changes. If the asset has no global fixed
    joint (already floating-friendly), or on any error / if pxr is unavailable,
    the input path is returned unchanged so this never blocks loading.

    Only call this when the task wants a free root (``fix_root_link=False``);
    fixed-base tasks must keep the authored ``root_joint`` + root API in place.
    """
    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__floatroot.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)

        # A "global" fixed joint attaches a single link to the simulation world:
        # exactly one of body0/body1 has a target (mirrors IsaacLab's
        # find_global_fixed_joint_prim). Internal joints (both bodies set) stay.
        global_fixed = []  # (joint_prim, root_body_path)
        for prim in stage.Traverse():
            joint = UsdPhysics.Joint(prim)
            if not joint:
                continue
            body_0 = list(joint.GetBody0Rel().GetTargets())
            body_1 = list(joint.GetBody1Rel().GetTargets())
            if bool(body_0) == bool(body_1):
                continue  # both set (internal) or both empty (degenerate)
            global_fixed.append((prim, (body_0 or body_1)[0]))

        if not global_fixed:
            return str(src)  # already floating-friendly; nothing to relocate.

        root_body_path = global_fixed[0][1]
        root_body_prim = stage.GetPrimAtPath(root_body_path)
        if not root_body_prim or not root_body_prim.IsValid():
            return str(src)

        # Move the articulation root onto the root rigid body, stripping it from
        # everywhere else (notably the world-fixed joint prim).
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.GetPath() != root_body_path:
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if not root_body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(root_body_prim)

        # Free the root link: drop the world-fixed joint(s) from composition.
        for joint_prim, _ in global_fixed:
            joint_prim.SetActive(False)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Converted %s -> %s to floating-base (root API -> %s; deactivated %s)",
            src.name,
            cleaned.name,
            root_body_path,
            ", ".join(str(j.GetPath()) for j, _ in global_fixed),
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Floating-root conversion failed for %s: %s", src, exc)
        return str(src)


def is_usd_file_path(path: str | Path) -> bool:
    """Return True if ``path`` looks like a single USD asset file (by extension)."""
    return str(path).lower().endswith(_USD_EXTENSIONS)


def collect_articulation_usds_from_dir(
    parent_dir: str | Path,
    *,
    articulation_ids: list[str] | None = None,
    target_slot: str | None = None,
) -> list[str]:
    """Discover ``<asset_id>/mobility.usd`` files under ``parent_dir``.

    Mirrors the pattern of :func:`pickup_object.base_cfg.collect_usd_files_from_dir`,
    adapted for the canonical PartNet-Mobility tree produced by
    ``partnet_mobility_to_usd.py``.

    Args:
        parent_dir: Directory containing one ``<asset_id>/`` subdir per asset,
            each holding ``mobility.usd`` (and a sibling ``meta.json`` produced
            by the processing script).
        articulation_ids: Optional explicit list of asset ids to keep
            (substring match against the path, same convention as pickup_object).
        target_slot: Optional ``"target_joint_prismatic"`` /
            ``"target_joint_revolute"`` filter. Reads each asset's
            ``meta.json`` and keeps only those whose ``target_slot`` matches.
            Assets without a parseable ``meta.json`` are dropped if the filter
            is set.

    Returns:
        A sorted list of absolute USD paths.
    """
    parent = Path(parent_dir)
    if not parent.is_dir():
        raise FileNotFoundError(f"USD parent directory does not exist: {parent}")

    candidates: list[str] = []
    for ext in _USD_EXTENSIONS:
        candidates.extend(glob.glob(os.path.join(str(parent), f"**/*{ext}"), recursive=True))
    candidates = sorted(set(candidates))
    candidates = [
        f
        for f in candidates
        if "instanceable_meshes" not in f
        and "configuration" not in os.path.relpath(f, parent).split(os.sep)
        and not os.path.basename(f).startswith(".")
    ]

    if articulation_ids is not None:
        wanted = list(articulation_ids)
        candidates = [f for f in candidates if any(aid in f for aid in wanted)]

    if target_slot is not None:
        kept: list[str] = []
        for usd in candidates:
            meta_path = Path(usd).with_name("meta.json")
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("target_slot") == target_slot:
                kept.append(usd)
        candidates = kept

    if not candidates:
        msg = f"No USD assets found under {parent}"
        if articulation_ids:
            msg += f" matching ids={articulation_ids}"
        if target_slot:
            msg += f" with target_slot={target_slot!r}"
        raise FileNotFoundError(msg)

    return candidates
