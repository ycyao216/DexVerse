# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discover object USDs with sibling ``manipulation_annotations.json`` files and
convert their ``active.place.id_0`` entries into canonical upright placements.

The expected annotation layout is::

    {
      "active": {
        "place": {
          "id_0": {
            "position": [x, y, z],        # supporting-contact point in object local frame
            "rotation": [ax, ay, az],     # object-local axis that should point world +z
            "face": "-z",                  # which local face is the supporting face (informational)
            "dimensions": [...],
            ...
          }
        }
      }
    }

This module is intentionally a lightweight, torch-free layer so it can run at
config-construction time before Isaac Lab is initialised.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ObjectPlacement",
    "IDENTITY_QUAT",
    "list_usd_files",
    "load_place_annotation",
    "load_bounding_box",
    "placement_from_annotation",
    "collect_object_pool",
    "placements_for_num_envs",
]


_USD_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")
_ANNOTATION_FILENAME = "manipulation_annotations.json"
_UNPACKED_DIRNAME = "base_rescaled_unpacked"
_UNPACKED_ROOT_USD = "output.usdc"


def prefer_unpacked_usd(files: Sequence[str]) -> list[str]:
    """Drop ``base_rescaled.usdz`` entries when a patched unpacked sibling exists.

    The ManiTwin pipeline unpacks ``base_rescaled.usdz`` into
    ``base_rescaled_unpacked/output.usdc`` and applies
    ``UsdPhysics.MeshCollisionAPI`` there. The raw ``.usdz`` archive is not
    patched, so if both end up in the asset pool PhysX complains about
    triangle-mesh collision on dynamic bodies. This filter keeps the unpacked
    USD and discards the sibling archive.
    """
    result: list[str] = []
    for f in files:
        if f.lower().endswith(".usdz"):
            sibling = os.path.join(os.path.dirname(f), _UNPACKED_DIRNAME, _UNPACKED_ROOT_USD)
            if os.path.isfile(sibling):
                continue
        result.append(f)
    return result


IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectPlacement:
    """Canonical upright placement derived from a ``place.id_0`` entry.

    Attributes
    ----------
    position_offset:
        Translation from the supporting surface to the object's origin, in the
        world-aligned "placed" frame. Concretely the object's origin sits at
        ``(x + position_offset[0], y + position_offset[1], support_z + position_offset[2])``
        so its annotated supporting-contact point lands on the surface.
    quat:
        Canonical upright quaternion in ``(w, x, y, z)`` order.
    usd_path:
        Absolute path to the USD this placement is associated with (convenience).
    raw:
        The original ``place.id_0`` dict, preserved for downstream use.
    """

    position_offset: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    usd_path: str | None = None
    raw: dict | None = None


def list_usd_files(usd_parent_dir: str, object_ids: Sequence[str] | None = None) -> list[str]:
    """Recursively list USD files under ``usd_parent_dir`` in deterministic order."""
    files: list[str] = []
    for ext in _USD_EXTENSIONS:
        files.extend(glob.glob(os.path.join(usd_parent_dir, f"**/*{ext}"), recursive=True))
    files = sorted({f for f in files if "instanceable_meshes" not in f})
    files = prefer_unpacked_usd(files)
    if object_ids is not None:
        files = [f for f in files if any(oid in f for oid in object_ids)]
    if not files:
        raise ValueError(f"No USD assets found under {usd_parent_dir}")
    return files


def _find_annotation_path(usd_path: str) -> str | None:
    """Find the nearest ``manipulation_annotations.json`` next to ``usd_path``.

    Looks in the USD's directory and one level up (many assets store the JSON
    alongside the object id folder rather than next to the concrete USD).
    """
    parent = Path(usd_path).resolve().parent
    for candidate in (parent, parent.parent):
        ann = candidate / _ANNOTATION_FILENAME
        if ann.is_file():
            return str(ann)
    return None


def _load_annotation_file(usd_path: str) -> dict | None:
    """Load the raw annotation JSON next to ``usd_path``, or ``None``."""
    ann_path = _find_annotation_path(usd_path)
    if ann_path is None:
        return None
    try:
        with open(ann_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_place_annotation(usd_path: str) -> dict | None:
    """Return the ``active.place.id_0`` dict for ``usd_path`` or ``None``."""
    data = _load_annotation_file(usd_path)
    if data is None:
        return None
    active = data.get("active") or {}
    place = active.get("place") or {}
    entry = place.get("id_0")
    if not isinstance(entry, dict):
        return None
    return entry


def load_bounding_box(usd_path: str) -> dict | None:
    """Return the top-level ``bounding_box`` block for ``usd_path`` or ``None``.

    The expected schema is the one produced by the ManiTwin annotation pipeline:
    ``{"min_bounds": [x, y, z], "max_bounds": [x, y, z], ...}`` in the object's
    local frame.
    """
    data = _load_annotation_file(usd_path)
    if data is None:
        return None
    bbox = data.get("bounding_box")
    if not isinstance(bbox, dict):
        return None
    return bbox


def _quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-9:
        return IDENTITY_QUAT
    ax, ay, az = ax / n, ay / n, az / n
    half = 0.5 * angle
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def _quat_align_vec_to_z(
    up_local: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion rotating ``up_local`` to world +z.

    Returned quaternion is in ``(w, x, y, z)`` order.
    """
    ux, uy, uz = up_local
    n = math.sqrt(ux * ux + uy * uy + uz * uz)
    if n < 1e-9:
        return IDENTITY_QUAT
    ux, uy, uz = ux / n, uy / n, uz / n
    dot = uz
    if dot > 1.0 - 1e-6:
        return IDENTITY_QUAT
    if dot < -1.0 + 1e-6:
        # up_local antiparallel to +z: pick a 180 rotation around x.
        return (0.0, 1.0, 0.0, 0.0)
    # axis = up_local x +z  (=> flip object so up_local points up)
    axis = (uy, -ux, 0.0)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    return _quat_from_axis_angle(axis, angle)


def _apply_quat(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate vector ``v`` by quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def _rotated_aabb_min_z(
    quat: tuple[float, float, float, float],
    min_bounds: tuple[float, float, float],
    max_bounds: tuple[float, float, float],
) -> float:
    """Return the minimum world-z over the 8 AABB corners rotated by ``quat``."""
    mn = min(
        _apply_quat(quat, (x, y, z))[2]
        for x in (min_bounds[0], max_bounds[0])
        for y in (min_bounds[1], max_bounds[1])
        for z in (min_bounds[2], max_bounds[2])
    )
    return mn


def placement_from_annotation(
    place: dict | None,
    *,
    bounding_box: dict | None = None,
    usd_path: str | None = None,
) -> ObjectPlacement:
    """Build an :class:`ObjectPlacement` from a ``place.id_0`` dict.

    When ``bounding_box`` is provided (the ManiTwin ``bounding_box`` block with
    ``min_bounds``/``max_bounds`` in the object-local frame), the z-component
    of the resulting ``position_offset`` is taken as the larger of:

    - ``-rotated_pos[2]`` (the contact-point-derived lift, current behaviour); and
    - ``-min_z_after_rotation`` of the eight rotated AABB corners.

    The latter is conservative: the mesh always lies inside the AABB, so this
    guarantees no vertex sinks below the support surface even when the
    ``place.position`` annotation is offset from the true lowest mesh point or
    when the upright quaternion swings a non-z face downward.

    If ``place`` is ``None`` (no annotation available), returns an identity
    placement (zero offset, identity quaternion) -- with a bbox-derived z lift
    if ``bounding_box`` is available, so even unannotated objects clear the
    table.
    """
    if place is None and bounding_box is None:
        return ObjectPlacement(
            position_offset=(0.0, 0.0, 0.0),
            quat=IDENTITY_QUAT,
            usd_path=usd_path,
            raw=None,
        )

    if place is None:
        pos_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
        up_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    else:
        pos_local = tuple(float(v) for v in place.get("position", (0.0, 0.0, 0.0)))
        up_local = tuple(float(v) for v in place.get("rotation", (0.0, 0.0, 1.0)))

    quat = _quat_align_vec_to_z(up_local)
    rotated_pos = _apply_quat(quat, pos_local)

    z_lift = -rotated_pos[2]
    if bounding_box is not None:
        try:
            min_bounds = tuple(float(v) for v in bounding_box["min_bounds"])
            max_bounds = tuple(float(v) for v in bounding_box["max_bounds"])
        except (KeyError, TypeError, ValueError):
            min_bounds = max_bounds = None  # type: ignore[assignment]
        if min_bounds is not None and max_bounds is not None and len(min_bounds) == 3 and len(max_bounds) == 3:
            bbox_lift = -_rotated_aabb_min_z(quat, min_bounds, max_bounds)
            z_lift = max(z_lift, bbox_lift)

    position_offset = (-rotated_pos[0], -rotated_pos[1], z_lift)

    return ObjectPlacement(
        position_offset=position_offset,
        quat=quat,
        usd_path=usd_path,
        raw=dict(place) if place is not None else None,
    )


def collect_object_pool(
    usd_parent_dir: str,
    *,
    object_ids: Sequence[str] | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[list[str], list[ObjectPlacement]]:
    """Discover USDs and their placements under ``usd_parent_dir``.

    Returns ``(usd_paths, placements)`` aligned by index. When ``shuffle=True``
    both lists are shuffled in lockstep using ``seed`` so the same seed gives
    the same (env_idx -> object) mapping across runs.
    """
    usd_files = list_usd_files(usd_parent_dir, object_ids)
    placements = [
        placement_from_annotation(
            load_place_annotation(f),
            bounding_box=load_bounding_box(f),
            usd_path=f,
        )
        for f in usd_files
    ]
    if shuffle:
        rng = random.Random(seed)
        indices = list(range(len(usd_files)))
        rng.shuffle(indices)
        usd_files = [usd_files[i] for i in indices]
        placements = [placements[i] for i in indices]
    return usd_files, placements


def placements_for_num_envs(
    placements: Sequence[ObjectPlacement],
    num_envs: int,
) -> list[ObjectPlacement]:
    """Cycle ``placements`` round-robin so each env index has an aligned entry.

    This matches Isaac Lab's ``MultiAssetSpawnerCfg(random_choice=False)`` which
    assigns ``assets[i % N]`` to environment ``i``.
    """
    if not placements:
        raise ValueError("placements must be non-empty")
    return [placements[i % len(placements)] for i in range(num_envs)]
