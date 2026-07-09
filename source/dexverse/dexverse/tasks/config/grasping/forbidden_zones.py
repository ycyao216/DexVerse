# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared contact-region declarations used across grasping task families.

Zones are no-touch regions defined in an object's local frame. Each task
family translates them into the flat parameter lists consumed by the
``mdp.success_no_forbidden_contact`` /
``mdp.lifted_no_forbidden_contact`` terminations and the
``mdp.forbidden_zones_vis`` observation.

Designated contact zones are the positive counterpart: success can require
configured robot/tool bodies to occupy these object-local regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class ForbiddenZone:
    """A no-touch region attached to the object's local frame.

    - ``kind="sphere"``: ``center`` + ``radius``.
    - ``kind="box"``: ``center`` + ``half_size`` (axis-aligned in obj frame).
    - ``kind="cylinder"``: ``center`` + ``radius`` + ``half_height``. Cylinders
      are aligned with the zone's local z-axis.

    Coordinates are in the object's root frame (i.e. the frame whose pose
    is ``object.data.root_pos_w`` / ``root_quat_w``). ``rotation_offset`` is
    a local quaternion ``(w, x, y, z)`` applied to cylinder zones relative to
    that object frame.
    """

    kind: Literal["sphere", "box", "cylinder"]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    half_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_height: float = 0.0
    rotation_offset: tuple[float, float, float, float] = IDENTITY_QUAT


@dataclass
class DesignatedContactZone:
    """A required-contact region attached to an object's local frame.

    Each zone is satisfied when at least one configured robot/tool body lies
    inside the region. Multiple zones are conjunctive: every declared zone
    must be satisfied for success.
    """

    kind: Literal["sphere", "box", "cylinder"]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    half_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_height: float = 0.0
    rotation_offset: tuple[float, float, float, float] = IDENTITY_QUAT


def split_zones(
    zones: tuple[ForbiddenZone | DesignatedContactZone, ...] | list[ForbiddenZone | DesignatedContactZone],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Split mixed zones into flat sphere/box/cylinder parameter lists for mdp params."""
    sphere_zones: list[list[float]] = []
    box_zones: list[list[float]] = []
    cylinder_zones: list[list[float]] = []
    for z in zones:
        if z.kind == "sphere":
            sphere_zones.append([*z.center, float(z.radius)])
        elif z.kind == "box":
            box_zones.append([*z.center, *z.half_size])
        elif z.kind == "cylinder":
            cylinder_zones.append([*z.center, float(z.radius), float(z.half_height), *z.rotation_offset])
        else:
            raise ValueError(f"Unsupported zone kind: {z.kind!r}")
    return sphere_zones, box_zones, cylinder_zones
