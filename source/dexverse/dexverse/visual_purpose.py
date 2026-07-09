# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tag prims as USD ``purpose='guide'``.

Cameras (the RTX render product behind ``TiledCamera``) ignore ``guide``
prims by default, while the Kit viewport shows them. Use this to hide
visualization-only assets (camera-body cuboids, goal markers, fingertip
spheres, frame markers, forbidden-zone shells, …) from policy/recorded
RGB without hiding them from the human teleoperator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.markers import VisualizationMarkers


def set_prim_purpose_guide(prim_path: str) -> None:
    """Set ``purpose=guide`` on the prim at ``prim_path`` (no-op if missing).

    Purpose is inheritable in USD namespace, so descendants inherit the
    guide tag unless they override it.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)


def hide_marker_from_cameras(marker: VisualizationMarkers) -> None:
    """Tag a ``VisualizationMarkers`` so its instances are camera-invisible."""
    set_prim_purpose_guide(marker.prim_path)
