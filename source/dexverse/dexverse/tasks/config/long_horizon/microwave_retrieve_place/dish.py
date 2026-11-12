# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A round dish of stew with a flat side handle, built as a USD asset.

The original food was a 24 x 16 x 4 cm tray with nothing to hold: operators could not
carry it with one hand and kept tipping it. This dish is held like the Cook Foods pot,
by a handle, which operators carry easily and nearly level.

Sizes, in metres, with the root at the centre of the dish's base and +Z up:

* dish: 15 cm across at the rim, 5 cm tall, 7 mm walls. It collides as its convex
  hull, a solid frustum, since nothing needs to go inside it.
* handle: a flat tab 9 cm long, 3.5 cm wide and 1.4 cm thick, flush with the rim,
  along local +X. Its underside is 3.6 cm above the surface the dish rests on, so
  fingers fit under it.
* overall 23.4 cm from the far rim to the handle's end, which fits the 29 cm deep
  microwave cavity with the door shut.

The stew and its chunks are visual only. The file is cached next to the old food
asset and rebuilt only when missing; bump the version in the name when this changes.
"""

from __future__ import annotations

import math
from pathlib import Path

DISH_RIM_RADIUS = 0.075
DISH_HEIGHT = 0.05
HANDLE_LENGTH = 0.09
HANDLE_WIDTH = 0.035
HANDLE_THICKNESS = 0.014
# The handle reaches 6 mm into the wall so the two pieces read as one.
HANDLE_OVERLAP = 0.006
HANDLE_CENTER_X = DISH_RIM_RADIUS + 0.5 * HANDLE_LENGTH - HANDLE_OVERLAP
HANDLE_END_X = HANDLE_CENTER_X + 0.5 * HANDLE_LENGTH

# (radius, height) around the dish, from the base centre out, up the outer wall, over
# the rim and down the inside to the inner floor.
_PROFILE = (
    (0.0, 0.0),
    (0.058, 0.0),
    (0.062, 0.004),
    (DISH_RIM_RADIUS, DISH_HEIGHT),
    (0.068, DISH_HEIGHT),
    (0.054, 0.008),
    (0.0, 0.008),
)
_SEGMENTS = 48
STEW_TOP = 0.035
STEW_RADIUS = 0.063

_CERAMIC = (0.93, 0.92, 0.88)
_STEW = (0.42, 0.22, 0.1)
# (x, y, edge, colour): beef, carrot, potato and peas, half sunk into the stew.
_CHUNKS = (
    (0.02, 0.015, 0.02, (0.32, 0.15, 0.08)),
    (-0.025, 0.02, 0.018, (0.32, 0.15, 0.08)),
    (0.005, -0.03, 0.02, (0.32, 0.15, 0.08)),
    (-0.035, -0.015, 0.016, (0.92, 0.45, 0.1)),
    (0.035, -0.012, 0.015, (0.92, 0.45, 0.1)),
    (0.0, 0.04, 0.017, (0.86, 0.76, 0.5)),
    (-0.01, -0.005, 0.016, (0.86, 0.76, 0.5)),
    (0.03, 0.035, 0.012, (0.35, 0.62, 0.22)),
    (-0.04, 0.025, 0.011, (0.35, 0.62, 0.22)),
)


def _revolve(profile, segments):
    """Points and quads of the surface swept by ``profile`` around +Z."""
    points, counts, indices = [], [], []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        c, s = math.cos(angle), math.sin(angle)
        points.extend((r * c, r * s, z) for r, z in profile)
    rows = len(profile)
    for index in range(segments):
        nxt = (index + 1) % segments
        for k in range(rows - 1):
            counts.append(4)
            indices.extend((index * rows + k, nxt * rows + k, nxt * rows + k + 1, index * rows + k + 1))
    return points, counts, indices


def _material(stage, path: str, color: tuple[float, float, float], roughness: float):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _box(stage, path: str, center, size, material, collide: bool):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    box = UsdGeom.Cube.Define(stage, path)
    box.GetSizeAttr().Set(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box


def build_handled_dish_usd(directory: str | Path) -> str:
    """Write the dish to ``directory`` if it is not there yet, and return its path."""
    path = Path(directory) / "handled_stew_dish_v1.usd"
    if path.exists():
        return str(path)

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Dish")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

    ceramic = _material(stage, "/Dish/Looks/Ceramic", _CERAMIC, 0.3)
    stew = _material(stage, "/Dish/Looks/Stew", _STEW, 0.6)

    points, counts, indices = _revolve(_PROFILE, _SEGMENTS)
    body = UsdGeom.Mesh.Define(stage, "/Dish/Body")
    body.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    body.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    body.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    body.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    body.GetDoubleSidedAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(ceramic)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    _box(
        stage,
        "/Dish/Handle",
        (HANDLE_CENTER_X, 0.0, DISH_HEIGHT - 0.5 * HANDLE_THICKNESS),
        (HANDLE_LENGTH, HANDLE_WIDTH, HANDLE_THICKNESS),
        ceramic,
        collide=True,
    )

    filling = UsdGeom.Cylinder.Define(stage, "/Dish/Stew")
    filling.GetRadiusAttr().Set(STEW_RADIUS)
    filling.GetHeightAttr().Set(STEW_TOP - 0.008)
    filling.GetAxisAttr().Set(UsdGeom.Tokens.z)
    filling.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5 * (STEW_TOP + 0.008)))
    UsdShade.MaterialBindingAPI.Apply(filling.GetPrim()).Bind(stew)
    for index, (x, y, edge, color) in enumerate(_CHUNKS):
        chunk = _material(stage, f"/Dish/Looks/Chunk{index}", color, 0.6)
        _box(stage, f"/Dish/Chunk{index}", (x, y, STEW_TOP), (edge, edge, edge), chunk, collide=False)

    stage.GetRootLayer().Save()
    return str(path)
