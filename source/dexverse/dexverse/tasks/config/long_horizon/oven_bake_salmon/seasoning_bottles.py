"""Locally authored shaker bottles, with simple grasp colliders and detailed visuals.

The cached USDs are generated outside the source asset bundle. A source digest
invalidates the cache whenever the shape, material or lettering changes.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
import tempfile

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .salmon_font import GLYPHS


def _material(stage, name, color, *, roughness=.4, metallic=0., opacity=1.):
    material = UsdShade.Material.Define(stage, "/Bottle/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.46)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _lathe(stage, path, profile, material=None, segments=64):
    points = [(radius * math.cos(2 * math.pi * i / segments),
               radius * math.sin(2 * math.pi * i / segments), z)
              for z, radius in profile for i in range(segments)]
    faces = [(j * segments + i, j * segments + (i + 1) % segments,
              (j + 1) * segments + (i + 1) % segments, (j + 1) * segments + i)
             for j in range(len(profile) - 1) for i in range(segments)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _letters(stage, path, word, material, back=False):
    # Lettering is real geometry, so it remains legible without texture paths.
    size = .006 / 32
    advance = 23
    points = []
    for index, char in enumerate(word):
        bitmap = GLYPHS[char]
        for row in range(32):
            bits = int.from_bytes(bitmap[row * 4:(row + 1) * 4], "big")
            for col in range(32):
                if bits & (1 << (31 - col)):
                    y = ((len(word) - 1) * advance / 2 + 16 - index * advance - col) * size
                    z = .060 - row * size
                    def point(y_pos, z_pos):
                        angle = y_pos / .0256
                        return (-.0256 * math.cos(angle), .0256 * math.sin(angle), z_pos)
                    quad = [point(y, z), point(y - size, z), point(y - size, z - size), point(y, z - size)]
                    points.extend([(-a, -b, c) for a, b, c in quad] if back else quad)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (len(points) // 4))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def seasoning_bottle_usd(kind: str) -> str:
    """Return a 5 cm wide, 14 cm tall salt or pepper shaker, base at z=0."""
    if kind not in ("salt", "pepper"):
        raise ValueError(kind)
    digest = hashlib.sha256(Path(__file__).read_bytes() + repr(GLYPHS).encode()).hexdigest()[:12]
    folder = Path(tempfile.gettempdir()) / "dexverse-seasoning-bottles" / digest
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{kind}.usdc"
    if output.is_file():
        return str(output)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.15)
    salt = kind == "salt"
    # Opaque glass tint keeps the neck visible in the collection performance
    # renderer, which does not support this PreviewSurface transparency path.
    glass = _material(stage, "Glass", (.65, .76, .72) if salt else (.30, .32, .24), roughness=.19)
    contents = _material(stage, "Seasoning", (.89, .86, .74) if salt else (.10, .065, .036), roughness=.85)
    grain = _material(stage, "Grains", (.99, .98, .91) if salt else (.025, .016, .01), roughness=.95)
    lid = _material(stage, "Cap", (.65, .69, .71) if salt else (.075, .065, .055),
                    roughness=.29 if salt else .45, metallic=.8 if salt else .1)
    label = _material(stage, "Label", (.92, .88, .74) if salt else (.16, .075, .035), roughness=.8)
    ink = _material(stage, "Ink", (.075, .12, .12) if salt else (.95, .86, .64), roughness=.8)
    hole = _material(stage, "ShakerHoles", (.015, .015, .015), roughness=.95)
    profile = [(0., 0.), (0., .020), (.002, .023), (.007, .025), (.082, .025),
               (.091, .024), (.100, .020), (.108, .015), (.123, .015), (.123, 0.)]
    _lathe(stage, "/Bottle/Glass", profile, glass)
    _lathe(stage, "/Bottle/Contents", [(.006, 0.), (.008, .02515), (.080, .02515), (.082, 0.)], contents)
    _lathe(stage, "/Bottle/PaperLabel", [(.032, .0254), (.070, .0254)], label)
    _letters(stage, "/Bottle/NameFront", kind.upper(), ink)
    _letters(stage, "/Bottle/NameBack", kind.upper(), ink, back=True)
    _lathe(stage, "/Bottle/Cap", [(.117, 0.), (.117, .017), (.120, .019),
                                  (.136, .019), (.140, .016), (.140, 0.)], lid)
    # Raised grip ribs and a perforated shaker top make the closures recognisable.
    for i in range(36):
        angle = i * 2 * math.pi / 36
        rib = UsdGeom.Cylinder.Define(stage, f"/Bottle/CapRib{i}")
        rib.CreateRadiusAttr(.00065)
        rib.CreateHeightAttr(.014)
        rib.AddTranslateOp().Set(Gf.Vec3d(.019 * math.cos(angle), .019 * math.sin(angle), .127))
        UsdShade.MaterialBindingAPI.Apply(rib.GetPrim()).Bind(lid)
    for i in range(7):
        angle = i * math.pi / 3
        r = 0. if i == 6 else .0085
        spot = UsdGeom.Cylinder.Define(stage, f"/Bottle/ShakerHole{i}")
        spot.CreateRadiusAttr(.0015)
        spot.CreateHeightAttr(.0002)
        spot.AddTranslateOp().Set(Gf.Vec3d(r * math.cos(angle), r * math.sin(angle), .14015))
        UsdShade.MaterialBindingAPI.Apply(spot.GetPrim()).Bind(hole)
    # Instanced grains are visual only; the simulator sees one convex body.
    particle = UsdGeom.Sphere.Define(stage, "/Bottle/Grains/Prototype")
    particle.CreateRadiusAttr(.00045 if salt else .0009)
    UsdShade.MaterialBindingAPI.Apply(particle.GetPrim()).Bind(grain)
    grains = UsdGeom.PointInstancer.Define(stage, "/Bottle/Grains")
    grains.CreatePrototypesRel().SetTargets([particle.GetPath()])
    rng = random.Random(29)
    positions = []
    for i in range(140):
        angle = rng.uniform(0., 2 * math.pi)
        # Expose grains only through the two contents bands, never through paper.
        z = rng.uniform(.010, .029) if i < 100 else rng.uniform(.074, .080)
        positions.append((.025 * math.cos(angle), .025 * math.sin(angle), z))
    grains.CreatePositionsAttr().Set(positions)
    grains.CreateProtoIndicesAttr().Set([0] * len(positions))
    collider = _lathe(stage, "/Bottle/Collider", profile[:-1] + [(.140, .019), (.140, 0.)])
    collider.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
    fd, temporary = tempfile.mkstemp(dir=folder, suffix=".usdc")
    os.close(fd)
    try:
        stage.GetRootLayer().Export(temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(output)
