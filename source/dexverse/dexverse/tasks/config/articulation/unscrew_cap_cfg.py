# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: unscrew a centrifuge-tube cap (autobio/centrifuge_15ml_screw)."""

import math

import isaaclab.sim as sim_utils
import numpy as np
from dexverse.assets import CORE_ASSETS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom, UsdShade

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

CENTRIFUGE_15ML_SCREW_USD_PATH = str(
    CORE_ASSETS_DIR / "autobio" / "centrifuge_15ml_screw" / "model_centrifuge_15ml_screw.usd"
)


def _add_cap_rotation_markings(prim):
    """Paint asymmetric stripes with thin visual-only surface patches.

    The imported STL has no UVs. Copy selected surface triangles just above
    the cap instead, preserving its instanced mesh and all collision shapes.
    Patches live under cap_link, so they follow the actual cap joint.
    """
    stage = prim.GetStage()
    cap = stage.GetPrimAtPath(f"{prim.GetPath()}/cap_link")
    transforms = UsdGeom.XformCache()
    world_to_cap = transforms.GetLocalToWorldTransform(cap).GetInverse()
    visuals = stage.GetPrimAtPath(f"{cap.GetPath()}/visuals")
    meshes = [p for p in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
    for index, mesh_prim in enumerate(meshes):
        mesh = UsdGeom.Mesh(mesh_prim)
        if any(count != 3 for count in mesh.GetFaceVertexCountsAttr().Get()):
            raise ValueError("UnscrewCap rotation markings expect the asset's triangulated STL mesh.")
        mesh_to_cap = transforms.GetLocalToWorldTransform(mesh_prim) * world_to_cap
        points = np.array([mesh_to_cap.Transform(tuple(map(float, p))) for p in mesh.GetPointsAttr().Get()])
        triangles = points[np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
        centers = triangles.mean(axis=1)
        angles = np.arctan2(centers[:, 1], centers[:, 0])
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-12)
        for name, color, angle, half_width in (
            ("white", (0.9, 0.9, 0.85), 0.0, math.radians(30)),
            ("yellow", (1.0, 0.65, 0.04), math.pi, math.radians(10)),
        ):
            selected = (np.cos(angles - angle) >= math.cos(half_width)) & (lengths > 1e-12)
            # 40 micrometres in asset units avoids coplanar flicker at task scale.
            vertices = triangles[selected] + normals[selected, None, :] * 0.00004
            patch = UsdGeom.Mesh.Define(stage, f"{cap.GetPath()}/RotationMarkings/{name}_{index}")
            patch.CreatePointsAttr(vertices.reshape(-1, 3).tolist())
            patch.CreateFaceVertexCountsAttr([3] * len(vertices))
            patch.CreateFaceVertexIndicesAttr(list(range(len(vertices) * 3)))
            patch.CreateSubdivisionSchemeAttr("none")
            material_path = f"{prim.GetPath()}/Looks/cap_{name}_{index}"
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4)
            material_cfg.func(material_path, material_cfg)
            UsdShade.MaterialBindingAPI.Apply(patch.GetPrim()).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
            )


@clone
def spawn_unscrew_cap(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Replace the imported all-white visuals with distinct plastic finishes.

    Bind above the instanceable visual meshes with stronger-than-descendant
    bindings, so their authored white materials cannot override these colors.
    Only the spawned scene is changed; the shared source USD stays untouched.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for link, color, roughness in (
        ("body_link", (0.55, 0.62, 0.65), 0.45),
        ("cap_link", (0.025, 0.14, 0.55), 0.32),
    ):
        material_path = f"{prim_path}/Looks/{link}_plastic"
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)
        material_cfg.func(material_path, material_cfg)
        # Isaac Lab's bind_visual_material helper skips instance roots. USD
        # allows an inherited binding on the root without de-instancing meshes.
        stage = prim.GetStage()
        visuals = stage.GetPrimAtPath(f"{prim_path}/{link}/visuals")
        UsdShade.MaterialBindingAPI.Apply(visuals).Bind(
            UsdShade.Material(stage.GetPrimAtPath(material_path)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    _add_cap_rotation_markings(prim)
    return prim


@configclass
class UnscrewCapEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Loosen a centrifuge tube cap while the tube body is free on the table.

    Task semantics: one hand stabilizes the tube body, the other applies
    rotational motion on the cap. Success is triggered once the cap has
    rotated enough to be considered opened (no need to fully remove it).
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = CENTRIFUGE_15ML_SCREW_USD_PATH
    # Authored tube is ~2.3 cm diameter and ~12.3 cm tall; scale up to
    # improve finger contact robustness for in-hand bimanual manipulation.
    articulation_scale: tuple = (1.8, 1.8, 1.8)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Rotate +90 deg about Y so the tube lies along world-x, making the
    # cap twist more accessible for one-hand hold + one-hand rotate.
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    # When laid sideways, table clearance is roughly the scaled radius.
    articulation_half_height_est: float = 0.02

    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    success_joint_names: list[str] = ["cap_revolute"]
    # Cap joint range is ~720 deg in the authored USD.
    success_threshold: float = math.radians(700.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.articulation.spawn.func = spawn_unscrew_cap
        self.terminations.success.func = mdp.joint_relative_move
        # Require near-complete unscrewing. Keep a tiny tolerance for
        # physics/contact jitter at the mechanical limit.
        self.terminations.success.params = {
            "threshold": 0.98,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
