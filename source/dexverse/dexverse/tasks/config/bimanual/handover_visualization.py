# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, non-colliding handover status panels visible in the XR render."""

from __future__ import annotations

import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdShade


from .progress_font import GLYPHS, GLYPH_ADVANCES, GLYPH_SIZE

# State order: waiting, source pickup, receiver hold, success, invalid, previous success.
_COLORS = (
    (0.7, 0.75, 0.8), (1.0, 0.55, 0.04), (1.0, 0.55, 0.04),
    (0.1, 0.9, 0.25), (1.0, 0.12, 0.08), (0.1, 0.9, 0.25),
)
# Labels remain in fixed positions so temporal antialiasing cannot blend one
# instruction into the next. Only the colored highlight and hold bar move.
_ROWS = ("Pick up with right hand", "Pass to left hand", "Keep left hand steady",
         "Handover successful", "Reset to try again", "Previous handover complete")
_ROW_TOPS = tuple(0.095 - index * 0.038 for index in range(len(_ROWS)))


def handover_display_state(state, previous_success=None):
    """Read live success and hold progress without advancing any task timers."""
    status = state.picked_up.long()
    status = torch.where(state.hold_count > 0, 2, status)
    status = torch.where(state.success, 3, status)
    status = torch.where(state.failed, 4, status)
    progress = (state.hold_count.float() / state.hold_steps).clamp(0, 1)
    progress = torch.where(state.failed, 0.0, progress)
    if previous_success is not None:
        show_previous = (status == 0) & previous_success
        status = torch.where(show_previous, 5, status)
        progress = torch.where(show_previous, 1.0, progress)
    return status, progress


def _material(stage, path, color):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage, path, rectangles, material, depth=0.0):
    """Make quads facing the operator (-X), with screen right along -Y."""
    points = []
    for left, bottom, width, height in rectangles:
        points.extend((
            (depth, -left, bottom), (depth, -left - width, bottom),
            (depth, -left - width, bottom + height), (depth, -left, bottom + height),
        ))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(rectangles))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    # Match Isaac Lab visualization markers: cues must not illuminate nearby
    # surfaces or appear in secondary rays.
    mesh.GetPrim().CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _text_rectangles(text, height, top):
    """Build proportional English lettering, merging adjacent pixels per row."""
    pixel_size = height / GLYPH_SIZE
    left = -sum(GLYPH_ADVANCES[character] for character in text) * pixel_size / 2
    rectangles = []
    cursor = 0
    for character in text:
        glyph = GLYPHS[character]
        for row in range(GLYPH_SIZE):
            bits = int.from_bytes(glyph[row * 4:(row + 1) * 4], "big")
            col = 0
            while col < GLYPH_SIZE:
                if not bits & (1 << (GLYPH_SIZE - 1 - col)):
                    col += 1
                    continue
                start = col
                while col < GLYPH_SIZE and bits & (1 << (GLYPH_SIZE - 1 - col)):
                    col += 1
                rectangles.append((
                    left + (cursor + start) * pixel_size, top - (row + 1) * pixel_size,
                    (col - start) * pixel_size, pixel_size,
                ))
        cursor += GLYPH_ADVANCES[character]
    return rectangles


class HandoverVisuals:
    """One instanced panel per environment; update USD only when status changes.

    Default purpose is deliberate: CloudXR can hide guide-purpose geometry.
    These operator cues therefore also appear in RGB when enabled. They have
    no collision/rigid-body APIs and are disabled by --no-enable_debug_vis.
    """

    def __init__(self, env):
        stage = env.sim.stage
        root = "/Visuals/HandoverStatus"
        self.panel = UsdGeom.PointInstancer.Define(stage, root + "/Panels")
        white = _material(stage, root + "/Materials/Text", (0.9, 0.9, 0.9))
        dark = _material(stage, root + "/Materials/Background", (0.012, 0.018, 0.025))
        track = _material(stage, root + "/Materials/Track", (0.08, 0.1, 0.12))
        colors = [_material(stage, f"{root}/Materials/State{i}", color)
                  for i, color in enumerate(_COLORS)]
        path = f"{root}/Panels/Panel"
        UsdGeom.Xform.Define(stage, path)
        _mesh(stage, path + "/Background", [(-0.29, -0.16, 0.58, 0.33)], dark, 0.002)
        _mesh(stage, path + "/Title", _text_rectangles("Bimanual handover", 0.038, 0.15), white)
        for row, (label, top) in enumerate(zip(_ROWS, _ROW_TOPS)):
            _mesh(stage, f"{path}/Row{row}", _text_rectangles(label, 0.028, top), white)
        _mesh(stage, path + "/Track", [(-0.235, -0.148, 0.47, 0.012)], track)
        self.panel.CreatePrototypesRel().SetTargets([path])
        self.panel.CreateProtoIndicesAttr().Set([0] * env.num_envs)

        # Keep the highlight separate so all lettering remains static as the
        # active row changes, avoiding overlapping text under temporal AA.
        self.highlight = UsdGeom.PointInstancer.Define(stage, root + "/Highlight")
        for i in range(len(_ROWS)):
            _mesh(stage, f"{root}/Highlight/State{i}", [
                (-0.27, -0.028, 0.008, 0.032),
                (-0.27, -0.028, 0.54, 0.003),
                (-0.27, 0.001, 0.54, 0.003),
                (0.262, -0.028, 0.008, 0.032),
            ], colors[i])
        self.highlight.CreatePrototypesRel().SetTargets([f"{root}/Highlight/State{i}" for i in range(len(_ROWS))])

        self.bar = UsdGeom.PointInstancer.Define(stage, root + "/Progress")
        for i in range(len(_ROWS)):
            _mesh(stage, f"{root}/Progress/State{i}", [(0, -0.148, 0.47, 0.012)], colors[i])
        self.bar.CreatePrototypesRel().SetTargets([f"{root}/Progress/State{i}" for i in range(len(_ROWS))])

        table = env.cfg.scene.table
        table_top = table.init_state.pos[2] + table.spawn.size[2] / 2
        positions = env.scene.env_origins.detach().cpu().numpy().copy()
        positions += np.array([table.init_state.pos[0] + table.spawn.size[0] * 0.36,
                               table.init_state.pos[1], table_top + 0.45])
        self.panel.CreatePositionsAttr().Set(positions.tolist())
        self._positions = positions.copy()
        self._positions[:, 0] -= 0.002
        # The bar grows from the left edge instead of expanding about its center.
        positions += np.array([-0.002, 0.235, 0.0])
        self.bar.CreatePositionsAttr().Set(positions.tolist())
        self._previous = None
        self._last_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.update(None, env.num_envs)

    def on_reset(self, state, env_ids):
        # Auto-reset can happen before the success frame is rendered. Keep an
        # explicitly labelled last-result message until the next pickup, while
        # leaving all task history and success flags free to reset normally.
        ids = slice(None) if env_ids is None else env_ids
        self._last_success[ids] = state.success[ids]

    def update(self, state, num_envs):
        if state is None:
            values = np.zeros((2, num_envs), dtype=np.float32)
        else:
            self._last_success &= ~(state.picked_up | state.failed)
            status, progress = handover_display_state(state, self._last_success)
            values = torch.stack((status, progress)).detach().cpu().numpy()
        if self._previous is not None and np.array_equal(values, self._previous):
            return
        indices = values[0].astype(np.int32).tolist()
        self.highlight.CreateProtoIndicesAttr().Set(indices)
        positions = self._positions.copy()
        positions[:, 2] += np.asarray(_ROW_TOPS)[indices]
        self.highlight.CreatePositionsAttr().Set(positions.tolist())
        self.bar.CreateProtoIndicesAttr().Set(indices)
        scales = np.ones((num_envs, 3), dtype=np.float32)
        scales[:, 1] = values[1]
        self.bar.CreateScalesAttr().Set(scales.tolist())
        self.bar.CreateInvisibleIdsAttr().Set(np.flatnonzero(values[1] == 0).tolist())
        self._previous = values.copy()
