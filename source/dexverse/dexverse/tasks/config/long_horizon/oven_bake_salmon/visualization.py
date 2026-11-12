"""Floating, non-colliding baking checklist visible in XR and camera RGB."""
from __future__ import annotations

import numpy as np
from pxr import UsdGeom

from ...bimanual.handover_visualization import _material, _mesh, _text_rectangles

ROOT = "/Visuals/SalmonProgress"
LABELS = ("Add salt", "Add black pepper", "Place tray in oven", "Close oven door", "Turn oven knob", "Task conditions met")
TOPS = tuple(.15 - .048 * i for i in range(6))


class SalmonProgress:
    def __init__(self, env):
        stage, n = env.sim.stage, env.num_envs
        white = _material(stage, ROOT + "/Materials/Text", (.9, .94, .97))
        dark = _material(stage, ROOT + "/Materials/Background", (.012, .018, .028))
        colors = [_material(stage, f"{ROOT}/Materials/Status{i}", color) for i, color in enumerate(
            ((.22, .28, .34), (1., .56, .08), (.12, .9, .35)))]
        path = ROOT + "/Panels/Panel"
        UsdGeom.Xform.Define(stage, path)
        _mesh(stage, path + "/Background", [(-.28, -.155, .56, .40)], dark, .003)
        _mesh(stage, path + "/Title", _text_rectangles("Bake salmon", .035, .217), white)
        for i, (label, top) in enumerate(zip(LABELS, TOPS)):
            _mesh(stage, path + f"/Step{i}", _text_rectangles(label, .026, top), white)
        self.panel = UsdGeom.PointInstancer.Define(stage, ROOT + "/Panels")
        self.panel.CreatePrototypesRel().SetTargets([path])
        self.panel.CreateProtoIndicesAttr().Set([0] * n)
        table = env.cfg.scene.table
        surface = table.init_state.pos[2] + table.spawn.size[2] / 2
        positions = env.scene.env_origins.detach().cpu().numpy() + np.array([.46, .53, surface + .88])
        self.panel.CreatePositionsAttr().Set(positions.tolist())
        paths = []
        for i, mat in enumerate(colors):
            path = f"{ROOT}/Indicators/Status{i}"
            UsdGeom.Xform.Define(stage, path)
            _mesh(stage, path + "/Square", [(-.011, -.011, .022, .022)], mat)
            if i == 2:
                _mesh(stage, path + "/Tick",
                      [(-.007 + j * .002, -.001 - j * .002, .003, .003) for j in range(3)] +
                      [(-.003 + j * .002, -.005 + j * .002, .003, .003) for j in range(6)], dark, -.0005)
            paths.append(path)
        self.indicators = UsdGeom.PointInstancer.Define(stage, ROOT + "/Indicators")
        self.indicators.CreatePrototypesRel().SetTargets(paths)
        offsets = np.array([[-.003, .245, top - .013] for top in TOPS])
        self.indicators.CreatePositionsAttr().Set((positions[:, None, :] + offsets).reshape(-1, 3).tolist())
        self._previous = None
        self.last_success = None

    def update(self, stages, success):
        self.last_success = success.detach().clone()
        data = np.c_[stages.detach().cpu().numpy(), 2 * success.detach().cpu().numpy()]
        if self._previous is None or not np.array_equal(data, self._previous):
            self.indicators.CreateProtoIndicesAttr().Set(data.reshape(-1).tolist())
            self._previous = data.copy()
