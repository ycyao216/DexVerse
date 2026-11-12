"""Non-colliding, default-purpose coffee cues visible in CloudXR and camera RGB."""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from ...bimanual.handover_visualization import _material, _mesh, _text_rectangles
from ..check_utils import local_point
from .checks import (
    MUG_MOUTH_RADIUS, POUR_SPOUT_XY_TOL, POUR_SPOUT_MIN_ABOVE_RIM,
    POUR_SPOUT_MAX_ABOVE_RIM,
)

TITLE = "Make coffee"
STEPS = ("Pour in milk", "Place mug and release", "Flip the switch")
CHECKS = ("Spout aligned with mug", "Spout 1-12 cm above rim", "Bottle tilt at least 70 deg", "Mug tilt no more than 20 deg")
RESULTS = ("Task conditions met", "Milk spilled - reset")
STEP_TOPS = (0.177, 0.136, 0.095)
CHECK_TOPS = (-0.003, -0.034, -0.065, -0.096)
RESULT_TOPS = (-0.216, -0.249)
COLORS = ((0.24, 0.3, 0.36), (1.0, 0.56, 0.08), (0.12, 0.9, 0.35), (1.0, 0.15, 0.08))
ROOT = "/Visuals/CoffeeFeedback"


def _bind(prim, material):
    prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _instancer(stage, path, prototypes):
    result = UsdGeom.PointInstancer.Define(stage, path)
    result.CreatePrototypesRel().SetTargets(prototypes)
    return result


def _visibility_scales(marker, visible, scales=None):
    # RTX/Fabric can retain stale instances when invisibleIds or singular (zero)
    # scales change at runtime. A tiny, nonzero scale reliably hides each cue
    # and supports independent visibility in vectorized environments.
    scales = np.ones((len(visible), 3)) if scales is None else scales.copy()
    scales[~visible] = 1e-6
    marker.CreateScalesAttr().Set(scales.tolist())


def _rings(stage, path, material):
    """Three thin horizontal rings outline the exact acceptance column."""
    points, counts = [], []
    for z in (0.0, POUR_SPOUT_MIN_ABOVE_RIM, POUR_SPOUT_MAX_ABOVE_RIM):
        for i in range(64):
            a, b = 2 * math.pi * i / 64, 2 * math.pi * (i + 1) / 64
            points.extend((r * math.cos(t), r * math.sin(t), z)
                          for r, t in ((POUR_SPOUT_XY_TOL - 0.001, a),
                                       (POUR_SPOUT_XY_TOL + 0.001, a),
                                       (POUR_SPOUT_XY_TOL + 0.001, b),
                                       (POUR_SPOUT_XY_TOL - 0.001, b)))
            counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path + "/Rings")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), material)
    for i in range(4):
        a = math.pi * (0.25 + i / 2)
        pole = UsdGeom.Cylinder.Define(stage, f"{path}/Edge{i}")
        pole.CreateRadiusAttr(0.0007)
        pole.CreateHeightAttr(POUR_SPOUT_MAX_ABOVE_RIM - POUR_SPOUT_MIN_ABOVE_RIM)
        UsdGeom.Xformable(pole).AddTranslateOp().Set(Gf.Vec3d(
            POUR_SPOUT_XY_TOL * math.cos(a), POUR_SPOUT_XY_TOL * math.sin(a),
            (POUR_SPOUT_MAX_ABOVE_RIM + POUR_SPOUT_MIN_ABOVE_RIM) / 2))
        _bind(pole.GetPrim(), material)


class CoffeeVisuals:
    """Static lettering and colored indicators; render reads never advance timers."""

    def __init__(self, env):
        stage, n = env.sim.stage, env.num_envs
        mats = [_material(stage, f"{ROOT}/Materials/State{i}", color) for i, color in enumerate(COLORS)]
        white = _material(stage, ROOT + "/Materials/Text", (0.88, 0.93, 0.97))
        dark = _material(stage, ROOT + "/Materials/Background", (0.012, 0.018, 0.028))
        track = _material(stage, ROOT + "/Materials/Track", (0.05, 0.07, 0.09))
        cream = _material(stage, ROOT + "/Materials/Milk", (0.97, 0.96, 0.83))
        guide = _material(stage, ROOT + "/Materials/Guide", (0.25, 0.67, 0.9))
        panel_path = ROOT + "/Panels/Panel"
        UsdGeom.Xform.Define(stage, panel_path)
        _mesh(stage, panel_path + "/Background", [(-0.32, -0.285, 0.64, 0.545)], dark, 0.003)
        labels = [("Title", TITLE, 0.034, 0.235), ("ChecksTitle", "Pour checks", 0.023, 0.038),
                  ("HoldLabel", "Pour steadily for 1 s", 0.02, -0.136)]
        labels += [(f"Step{i}", label, 0.026, top) for i, (label, top) in enumerate(zip(STEPS, STEP_TOPS))]
        labels += [(f"Check{i}", label, 0.021, top) for i, (label, top) in enumerate(zip(CHECKS, CHECK_TOPS))]
        labels += [(f"Result{i}", label, 0.021, top) for i, (label, top) in enumerate(zip(RESULTS, RESULT_TOPS))]
        for name, label, height, top in labels:
            _mesh(stage, panel_path + "/" + name, _text_rectangles(label, height, top), white)
        _mesh(stage, panel_path + "/Track", [(-0.27, -0.184, 0.54, 0.012)], track)
        self.panel = _instancer(stage, ROOT + "/Panels", [panel_path])
        self.panel.CreateProtoIndicesAttr().Set([0] * n)
        self.positions = env.scene.env_origins.detach().cpu().numpy().copy()
        # Above the holder, left of the machine, facing the operator at -X.
        table = env.cfg.scene.table
        top = table.init_state.pos[2] + table.spawn.size[2] / 2
        self.positions += np.array([0.58, 0.43, top + 0.82])
        self.panel.CreatePositionsAttr().Set(self.positions.tolist())

        paths = []
        for i, mat in enumerate(mats):
            path = f"{ROOT}/Indicators/Color{i}"
            UsdGeom.Xform.Define(stage, path)
            _mesh(stage, path + "/Square", [(-0.009, -0.009, 0.018, 0.018)], mat)
            # A check mark differentiates completion from color alone.
            if i == 2:
                _mesh(stage, path + "/Tick", [(-0.006 + j * 0.002, -0.001 - j * 0.002, 0.0025, 0.0025)
                      for j in range(3)] + [(-0.002 + j * 0.002, -0.005 + j * 0.002, 0.0025, 0.0025)
                      for j in range(5)], dark, -0.0005)
            paths.append(path)
        self.indicators = _instancer(stage, ROOT + "/Indicators", paths)
        offsets = np.array([[-0.003, 0.283, top - 0.013] for top in (*STEP_TOPS, *CHECK_TOPS, *RESULT_TOPS)])
        self.indicators.CreatePositionsAttr().Set((self.positions[:, None, :] + offsets).reshape(-1, 3).tolist())
        # Fixed segments avoid temporal-AA trails when a filled mesh shrinks.
        bar_paths = [ROOT + "/Progress/Empty", ROOT + "/Progress/Filled"]
        for path, material in zip(bar_paths, (track, mats[2])):
            _mesh(stage, path, [(0, -0.184, 0.54 / 30, 0.012)], material)
        self.bar = _instancer(stage, ROOT + "/Progress", bar_paths)
        offsets = np.array([[-0.003, 0.27 - i * 0.54 / 30, 0] for i in range(30)])
        self.bar.CreatePositionsAttr().Set((self.positions[:, None, :] + offsets).reshape(-1, 3).tolist())

        target_paths, spout_paths = [], []
        for i, mat in enumerate((mats[1], mats[2])):
            path = f"{ROOT}/Target/State{i}"
            UsdGeom.Xform.Define(stage, path)
            _rings(stage, path, mat)
            target_paths.append(path)
            path = f"{ROOT}/Spout/State{i}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(0.005)
            _bind(sphere.GetPrim(), mat)
            spout_paths.append(path)
        self.target = _instancer(stage, ROOT + "/Target", target_paths)
        self.spout = _instancer(stage, ROOT + "/Spout", spout_paths)
        self.lines = []
        for name, radius, mat in (("Projection", 0.0008, guide), ("Stream", 0.003, cream)):
            path = f"{ROOT}/{name}/Shape"
            cylinder = UsdGeom.Cylinder.Define(stage, path)
            cylinder.CreateRadiusAttr(radius)
            cylinder.CreateHeightAttr(1.0)
            _bind(cylinder.GetPrim(), mat)
            marker = _instancer(stage, f"{ROOT}/{name}", [path])
            marker.CreateProtoIndicesAttr().Set([0] * n)
            self.lines.append(marker)
        path = ROOT + "/MilkSurface/Shape"
        surface = UsdGeom.Cylinder.Define(stage, path)
        surface.CreateRadiusAttr(MUG_MOUTH_RADIUS * 0.82)
        surface.CreateHeightAttr(0.001)
        _bind(surface.GetPrim(), cream)
        self.surface = _instancer(stage, ROOT + "/MilkSurface", [path])
        self.surface.CreateProtoIndicesAttr().Set([0] * n)
        self._previous_panel = None
        self.last_success = None

    def update(self, env, state):
        self.last_success = state["stages"][:, 2] == 2
        # Quantize only the visual fill, never the success counter.
        progress = np.round(state["progress"].detach().cpu().numpy() * 30) / 30
        poured = state["poured"].detach().cpu().numpy()
        spilled = state["spilled"].detach().cpu().numpy()
        conditions = state["conditions"].detach().cpu().numpy()
        checks = np.where(conditions, 2, 1)
        checks[poured] = 2  # The accepted pour's requirements stay checked.
        checks[spilled] = 3
        stages = state["stages"].detach().cpu().numpy()
        status = state["status"].detach().cpu().numpy()
        panel = np.c_[stages, checks, progress, status]
        if self._previous_panel is None or not np.array_equal(panel, self._previous_panel):
            # Result labels also stay fixed, as in handover: swapping text at
            # one location leaves ghost letters under headset temporal AA.
            results = np.c_[np.where(status == 3, 2, 0), np.where(status == 4, 3, 0)]
            self.indicators.CreateProtoIndicesAttr().Set(np.c_[stages, checks, results].reshape(-1).tolist())
            filled = np.arange(30)[None, :] < np.rint(progress[:, None] * 30)
            self.bar.CreateProtoIndicesAttr().Set(filled.astype(int).reshape(-1).tolist())
            self._previous_panel = panel.copy()
        spout = state["spout"].detach().cpu().numpy()
        rim = state["rim"].detach().cpu().numpy()
        pouring = state["pouring"].detach().cpu().numpy()
        flowing = state["flowing"].detach().cpu().numpy()
        for marker, positions in ((self.target, rim), (self.spout, spout)):
            marker.CreateProtoIndicesAttr().Set(pouring.astype(int).tolist())
            marker.CreatePositionsAttr().Set(positions.tolist())
            _visibility_scales(marker, ~poured & ~spilled)
        # Gravity-aligned projection lands at the bottle's XY, not at the mug's
        # center: the cue must never depict a missed pour as entering the mug.
        height = spout[:, 2] - rim[:, 2]
        positions = spout.copy()
        positions[:, 2] -= height / 2
        scales = np.ones((env.num_envs, 3))
        scales[:, 2] = np.maximum(height, 0.0001)
        for marker, visible in zip(self.lines, ((height > 0) & ~poured & ~spilled & ~flowing, flowing)):
            marker.CreatePositionsAttr().Set(positions.tolist())
            _visibility_scales(marker, visible, scales)
        mug_quats = env.scene["object"].data.root_quat_w.detach().cpu().numpy()
        surface = local_point(env.scene["object"].data.root_pos_w, env.scene["object"].data.root_quat_w,
                              (0, 0, 0.071)).detach().cpu().numpy()
        self.surface.CreatePositionsAttr().Set(surface.tolist())
        self.surface.CreateOrientationsAttr().Set([Gf.Quath(float(q[0]), Gf.Vec3h(*q[1:].tolist())) for q in mug_quats])
        _visibility_scales(self.surface, poured & ~spilled)
