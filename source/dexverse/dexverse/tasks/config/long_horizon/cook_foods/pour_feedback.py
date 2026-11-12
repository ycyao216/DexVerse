"""Pour aiming cues for water and soup; no floating panel or task-state mutation."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..check_utils import clear_stage_latches, ever_held
from .checks import pour_measurements


def pour_display_state(env, sources):
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    spilled = ever_held(env, "spill", zero)
    samples = []
    for params in sources.values():
        sample = pour_measurements(env, **{k: v for k, v in params.items() if k != "label"})
        done = ever_held(env, "pour_" + params["pour_name"], zero)
        samples.append({**sample, "done": done, "visible": ~done & ~spilled})
    return samples


def update_pour_visuals(env, sources):
    if not env.cfg.enable_debug_vis:
        return
    visuals = getattr(env, "_cook_pour_visuals", None)
    if visuals is None:
        visuals = env._cook_pour_visuals = CookingPourVisuals(env)
    visuals.update(env, pour_display_state(env, sources))


def reset_cooking_feedback(env, env_ids, task_key, sources, liquid_prims, burner_prims):
    from ..handling import show_by_state

    clear_stage_latches(env, env_ids, task_key)
    # Reset may pause teleop immediately. Clear the authored USD visibility now,
    # rather than waiting for the next success evaluation/physics step.
    for key, prims in (("liquid", liquid_prims), ("burner", burner_prims)):
        state = getattr(env, "_long_horizon_shown", {}).get(
            key, torch.zeros(env.num_envs, device=env.device, dtype=torch.long)).clone()
        state[slice(None) if env_ids is None else env_ids] = 0
        show_by_state(env, key, prims, state)
    update_pour_visuals(env, sources)


class CookingPourVisuals:
    """Default-purpose, non-colliding geometry visible in XR and camera RGB."""
    ROOT = "/Visuals/CookingPour"

    def __init__(self, env):
        from pxr import Sdf, UsdGeom, UsdShade
        from ...bimanual.handover_visualization import _material

        self._geom = UsdGeom
        stage = env.sim.stage
        root = self.ROOT
        colors = ((1.0, .56, .08), (.12, .9, .35), (.15, .55, 1.0), (.9, .18, .05))
        materials = [_material(stage, f"{root}/Materials/Color{i}", c) for i, c in enumerate(colors)]

        def bind(prim, material):
            prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        def instancer(name, paths):
            marker = UsdGeom.PointInstancer.Define(stage, root + "/" + name)
            marker.CreatePrototypesRel().SetTargets(paths)
            return marker

        target_paths, spout_paths = [], []
        for i in range(2):
            path = f"{root}/Target/Color{i}"
            points = []
            radius = env.cfg.pour_goal_xy_threshold
            for j in range(96):
                a, b = 2 * math.pi * j / 96, 2 * math.pi * (j + 1) / 96
                points.extend((r * math.cos(t), r * math.sin(t), 0)
                              for r, t in ((radius - .002, a), (radius + .002, a),
                                           (radius + .002, b), (radius - .002, b)))
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([4] * 96)
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            bind(mesh.GetPrim(), materials[i])
            target_paths.append(path)
            path = f"{root}/Spouts/Color{i}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(.007)
            bind(sphere.GetPrim(), materials[i])
            spout_paths.append(path)
        self.target = instancer("Target", target_paths)
        self.spouts = instancer("Spouts", spout_paths)
        self.streams, self.guides = [], []
        for i, liquid in enumerate(("Water", "Soup")):
            for name, radius, collection in (("Stream", .004, self.streams), ("Aim", .0008, self.guides)):
                path = f"{root}/{liquid}{name}/Shape"
                cylinder = UsdGeom.Cylinder.Define(stage, path)
                cylinder.CreateRadiusAttr(radius)
                cylinder.CreateHeightAttr(1.0)
                bind(cylinder.GetPrim(), materials[i + 2])
                collection.append(instancer(liquid + name, [path]))

    @staticmethod
    def draw(marker, positions, indices, visible, scales=None):
        scales = np.ones_like(positions) if scales is None else scales.copy()
        # Tiny nonzero scales avoid stale instances in RTX/Fabric invisibleIds.
        scales[~visible] = 1e-6
        marker.CreatePositionsAttr().Set(positions.tolist())
        marker.CreateProtoIndicesAttr().Set(indices.tolist())
        marker.CreateScalesAttr().Set(scales.tolist())

    def update(self, env, samples):
        def array(value):
            return value.detach().cpu().numpy()
        rim = array(env.scene["pot"].data.root_pos_w).copy()
        # Pot shell rim is at local z=.08482 before scaling. The horizontal
        # target follows the gate's root-centered XY cylinder, even when tipped.
        rim[:, 2] += .085 * env.cfg.pot_scale[2] + .002
        visible = np.stack([array(s["visible"]) for s in samples], axis=1)
        valid = np.stack([array(s["pouring"]) for s in samples], axis=1)
        spouts = np.stack([array(s["spout"]) for s in samples], axis=1)
        self.draw(self.target, rim, (valid & visible).any(axis=1).astype(int), visible.any(axis=1))
        self.draw(self.spouts, spouts.reshape(-1, 3), valid.astype(int).reshape(-1), visible.reshape(-1))
        for i in range(2):
            height = spouts[:, i, 2] - rim[:, 2]
            pos = spouts[:, i].copy()
            pos[:, 2] -= height / 2
            scales = np.ones_like(pos)
            scales[:, 2] = np.maximum(height, .0001)
            # Project straight down at the spout's XY, never toward the pot's
            # center. No upward "stream" if the spout is below the rim.
            shown = visible[:, i] & (height > 0)
            for marker, active in ((self.streams[i], shown & valid[:, i]),
                                   (self.guides[i], shown & ~valid[:, i])):
                self.draw(marker, pos, np.zeros(env.num_envs, dtype=int), active, scales)
