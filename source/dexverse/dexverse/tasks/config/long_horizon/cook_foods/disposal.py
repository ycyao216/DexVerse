"""Live disposal checks for the two used cooking containers (no simulator imports)."""
from __future__ import annotations

import torch

from ..check_utils import announce_once, rotate, rotate_inverse, track

# Bin origin is the underside of its base; interior matches the physical walls.
BIN_HALF_XY = (0.088, 0.19)
BIN_WALL = 0.012
BIN_HEIGHT = 0.292
# Counter notch spans world x=.10..30, y=.64..1.05. Fit the narrow bin
# ahead-left of the operator, with its open rim level with the .80 m worktop.
BIN_POSITION = (0.20, 0.8475, 0.80 - BIN_HEIGHT)
DISPOSAL_HOLD_S = 0.25
# Bounds are used only for conservative hand clearance, never for containment:
# a rotated box has imaginary corners outside the round bottle/can surface.
CONTAINER_BOUNDS = {
    "object": ((-0.029, 0.0, -0.029), (0.029, 0.201, 0.029)),
    "can": ((-0.035, -0.043, -0.033), (0.034, 0.060, 0.036)),
}


def mesh_hull_points(stage, root):
    """Compact, asset-derived surface in the rigid body's frame (metres).

    Quantization bounds vertex error below 0.87 mm. A convex hull preserves all
    extrema against the bin's planar walls, without testing empty box corners.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    from scipy.spatial import ConvexHull

    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    points = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            xyz = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            transform = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
            points.append(xyz @ transform[:3, :3] + transform[3, :3])
    if not points:
        raise ValueError(f"No container mesh under {root.GetPath()}")
    points = np.unique(np.round(np.concatenate(points) / .001) * .001, axis=0)
    return points[ConvexHull(points).vertices]


def container_points(env, name):
    cache = env.__dict__.setdefault("_cook_container_points", {})
    if name not in cache:
        asset = env.scene[name]
        root = env.scene.stage.GetPrimAtPath(asset.root_physx_view.prim_paths[0])
        cache[name] = torch.as_tensor(mesh_hull_points(env.scene.stage, root),
                                      device=env.device, dtype=asset.data.root_pos_w.dtype)
    return cache[name]


def disposal_measurements(env, fingertip_body_names):
    bin_data = env.scene["trash_bin"].data
    robot = env.scene["robot"]
    # Body indices are fixed for an articulation's lifetime. Resolve the regex
    # names once, while still reading current poses and velocities on every call.
    names = tuple(fingertip_body_names)
    cache = env.__dict__.setdefault("_cook_disposal_body_ids", {})
    entry = cache.get(names)
    if entry is None or entry[0] is not robot:
        entry = (robot, robot.find_bodies(list(names), preserve_order=True)[0])
        cache[names] = entry
    ids = entry[1]
    tips = robot.data.body_pos_w[:, ids]
    inside, released, settled, margins, clearances, speeds, spins = [], [], [], [], [], [], []
    for name, (low, high) in CONTAINER_BOUNDS.items():
        data = env.scene[name].data
        n = data.root_pos_w.shape[0]
        points = container_points(env, name)
        count = len(points)
        q = data.root_quat_w[:, None, :].expand(-1, count, -1).reshape(-1, 4)
        world = data.root_pos_w[:, None, :] + rotate(q, points.repeat(n, 1)).reshape(n, count, 3)
        bin_q = bin_data.root_quat_w[:, None, :].expand(-1, count, -1).reshape(-1, 4)
        relative = rotate_inverse(bin_q, (world - bin_data.root_pos_w[:, None, :]).reshape(-1, 3)).reshape(n, count, 3)
        margin = torch.stack((BIN_HALF_XY[0] - relative[..., 0].abs().amax(dim=-1),
                              BIN_HALF_XY[1] - relative[..., 1].abs().amax(dim=-1),
                              relative[..., 2].amin(dim=-1) - BIN_WALL,
                              BIN_HEIGHT - relative[..., 2].amax(dim=-1)), dim=-1)
        # Millimetre allowances cover hull quantization and contact penetration;
        # keep the full container below the rim rather than checking its root.
        tolerance = torch.tensor([.002, .002, .004, .001], device=env.device)
        fits = (margin >= -tolerance).all(dim=-1)
        # Require both hands clear of the object's box by 3 cm. This rejects a
        # container held motionless in the bin while allowing either hand to toss it.
        tip_q = data.root_quat_w[:, None, :].expand(-1, tips.shape[1], -1).reshape(-1, 4)
        local_tips = rotate_inverse(tip_q, (tips - data.root_pos_w[:, None, :]).reshape(-1, 3)).reshape_as(tips)
        lo, hi = (torch.tensor(v, device=tips.device, dtype=tips.dtype) for v in (low, high))
        distances = torch.maximum(lo - local_tips, local_tips - hi).clamp_min(0).norm(dim=-1)
        inside.append(fits)
        released.append((distances >= 0.03).all(dim=-1))
        speed, spin = data.root_lin_vel_w.norm(dim=-1), data.root_ang_vel_w.norm(dim=-1)
        settled.append((speed < 0.15) & (spin < 1.0))
        margins.append(margin)
        clearances.append(distances.amin(dim=-1))
        speeds.append(speed)
        spins.append(spin)
    result = {key: torch.stack(values, dim=-1) for key, values in
              (("inside", inside), ("released", released), ("settled", settled))}
    result["ready"] = result["inside"] & result["released"] & result["settled"]
    result.update(margins=torch.stack(margins, dim=1), clearance=torch.stack(clearances, dim=1),
                  speed=torch.stack(speeds, dim=1), spin=torch.stack(spins, dim=1))
    return result


def containers_disposed(env, fingertip_body_names):
    ready = disposal_measurements(env, fingertip_body_names)["ready"].all(dim=-1)
    held, _ = track(env, "disposal", ready, DISPOSAL_HOLD_S)
    announce_once(env, "Cook Foods", "disposal", held, "Both used containers are in the trash bin.")
    return ready & held
