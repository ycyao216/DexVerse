# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for randomly grasping two selected primitive objects."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg as EventTerm
from isaaclab.managers.manager_term_cfg import RewardTermCfg as RewTerm
from isaaclab.managers.manager_term_cfg import TerminationTermCfg as DoneTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop

TABLE_CLEARANCE = 0.004
LIFT_SUCCESS_HEIGHT_M = 0.20
LIFT_SUCCESS_Z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT + LIFT_SUCCESS_HEIGHT_M
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5

OBJECT_MASS = 0.06
OBJECT_OBJECT_SEPARATION_MARGIN = 0.012
ACTIVE_OBJECT_COUNT = 2

OBJECT_SPAWN_X_RANGE = (-0.06, 0.06)
OBJECT_SPAWN_Y_RANGE = (-0.06, 0.06)
INACTIVE_OBJECT_PARK_Z = -10.0
INACTIVE_OBJECT_PARK_X_SPACING = 10.0

SPHERE_RADIUS = 0.028
CYLINDER_RADIUS = 0.022
CYLINDER_HEIGHT = 0.060
TRIANGULAR_PRISM_SIDE_LENGTH = 0.060
TRIANGULAR_PRISM_HEIGHT = 0.055
CUBE_SIZE = 0.045
CUBOID_SIZE = (0.065, 0.035, 0.045)


@sim_utils.clone
def spawn_triangular_prism(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn an upright equilateral triangular prism as a convex mesh primitive."""
    del kwargs
    if cfg.deformable_props is not None:
        raise ValueError("TriangularPrismCfg only supports rigid primitive objects.")

    from pxr import UsdPhysics

    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"
    circumradius = cfg.side_length / math.sqrt(3.0)
    half_height = cfg.height * 0.5
    y_scale = math.sqrt(3.0) * 0.5 * circumradius
    points = [
        (circumradius, 0.0, -half_height),
        (-0.5 * circumradius, y_scale, -half_height),
        (-0.5 * circumradius, -y_scale, -half_height),
        (circumradius, 0.0, half_height),
        (-0.5 * circumradius, y_scale, half_height),
        (-0.5 * circumradius, -y_scale, half_height),
    ]
    faces = [
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4),
        (0, 4, 3),
        (1, 2, 5),
        (1, 5, 4),
        (2, 0, 3),
        (2, 3, 5),
    ]
    mesh_prim = prim_utils.create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": points,
            "faceVertexIndices": [index for face in faces for index in face],
            "faceVertexCounts": [3] * len(faces),
            "subdivisionScheme": "bilinear",
        },
    )

    if cfg.collision_props is not None:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        sim_utils.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

    if cfg.visual_material is not None:
        if cfg.visual_material_path.startswith("/"):
            material_path = cfg.visual_material_path
        else:
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        sim_utils.bind_visual_material(mesh_prim_path, material_path, stage=stage)

    if cfg.physics_material is not None:
        if cfg.physics_material_path.startswith("/"):
            material_path = cfg.physics_material_path
        else:
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        cfg.physics_material.func(material_path, cfg.physics_material)
        sim_utils.bind_physics_material(mesh_prim_path, material_path, stage=stage)

    if cfg.mass_props is not None:
        sim_utils.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)


@configclass
class TriangularPrismCfg(sim_utils.MeshCfg):
    """Configuration for a rigid equilateral triangular prism mesh primitive."""

    func: Callable = spawn_triangular_prism

    side_length: float = TRIANGULAR_PRISM_SIDE_LENGTH
    height: float = TRIANGULAR_PRISM_HEIGHT


@dataclass(frozen=True)
class PrimitiveObjectSpec:
    spawn: sim_utils.SpawnerCfg
    radius: float = 0.0
    half_height: float = 0.0


def _rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=0,
        disable_gravity=False,
    )


def _spawn_kwargs(color: tuple[float, float, float]) -> dict:
    return {
        "rigid_props": _rigid_props(),
        "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        "mass_props": sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.55,
            metallic=0.0,
        ),
    }


PRIMITIVE_OBJECT_SPECS: dict[str, PrimitiveObjectSpec] = {
    "sphere": PrimitiveObjectSpec(
        sim_utils.SphereCfg(radius=SPHERE_RADIUS, **_spawn_kwargs((0.15, 0.35, 0.95))),
        radius=SPHERE_RADIUS,
        half_height=SPHERE_RADIUS,
    ),
    "cylinder": PrimitiveObjectSpec(
        sim_utils.CylinderCfg(
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
            axis="Z",
            **_spawn_kwargs((0.1, 0.65, 0.35)),
        ),
        radius=CYLINDER_RADIUS,
        half_height=CYLINDER_HEIGHT * 0.5,
    ),
    "triangular_prism": PrimitiveObjectSpec(
        TriangularPrismCfg(
            side_length=TRIANGULAR_PRISM_SIDE_LENGTH,
            height=TRIANGULAR_PRISM_HEIGHT,
            **_spawn_kwargs((0.95, 0.58, 0.12)),
        ),
        radius=TRIANGULAR_PRISM_SIDE_LENGTH / math.sqrt(3.0),
        half_height=TRIANGULAR_PRISM_HEIGHT * 0.5,
    ),
    "cube": PrimitiveObjectSpec(
        sim_utils.CuboidCfg(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), **_spawn_kwargs((0.85, 0.18, 0.18))),
        radius=math.sqrt(2.0) * CUBE_SIZE * 0.5,
        half_height=CUBE_SIZE * 0.5,
    ),
    "cuboid": PrimitiveObjectSpec(
        sim_utils.CuboidCfg(size=CUBOID_SIZE, **_spawn_kwargs((0.42, 0.45, 0.50))),
        radius=math.hypot(CUBOID_SIZE[0], CUBOID_SIZE[1]) * 0.5,
        half_height=CUBOID_SIZE[2] * 0.5,
    ),
}
OBJECT_NAMES: tuple[str, ...] = tuple(PRIMITIVE_OBJECT_SPECS)
OBJECT_RADII: tuple[float, ...] = tuple(spec.radius for spec in PRIMITIVE_OBJECT_SPECS.values())
OBJECT_HALF_HEIGHTS: tuple[float, ...] = tuple(spec.half_height for spec in PRIMITIVE_OBJECT_SPECS.values())


def _make_primitive_object_cfg(name: str, spec: PrimitiveObjectSpec) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spec.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT + spec.half_height + TABLE_CLEARANCE),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


OBJECT_CFGS: dict[str, RigidObjectCfg] = {
    name: _make_primitive_object_cfg(name, spec) for name, spec in PRIMITIVE_OBJECT_SPECS.items()
}


def _resolve_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_active_object_state(env) -> torch.Tensor:
    if not hasattr(env, "_active_grasp_objects"):
        env._active_grasp_objects = torch.zeros(env.num_envs, len(OBJECT_NAMES), device=env.device, dtype=torch.bool)
        env._active_grasp_objects[:, :ACTIVE_OBJECT_COUNT] = True
    return env._active_grasp_objects


def _sample_active_pair(n_envs: int, n_objects: int, device) -> torch.Tensor:
    scores = torch.rand(n_envs, n_objects, device=device)
    selected = scores.topk(k=ACTIVE_OBJECT_COUNT, dim=1).indices
    mask = torch.zeros(n_envs, n_objects, device=device, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask


def _install_active_object_metadata_getter(env) -> None:
    def get_active_object_metadata(env_index: int = 0) -> dict:
        active = _ensure_active_object_state(env)
        env_index_int = int(env_index)
        return {
            "groups": {
                "grasp": {
                    "object_names": list(OBJECT_NAMES),
                    "active_mask": active[env_index_int].detach().cpu().to(torch.bool).tolist(),
                },
            },
        }

    env.get_active_object_metadata = get_active_object_metadata


def reset_active_grasp_object_pair(env, env_ids):
    """Sample exactly two active primitive objects for each reset env."""
    env_ids_t = _resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    active = _ensure_active_object_state(env)
    active[env_ids_t] = _sample_active_pair(env_ids_t.shape[0], len(OBJECT_NAMES), env.device)
    env._active_grasp_object_indices = active.to(dtype=torch.int64).argsort(dim=1, descending=True)[
        :, :ACTIVE_OBJECT_COUNT
    ]
    _install_active_object_metadata_getter(env)


def reset_active_grasp_objects_random(env, env_ids):
    """Place active objects on the table and hide inactive candidates."""
    env_ids_t = _resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return

    active = _ensure_active_object_state(env)[env_ids_t]
    n = env_ids_t.shape[0]
    env_origins = env.scene.env_origins[env_ids_t]
    env_origins_xy = env_origins[:, 0:2]
    table_top_z = env_origins[:, 2] + dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT

    object_radii = torch.tensor(OBJECT_RADII, device=env.device)
    placed_xy = torch.full((n, len(OBJECT_NAMES), 2), float("nan"), device=env.device)

    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        is_active = active[:, i]
        base_root = obj.data.default_root_state[env_ids_t].clone()

        candidate_xy_local = torch.empty((n, 2), device=env.device, dtype=base_root.dtype)
        candidate_xy_local[:, 0].uniform_(OBJECT_SPAWN_X_RANGE[0], OBJECT_SPAWN_X_RANGE[1])
        candidate_xy_local[:, 1].uniform_(OBJECT_SPAWN_Y_RANGE[0], OBJECT_SPAWN_Y_RANGE[1])

        for _ in range(16):
            moved = False
            for j in range(i):
                prior_xy = placed_xy[:, j, :]
                prior_active = ~torch.isnan(prior_xy[:, 0])
                push_mask = is_active & prior_active
                if not bool(push_mask.any()):
                    continue
                min_dist = float(object_radii[i].item() + object_radii[j].item() + OBJECT_OBJECT_SEPARATION_MARGIN)
                direction = candidate_xy_local - torch.where(
                    prior_active.unsqueeze(1), prior_xy, torch.zeros_like(prior_xy)
                )
                dist = direction.norm(dim=1).clamp_min(1e-6)
                too_close = (dist < min_dist) & push_mask
                if bool(too_close.any()):
                    direction = direction / dist.unsqueeze(1)
                    candidate_xy_local = torch.where(
                        too_close.unsqueeze(1),
                        prior_xy + direction * min_dist,
                        candidate_xy_local,
                    )
                    moved = True
            candidate_xy_local[:, 0].clamp_(OBJECT_SPAWN_X_RANGE[0], OBJECT_SPAWN_X_RANGE[1])
            candidate_xy_local[:, 1].clamp_(OBJECT_SPAWN_Y_RANGE[0], OBJECT_SPAWN_Y_RANGE[1])
            if not moved:
                break

        placed_xy[:, i, :] = torch.where(
            is_active.unsqueeze(1),
            candidate_xy_local,
            torch.full_like(candidate_xy_local, float("nan")),
        )

        park_xy_local = torch.zeros_like(candidate_xy_local)
        park_xy_local[:, 0] = i * INACTIVE_OBJECT_PARK_X_SPACING

        root_state = base_root.clone()
        active_xy_world = env_origins_xy + candidate_xy_local
        park_xy_world = env_origins_xy + park_xy_local
        root_state[:, 0:2] = torch.where(is_active.unsqueeze(1), active_xy_world, park_xy_world)
        root_state[:, 2] = torch.where(
            is_active,
            table_top_z + OBJECT_HALF_HEIGHTS[i] + TABLE_CLEARANCE,
            env_origins[:, 2] + INACTIVE_OBJECT_PARK_Z,
        )

        yaw = torch.empty(n, device=env.device, dtype=base_root.dtype).uniform_(-math.pi, math.pi)
        half_yaw = yaw * 0.5
        root_state[:, 3] = torch.where(is_active, torch.cos(half_yaw), torch.ones_like(yaw))
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.where(is_active, torch.sin(half_yaw), torch.zeros_like(yaw))
        root_state[:, 7:13] = 0.0

        obj.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)
        obj.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids_t)

        active_env_ids = env_ids_t[is_active]
        inactive_env_ids = env_ids_t[~is_active]
        if active_env_ids.numel() > 0:
            obj.set_visibility(True, env_ids=active_env_ids)
        if inactive_env_ids.numel() > 0:
            obj.set_visibility(False, env_ids=inactive_env_ids)


def active_objects_lifted(env) -> torch.Tensor:
    """Success when the two active object roots are above table top + 0.20 m."""
    active = _ensure_active_object_state(env)
    success = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    threshold_z = env.scene.env_origins[:, 2] + LIFT_SUCCESS_Z
    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        lifted = obj.data.root_pos_w[:, 2] >= threshold_z
        success &= (~active[:, i]) | lifted
    return success


def active_objects_lifted_reward(env) -> torch.Tensor:
    return active_objects_lifted(env).float()


def active_object_lift_progress(env) -> torch.Tensor:
    active = _ensure_active_object_state(env)
    start_z = env.scene.env_origins[:, 2] + dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
    progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        lifted = torch.clamp((obj.data.root_pos_w[:, 2] - start_z) / LIFT_SUCCESS_HEIGHT_M, 0.0, 1.0)
        progress += torch.where(active[:, i], lifted, torch.zeros_like(lifted))
    return progress / float(ACTIVE_OBJECT_COUNT)


def active_object_ee_distance(env, asset_cfg: SceneEntityCfg, std: float, distance_gain: float = 10.0) -> torch.Tensor:
    _ = std
    active = _ensure_active_object_state(env)
    robot = env.scene[asset_cfg.name]
    fingertip_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        dist_sum = torch.norm(fingertip_pos - obj.data.root_pos_w[:, None, :], dim=-1).sum(dim=-1)
        item_reward = torch.exp(-distance_gain * dist_sum)
        reward += torch.where(active[:, i], item_reward, torch.zeros_like(item_reward))
    return reward / float(ACTIVE_OBJECT_COUNT)


def active_objects_out_of_bound(env) -> torch.Tensor:
    active = _ensure_active_object_state(env)
    table_size = env.cfg.scene.table.spawn.size
    half_x = table_size[0] * 0.5
    half_y = table_size[1] * 0.5
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        item_out = (
            (pos[:, 0] < -half_x)
            | (pos[:, 0] > half_x)
            | (pos[:, 1] < -half_y)
            | (pos[:, 1] > half_y)
            | (pos[:, 2] < BOUND_Z_MIN)
            | (pos[:, 2] > BOUND_Z_MAX)
        )
        out |= active[:, i] & item_out
    return out


@configclass
class GraspTwoItemsRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for reaching and lifting the two active objects."""

    fingers_to_active_objects = RewTerm(
        func=active_object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )
    active_lift_height = RewTerm(func=active_object_lift_progress, weight=4.0)
    all_active_lifted_bonus = RewTerm(func=active_objects_lifted_reward, weight=8.0)


@configclass
class GraspTwoItemsTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for randomized two-object grasp task."""

    active_objects_out_of_bound = DoneTerm(func=active_objects_out_of_bound)
    success = DoneTerm(func=active_objects_lifted)


@configclass
class GraspTwoItemsEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for randomized two-object grasp task."""

    sample_active_objects = EventTerm(func=reset_active_grasp_object_pair, mode="reset")
    reset_active_objects = EventTerm(func=reset_active_grasp_objects_random, mode="reset")


@configclass
class GraspTwoItemsSceneCfg(dexverse_base_env.SceneCfg):
    sphere: RigidObjectCfg = OBJECT_CFGS["sphere"]
    cylinder: RigidObjectCfg = OBJECT_CFGS["cylinder"]
    triangular_prism: RigidObjectCfg = OBJECT_CFGS["triangular_prism"]
    cube: RigidObjectCfg = OBJECT_CFGS["cube"]
    cuboid: RigidObjectCfg = OBJECT_CFGS["cuboid"]
    object = None


@configclass
class GraspTwoItemsEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Short-horizon task: randomly select and lift two primitive objects."""

    rewards: GraspTwoItemsRewardsCfg = GraspTwoItemsRewardsCfg()
    terminations: GraspTwoItemsTerminationsCfg = GraspTwoItemsTerminationsCfg()
    events: GraspTwoItemsEventCfg = GraspTwoItemsEventCfg()
    scene: GraspTwoItemsSceneCfg = GraspTwoItemsSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()

        self.commands.object_pose = None
        self.observations.contact = None
        self.observations.state = None
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
        self.events.reset_object = None
        self.terminations.object_out_of_bound = None

        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )
        self.rewards.fingers_to_active_objects.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.episode_length_s = 20.0


@configclass
class GraspTwoItemsEnvFloatingDexHandRightCfg(GraspTwoItemsEnvCfg):
    """Grasp-two-items environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
