# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for stacking one primitive cube on another."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import PickupObjectObservationsCfg

CUBE_SIZE = 0.05
CUBE_HALF_SIZE = CUBE_SIZE * 0.5
MOVABLE_CUBE_MASS_KG = 0.08
CENTER_SQUARE_SIZE = 0.45
STACK_TARGET_STD = 0.10
STACK_XY_THRESHOLD = 0.035
STACK_Z_THRESHOLD = 0.025
STACK_LIN_VEL_MAX_MPS = 0.08
STACK_ANG_VEL_MAX_RADPS = 1.0
LIFT_SHAPING_HEIGHT_M = 0.12
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5


MOVABLE_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=MOVABLE_CUBE_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.45, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.12, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)

BASE_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BaseCube",
    spawn=sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)


def cube_stack_target_error(
    env: ManagerBasedRLEnv,
    movable_cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    base_cube_cfg: SceneEntityCfg = SceneEntityCfg("base_cube"),
    cube_size: float = CUBE_SIZE,
) -> torch.Tensor:
    """Distance from movable cube center to the stacked center pose above the base cube."""
    movable_cube = env.scene[movable_cube_cfg.name]
    base_cube = env.scene[base_cube_cfg.name]
    target_pos = base_cube.data.root_pos_w.clone()
    target_pos[:, 2] += cube_size
    return torch.norm(movable_cube.data.root_pos_w - target_pos, dim=1)


def cube_stack_target_reward(
    env: ManagerBasedRLEnv,
    std: float = STACK_TARGET_STD,
    movable_cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    base_cube_cfg: SceneEntityCfg = SceneEntityCfg("base_cube"),
    cube_size: float = CUBE_SIZE,
) -> torch.Tensor:
    """Dense reward for moving the cube toward the stacked pose."""
    distance = cube_stack_target_error(env, movable_cube_cfg, base_cube_cfg, cube_size)
    return torch.exp(-distance / max(std, 1e-6))


def cube_stacked(
    env: ManagerBasedRLEnv,
    movable_cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    base_cube_cfg: SceneEntityCfg = SceneEntityCfg("base_cube"),
    cube_size: float = CUBE_SIZE,
    xy_threshold: float = STACK_XY_THRESHOLD,
    z_threshold: float = STACK_Z_THRESHOLD,
    lin_vel_max: float = STACK_LIN_VEL_MAX_MPS,
    ang_vel_max: float = STACK_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    """Success when the movable cube is centered and settled on top of the base cube."""
    movable_cube = env.scene[movable_cube_cfg.name]
    base_cube = env.scene[base_cube_cfg.name]
    delta = movable_cube.data.root_pos_w - base_cube.data.root_pos_w
    xy_ok = torch.norm(delta[:, :2], dim=1) <= xy_threshold
    z_ok = torch.abs(delta[:, 2] - cube_size) <= z_threshold
    lin_vel_ok = torch.norm(movable_cube.data.root_lin_vel_w, dim=1) <= lin_vel_max
    ang_vel_ok = torch.norm(movable_cube.data.root_ang_vel_w, dim=1) <= ang_vel_max
    return xy_ok & z_ok & lin_vel_ok & ang_vel_ok


def cube_stacked_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward matching the stack-cube success condition."""
    return cube_stacked(env).float()


@configclass
class StackCubeObservationsCfg(PickupObjectObservationsCfg):
    """Observation layout for stack-cube.

    Inherits the pickup-object split (movable-cube pose in ``state``, movable-cube
    linear / angular velocity in ``privileged``) and adds, to ``state``, the base
    cube's pose and the movable cube's pose *relative to* the base cube. The base
    cube is kinematic, so its velocity — and the relative velocity, which equals
    the movable cube's — is redundant; only the relative *pose* is exposed.
    """

    @configclass
    class StateObsCfg(PickupObjectObservationsCfg.StateObsCfg):
        base_cube_pos_b = ObsTerm(
            func=mdp.object_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("base_cube")},
        )
        base_cube_quat_b = ObsTerm(
            func=mdp.object_quat_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"object_cfg": SceneEntityCfg("base_cube")},
        )
        object_to_base_cube = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("base_cube")},
        )

    state: StateObsCfg = StateObsCfg()


@configclass
class StackCubeRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for stack-cube."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )
    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )
    lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": LIFT_SHAPING_HEIGHT_M},
    )
    stack_target = RewTerm(func=cube_stack_target_reward, weight=3.0)
    stacked_bonus = RewTerm(func=cube_stacked_reward, weight=8.0)


@configclass
class StackCubeTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for stack-cube."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    success = DoneTerm(func=cube_stacked)


@configclass
class StackCubeEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for stack-cube."""

    object_scale_mass = None
    reset_base_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("base_cube"),
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class StackCubeSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = MOVABLE_CUBE_CFG
    base_cube: RigidObjectCfg = BASE_CUBE_CFG


@configclass
class StackCubeEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Stack-cube task configuration (base, robot-agnostic)."""

    observations: StackCubeObservationsCfg = StackCubeObservationsCfg()
    rewards: StackCubeRewardsCfg = StackCubeRewardsCfg()
    terminations: StackCubeTerminationsCfg = StackCubeTerminationsCfg()
    events: StackCubeEventCfg = StackCubeEventCfg()
    scene: StackCubeSceneCfg = StackCubeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=MOVABLE_CUBE_CFG,
        base_cube=BASE_CUBE_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.events.object_physics_material = None

        table_size = self.scene.table.spawn.size
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        object_pos = self.scene.object.init_state.pos
        base_pos = self.scene.base_cube.init_state.pos
        self.scene.object.init_state.pos = (object_pos[0], object_pos[1], table_top_z + CUBE_HALF_SIZE)
        self.scene.base_cube.init_state.pos = (base_pos[0], base_pos[1], table_top_z + CUBE_HALF_SIZE)
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.base_cube.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        if self.events.reset_object is not None:
            half_side = CENTER_SQUARE_SIZE * 0.5
            self.events.reset_object.params["pose_range"] = {
                "x": [-0.18, -0.08],
                "y": [-half_side * 0.5, half_side * 0.5],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-3.14, 3.14],
            }

        if self.terminations.object_out_of_bound is not None:
            half_x = table_size[0] * 0.5
            half_y = table_size[1] * 0.5
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-half_x, half_x),
                "y": (-half_y, half_y),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class StackCubeEnvFloatingDexHandRightCfg(StackCubeEnvCfg):
    """Stack-cube config for floating dexterous hands (Shadow / Leap)."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
