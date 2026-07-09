# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual handover task configuration for tabletop manipulation."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .bimanual_contact_links import contact_sensor_names, resolve_bimanual_contact_links

OBJECT_MASS_KG = 0.35
CUBE_SIZE = 0.06
ROBOT_INIT_ROT = (0.0, 1.0, 0.0, 0.0)
ROBOT_INIT_POS = (-0.75, 0.0, 1.0)
OBJECT_INIT_POS = (-0.05, -0.30, 0.70)
OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

# Contact-transfer success thresholds.
SUCCESS_CONTACT_ACTIVE_FORCE_N = 1.0
SUCCESS_LEFT_FORCE_MIN_N = 1.6
SUCCESS_RIGHT_FORCE_MIN_N = SUCCESS_LEFT_FORCE_MIN_N
SUCCESS_LEFT_FORCE_MAX_N = 1.2
SUCCESS_RIGHT_FORCE_MAX_N = SUCCESS_LEFT_FORCE_MAX_N
SUCCESS_LEFT_ACTIVE_LINKS_MIN = 1
SUCCESS_RIGHT_ACTIVE_LINKS_MIN = SUCCESS_LEFT_ACTIVE_LINKS_MIN
SUCCESS_LEFT_ACTIVE_LINKS_MAX = 0
SUCCESS_RIGHT_ACTIVE_LINKS_MAX = SUCCESS_LEFT_ACTIVE_LINKS_MAX
SUCCESS_OBJECT_LIN_VEL_MAX_MPS = 0.05
SUCCESS_OBJECT_ANG_VEL_MAX_RADPS = 0.5

SUCCESS_MIN_LIFT_M = 0.03

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5

BIMANUAL_HANDOVER_STAGE_GRAPH_KEY_PREFIX = "bimanual_handover"
BIMANUAL_HANDOVER_RIGHT_STAGE = "right_grasp_stable"
BIMANUAL_HANDOVER_LEFT_STAGE = "left_handover_stable"


HANDOVER_OBJECT_CFG = RigidObjectCfg(
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
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
        semantic_tags=[("class", "primitive_cube")],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=OBJECT_INIT_POS,
        rot=OBJECT_INIT_ROT,
    ),
)


def _false_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _aggregate_contact(
    env: ManagerBasedRLEnv,
    sensor_names: list[str] | tuple[str, ...],
    contact_active_force_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sensor_names:
        zeros = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        return zeros, zeros.to(dtype=torch.long)

    forces = []
    for sensor_name in sensor_names:
        sensor = env.scene.sensors[sensor_name]
        force = sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3).norm(dim=-1).sum(dim=1)
        forces.append(force)
    force_mag = torch.stack(forces, dim=1)
    total_force = force_mag.sum(dim=1)
    active_links = (force_mag >= contact_active_force_n).sum(dim=1)
    return total_force, active_links


def _object_motion_stable(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_lin_vel_max_mps: float = SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
    object_ang_vel_max_radps: float = SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    lin_vel_ok = obj.data.root_lin_vel_w.norm(dim=1) <= object_lin_vel_max_mps
    ang_vel_ok = obj.data.root_ang_vel_w.norm(dim=1) <= object_ang_vel_max_radps
    return lin_vel_ok & ang_vel_ok


def _single_hand_hold_stable(
    env: ManagerBasedRLEnv,
    hold_contact_sensor_names: list[str] | tuple[str, ...] | None,
    release_contact_sensor_names: list[str] | tuple[str, ...] | None,
    hold_force_min_n: float,
    release_force_max_n: float,
    hold_active_links_min: int,
    release_active_links_max: int,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_active_force_n: float = SUCCESS_CONTACT_ACTIVE_FORCE_N,
    object_lin_vel_max_mps: float = SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
    object_ang_vel_max_radps: float = SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    hold_contact_sensor_names = hold_contact_sensor_names or []
    release_contact_sensor_names = release_contact_sensor_names or []
    if not hold_contact_sensor_names or not release_contact_sensor_names:
        return _false_flags(env)

    hold_force_sum, hold_active_links = _aggregate_contact(env, hold_contact_sensor_names, contact_active_force_n)
    release_force_sum, release_active_links = _aggregate_contact(
        env, release_contact_sensor_names, contact_active_force_n
    )

    hold_ok = (hold_force_sum >= hold_force_min_n) & (hold_active_links >= hold_active_links_min)
    release_ok = (release_force_sum <= release_force_max_n) & (release_active_links <= release_active_links_max)
    stable_ok = _object_motion_stable(
        env,
        object_cfg=object_cfg,
        object_lin_vel_max_mps=object_lin_vel_max_mps,
        object_ang_vel_max_radps=object_ang_vel_max_radps,
    )
    return hold_ok & release_ok & stable_ok


def right_grasp_stable(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_contact_sensor_names: list[str] | tuple[str, ...] | None = None,
    right_contact_sensor_names: list[str] | tuple[str, ...] | None = None,
    contact_active_force_n: float = SUCCESS_CONTACT_ACTIVE_FORCE_N,
    right_force_min_n: float = SUCCESS_RIGHT_FORCE_MIN_N,
    left_force_max_n: float = SUCCESS_LEFT_FORCE_MAX_N,
    right_active_links_min: int = SUCCESS_RIGHT_ACTIVE_LINKS_MIN,
    left_active_links_max: int = SUCCESS_LEFT_ACTIVE_LINKS_MAX,
    object_lin_vel_max_mps: float = SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
    object_ang_vel_max_radps: float = SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    """Stage 1: the right hand holds the object while the left hand is released."""
    return _single_hand_hold_stable(
        env,
        hold_contact_sensor_names=right_contact_sensor_names,
        release_contact_sensor_names=left_contact_sensor_names,
        hold_force_min_n=right_force_min_n,
        release_force_max_n=left_force_max_n,
        hold_active_links_min=right_active_links_min,
        release_active_links_max=left_active_links_max,
        object_cfg=object_cfg,
        contact_active_force_n=contact_active_force_n,
        object_lin_vel_max_mps=object_lin_vel_max_mps,
        object_ang_vel_max_radps=object_ang_vel_max_radps,
    )


def left_handover_stable(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_contact_sensor_names: list[str] | tuple[str, ...] | None = None,
    right_contact_sensor_names: list[str] | tuple[str, ...] | None = None,
    contact_active_force_n: float = SUCCESS_CONTACT_ACTIVE_FORCE_N,
    left_force_min_n: float = SUCCESS_LEFT_FORCE_MIN_N,
    right_force_max_n: float = SUCCESS_RIGHT_FORCE_MAX_N,
    left_active_links_min: int = SUCCESS_LEFT_ACTIVE_LINKS_MIN,
    right_active_links_max: int = SUCCESS_RIGHT_ACTIVE_LINKS_MAX,
    object_lin_vel_max_mps: float = SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
    object_ang_vel_max_radps: float = SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    """Stage 2: the left hand holds the object while the right hand is released."""
    return _single_hand_hold_stable(
        env,
        hold_contact_sensor_names=left_contact_sensor_names,
        release_contact_sensor_names=right_contact_sensor_names,
        hold_force_min_n=left_force_min_n,
        release_force_max_n=right_force_max_n,
        hold_active_links_min=left_active_links_min,
        release_active_links_max=right_active_links_max,
        object_cfg=object_cfg,
        contact_active_force_n=contact_active_force_n,
        object_lin_vel_max_mps=object_lin_vel_max_mps,
        object_ang_vel_max_radps=object_ang_vel_max_radps,
    )


def make_handover_stage_graph(
    left_contact_sensor_names: list[str] | tuple[str, ...],
    right_contact_sensor_names: list[str] | tuple[str, ...],
) -> mdp.StageGraphSpec:
    shared_params = {
        "object_cfg": SceneEntityCfg("object"),
        "left_contact_sensor_names": tuple(left_contact_sensor_names),
        "right_contact_sensor_names": tuple(right_contact_sensor_names),
        "contact_active_force_n": SUCCESS_CONTACT_ACTIVE_FORCE_N,
        "object_lin_vel_max_mps": SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
        "object_ang_vel_max_radps": SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
    }
    return mdp.StageGraphSpec(
        stages=(
            mdp.StageSpec(
                name=BIMANUAL_HANDOVER_RIGHT_STAGE,
                func=right_grasp_stable,
                params={
                    **shared_params,
                    "right_force_min_n": SUCCESS_RIGHT_FORCE_MIN_N,
                    "left_force_max_n": SUCCESS_LEFT_FORCE_MAX_N,
                    "right_active_links_min": SUCCESS_RIGHT_ACTIVE_LINKS_MIN,
                    "left_active_links_max": SUCCESS_LEFT_ACTIVE_LINKS_MAX,
                },
            ),
            mdp.StageSpec(
                name=BIMANUAL_HANDOVER_LEFT_STAGE,
                func=left_handover_stable,
                params={
                    **shared_params,
                    "left_force_min_n": SUCCESS_LEFT_FORCE_MIN_N,
                    "right_force_max_n": SUCCESS_RIGHT_FORCE_MAX_N,
                    "left_active_links_min": SUCCESS_LEFT_ACTIVE_LINKS_MIN,
                    "right_active_links_max": SUCCESS_RIGHT_ACTIVE_LINKS_MAX,
                },
                deps=(BIMANUAL_HANDOVER_RIGHT_STAGE,),
            ),
        ),
        terminal_stage=BIMANUAL_HANDOVER_LEFT_STAGE,
        ordering_mode="strict",
        success_mode="substage",
    )


mdp.register_stage_graph(
    BIMANUAL_HANDOVER_STAGE_GRAPH_KEY_PREFIX,
    make_handover_stage_graph((), ()),
    override=True,
)


@configclass
class BimanualHandoverObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for bimanual handover.

    Object position / up-axis / tilt angle live in ``proprio`` (a real setup
    can recover all three from a pose estimate). Object linear / angular
    velocity live in ``privileged``.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_tilt_angle = ObsTerm(func=mdp.object_tilt_angle, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class BimanualHandoverRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for bimanual handover."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=2.0,
    )

    lift_height = RewTerm(
        func=mdp.object_lift_height,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": SUCCESS_MIN_LIFT_M,
        },
        weight=0.5,
    )


@configclass
class BimanualHandoverTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for bimanual handover."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.stage_success,
        params={
            "task_key": BIMANUAL_HANDOVER_STAGE_GRAPH_KEY_PREFIX,
            "terminal_stage": BIMANUAL_HANDOVER_LEFT_STAGE,
            "persistent": True,
            "success_mode": "substage",
            "ordering_mode": "strict",
        },
    )


@configclass
class BimanualHandoverSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = HANDOVER_OBJECT_CFG


@configclass
class BimanualHandoverEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Bimanual object handover on a tabletop scene (robot-agnostic base)."""

    observations: BimanualHandoverObservationsCfg = BimanualHandoverObservationsCfg()
    rewards: BimanualHandoverRewardsCfg = BimanualHandoverRewardsCfg()
    terminations: BimanualHandoverTerminationsCfg = BimanualHandoverTerminationsCfg()
    scene: BimanualHandoverSceneCfg = BimanualHandoverSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=HANDOVER_OBJECT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.commands.object_pose = None
        self.scene.object.init_state.pos = OBJECT_INIT_POS
        self.scene.object.init_state.rot = OBJECT_INIT_ROT

        # Keep reset deterministic: robot and object do not move at reset.
        if self.events.reset_robot_joints is not None:
            self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
            self.events.reset_robot_joints.params["velocity_range"] = [0.0, 0.0]
        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }
            self.events.reset_object.params["velocity_range"] = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            }

        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-table_size[0] * 0.5, table_size[0] * 0.5),
                "y": (-table_size[1] * 0.5, table_size[1] * 0.5),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        left_sensor_names = []
        right_sensor_names = []
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            contact_links = resolve_bimanual_contact_links(
                robot_type=self.robot_type,
                robot_config=self.robot_config,
                robot_cfg=self.scene.robot,
            )
            for link_name in contact_links.all:
                setattr(
                    self.scene,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path=f"{tip_prim_prefix}{link_name}",
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )
            all_sensor_names = contact_sensor_names(contact_links.all)
            left_sensor_names = contact_sensor_names(contact_links.left)
            right_sensor_names = contact_sensor_names(contact_links.right)
            self.observations.contact.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": all_sensor_names},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None

        stage_graph_key = f"{BIMANUAL_HANDOVER_STAGE_GRAPH_KEY_PREFIX}:{self.robot_type}"
        mdp.register_stage_graph(
            stage_graph_key,
            make_handover_stage_graph(left_sensor_names, right_sensor_names),
            override=True,
        )
        self.terminations.success.params["task_key"] = stage_graph_key

        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class BimanualHandoverEnvFloatingShadowBimanualCfg(BimanualHandoverEnvCfg):
    """Bimanual handover task with floating bimanual Shadow hands."""

    robot_type: str = "floating_shadow_bimanual"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
