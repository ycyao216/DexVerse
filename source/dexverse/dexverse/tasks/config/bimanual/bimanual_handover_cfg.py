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
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .bimanual_contact_links import (
    contact_sensor_names, resolve_bimanual_collision_links, resolve_bimanual_contact_links,
)
from .handover import (
    HandoverState, HandoverWristAction, constrain_handover_wrists, reset_handover_state,
    update_handover_visuals,
)

OBJECT_MASS_KG = 0.35
CUBE_SIZE = 0.06
ROBOT_INIT_ROT = (0.0, 1.0, 0.0, 0.0)
ROBOT_INIT_POS = (-0.75, 0.0, 1.0)
OBJECT_INIT_POS = (-0.05, -0.30, 0.70)
OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

# Contact-transfer success thresholds.
SUCCESS_CONTACT_ACTIVE_FORCE_N = 0.2
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


HANDOVER_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        activate_contact_sensors=True,
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
    filter_indices: tuple[int, ...],
    contact_active_force_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not filter_indices:
        zeros = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        return zeros, zeros.to(dtype=torch.long)

    contact = env.scene.sensors["handover_object_contact"].data
    forces = contact.force_matrix_w.reshape(env.num_envs, -1, 3)[:, filter_indices]
    # Sum magnitudes per colliding body: opposing finger forces must not cancel.
    force_mag = forces.norm(dim=-1)
    total_force = force_mag.sum(dim=1)
    active_links = (force_mag >= contact_active_force_n).sum(dim=1)
    return total_force, active_links


def _object_motion_stable(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_lin_vel_max_mps: float = SUCCESS_OBJECT_LIN_VEL_MAX_MPS,
    object_ang_vel_max_radps: float = SUCCESS_OBJECT_ANG_VEL_MAX_RADPS,
) -> torch.Tensor:
    """Require a settled hold relative to the receiver, allowing the hand to move."""
    obj = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    palm_id = robot.body_names.index(env.cfg.robot_config.left_palm_body_name)
    palm_pos = robot.data.body_com_pos_w[:, palm_id]
    palm_omega = robot.data.body_com_ang_vel_w[:, palm_id]
    palm_velocity_at_object = robot.data.body_com_lin_vel_w[:, palm_id] + torch.cross(
        palm_omega, obj.data.root_pos_w - palm_pos, dim=-1,
    )
    lin_vel_ok = (obj.data.root_lin_vel_w - palm_velocity_at_object).norm(dim=1) <= object_lin_vel_max_mps
    ang_vel_ok = (obj.data.root_ang_vel_w - palm_omega).norm(dim=1) <= object_ang_vel_max_radps
    return lin_vel_ok & ang_vel_ok


def handover_success(
    env: ManagerBasedRLEnv,
    left_contact_filter_indices: tuple[int, ...] = (),
    right_contact_filter_indices: tuple[int, ...] = (),
    min_lift: float = SUCCESS_MIN_LIFT_M,
    receiver_hold_s: float = 0.3,
    contact_gap_s: float = 0.1,
) -> torch.Tensor:
    """Right pickup followed by a left-only hold, without a drop or table support."""
    if not left_contact_filter_indices or not right_contact_filter_indices:
        return _false_flags(env)
    state = getattr(env, "_handover_state", None)
    if state is None:
        state = HandoverState(env.num_envs, env.device, env.step_dt, receiver_hold_s, contact_gap_s)
        env._handover_state = state

    left_force, left_links = _aggregate_contact(env, left_contact_filter_indices, SUCCESS_CONTACT_ACTIVE_FORCE_N)
    right_force, right_links = _aggregate_contact(env, right_contact_filter_indices, SUCCESS_CONTACT_ACTIVE_FORCE_N)
    source_only = (
        (right_force >= SUCCESS_RIGHT_FORCE_MIN_N)
        & (right_links >= SUCCESS_RIGHT_ACTIVE_LINKS_MIN)
        & (left_force <= SUCCESS_LEFT_FORCE_MAX_N)
        & (left_links <= SUCCESS_LEFT_ACTIVE_LINKS_MAX)
    )
    receiver_only = (
        (left_force >= SUCCESS_LEFT_FORCE_MIN_N)
        & (left_links >= SUCCESS_LEFT_ACTIVE_LINKS_MIN)
        & (right_force <= SUCCESS_RIGHT_FORCE_MAX_N)
        & (right_links <= SUCCESS_RIGHT_ACTIVE_LINKS_MAX)
    )
    obj = env.scene["object"]
    table = env.cfg.scene.table
    table_top = table.init_state.pos[2] + table.spawn.size[2] / 2
    object_pos = obj.data.root_pos_w - env.scene.env_origins
    # Exact vertical extent of the rotated cube, rather than its center height.
    up = torch.zeros_like(object_pos)
    up[:, 2] = 1.0
    half_extent_z = quat_apply_inverse(obj.data.root_quat_w, up).abs().sum(dim=-1) * (CUBE_SIZE / 2)
    clearance = object_pos[:, 2] - half_extent_z - table_top
    contact = env.scene.sensors["handover_object_contact"].data
    table_force = contact.force_matrix_w[:, 0, 0].norm(dim=-1)
    success = state.update(
        env.episode_length_buf,
        source_only=source_only,
        receiver_only=receiver_only,
        lifted=clearance >= min_lift,
        clear_of_table=(clearance > 0.002) & (table_force < 0.1),
        supported=(left_force + right_force) >= SUCCESS_CONTACT_ACTIVE_FORCE_N,
        stable=_object_motion_stable(env),
    )
    update_handover_visuals(env)
    return success


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

    success = DoneTerm(func=handover_success)


@configclass
class BimanualHandoverSceneCfg(dexverse_base_env.SceneCfg):
    object: RigidObjectCfg = HANDOVER_OBJECT_CFG
    handover_object_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        max_contact_data_count_per_prim=64,
    )


@configclass
class BimanualHandoverEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Bimanual object handover on a tabletop scene (robot-agnostic base)."""

    # Wrists remain in their own table half; fingers can meet at y=0.
    handover_receiver_hold_s: float = 0.3
    handover_contact_gap_s: float = 0.1

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

        # Install physics stops and clamp the processed joint targets, covering
        # both absolute/relative VR retargeting and policy actions.
        for name, action in vars(self.actions).items():
            if "wrist" in name and isinstance(action, JointPositionActionCfg):
                action.class_type = HandoverWristAction
        self.events.handover_workspace = EventTerm(func=constrain_handover_wrists, mode="startup")
        self.events.reset_handover_workspace = EventTerm(func=constrain_handover_wrists, mode="reset")
        self.events.reset_handover = EventTerm(func=reset_handover_state, mode="reset")

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

        left_filter_indices = ()
        right_filter_indices = ()
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
            collision_links = resolve_bimanual_collision_links(self.scene.robot)
            # One object sensor, with a separate filter for each colliding body.
            # Keep table first so support on the table is unambiguously rejected.
            self.scene.handover_object_contact.filter_prim_paths_expr = [
                "{ENV_REGEX_NS}/table",
                *(f"{tip_prim_prefix}{link}" for link in collision_links.all),
            ]
            right_filter_indices = tuple(range(1, 1 + len(collision_links.right)))
            left_filter_indices = tuple(range(1 + len(collision_links.right), 1 + len(collision_links.all)))
            self.observations.contact.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": all_sensor_names},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None

        self.terminations.success.params = {
            "left_contact_filter_indices": left_filter_indices,
            "right_contact_filter_indices": right_filter_indices,
            "receiver_hold_s": self.handover_receiver_hold_s,
            "contact_gap_s": self.handover_contact_gap_s,
        }

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
