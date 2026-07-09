# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-door task: hinge a door past a fraction of its open range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`; only the
reward (``joint_open_fraction_reward``) and termination
(``joint_relative_move`` with ``mode="ratio"``) are task-specific.
"""

from __future__ import annotations

from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "door"
DOOR_USD_PATH = str(ASSET_DIR / "door.usd")
DOOR_SCALE = (1.5, 1.5, 1.5)
DOOR_ROT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
OPEN_ANGLE_RAD = 1.4
CLOSE_ANGLE_RAD = 0.0
INIT_ANGLE_RAD = CLOSE_ANGLE_RAD
DOOR_JOINT_NAME = "door_hinge"
HANDLE_JOINT_NAME = "handle_hinge"

SUCCESS_OPEN_RATIO = 0.8


@configclass
class OpenDoorRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to fraction of door opening (close → open)."""

    open_angle = RewTerm(
        func=mdp.joint_open_fraction_reward,
        weight=5.0,
        params={
            "close_angle_rad": CLOSE_ANGLE_RAD,
            "open_angle_rad": OPEN_ANGLE_RAD,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[DOOR_JOINT_NAME]),
        },
    )


@configclass
class OpenDoorTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when door rotated past ``SUCCESS_OPEN_RATIO`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": SUCCESS_OPEN_RATIO,
            "mode": "ratio",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[DOOR_JOINT_NAME]),
        },
    )


@configclass
class OpenDoorEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Open-door env (floating dex-hand variant)."""

    articulation_usd_path: str = DOOR_USD_PATH
    articulation_scale: tuple[float, float, float] = DOOR_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = DOOR_ROT
    articulation_half_height_est: float = 0.0
    articulation_fix_root_link: bool | None = True
    articulation_hinge_stiffness: float = 0.0
    articulation_hinge_damping: float = 1.0
    articulation_handle_damping: float = 0.2
    articulation_handle_friction: float = 0.05

    success_joint_names: list[str] = [DOOR_JOINT_NAME]
    # success_threshold left at None (the base default): opendoor's
    # termination is a ratio-mode ``joint_relative_move`` and its reward is
    # ``joint_open_fraction_reward``, neither of which use the base's
    # absolute-radian threshold.

    rewards: OpenDoorRewardsCfg = OpenDoorRewardsCfg()
    terminations: OpenDoorTerminationsCfg = OpenDoorTerminationsCfg()

    def __post_init__(self):
        # Start fully closed and latched. Set BEFORE super so it's wired
        # into the articulation.
        self.articulation_init_joint_pos = {
            DOOR_JOINT_NAME: INIT_ANGLE_RAD,
            HANDLE_JOINT_NAME: 0.0,
        }
        self.articulation_reset_pose_range = {
            "x": [0.0, 0.0],
            "y": [-0.1, 0.1],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
        super().__post_init__()
        self.scene.articulation.actuators = {
            "door_hinge": ImplicitActuatorCfg(
                joint_names_expr=[DOOR_JOINT_NAME],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=self.articulation_hinge_stiffness,
                damping=self.articulation_hinge_damping,
            ),
            "handle_hinge": ImplicitActuatorCfg(
                joint_names_expr=[HANDLE_JOINT_NAME],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_handle_damping,
                friction=self.articulation_handle_friction,
            ),
        }


# Backward-compat alias. ``multigraspenvs.opendoor_rigid_variants_cfg`` (to be
# refactored next) and the gym registration imported the old class name.
OpenDoorEnvCfg = OpenDoorEnvFloatingDexHandRightCfg
