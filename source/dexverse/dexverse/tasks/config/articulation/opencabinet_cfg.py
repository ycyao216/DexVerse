# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-cabinet task using storage-unit asset 45249.

The task succeeds when the door joint opens a fixed angle from the reset pose.
"""

from __future__ import annotations

import math

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

# Storage-unit cabinet 45249 with a single revolute door joint (joint_0)
# hinging the panel on a vertical edge.
ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "storage_unit"
CABINET_USD_PATH = str(ASSET_DIR / "45249" / "mobility.usd")
CABINET_SCALE = (0.35, 0.35, 0.35)
# Root-origin-to-tabletop offset measured for 45249 at 0.35 scale.
CABINET_HALF_HEIGHT_EST = 0.274
SUCCESS_JOINT_NAME = "joint_0"
# Define the task against an explicit target angle instead of the joint
# range (which the old progress/ratio modes would normalize by): success =
# swinging the door OPEN_SUCCESS_FRACTION * OPEN_ANGLE_RAD from its reset
# pose (~72 deg).
OPEN_ANGLE_RAD = math.radians(90.0)
OPEN_SUCCESS_FRACTION = 0.8
INIT_OPEN_RAD = 0.1


@configclass
class OpenCabinetRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to fraction of door opening (reset pose → target angle)."""

    open_amount = RewTerm(
        func=mdp.joint_open_fraction_reward,
        weight=5.0,
        params={
            "close_angle_rad": INIT_OPEN_RAD,
            "open_angle_rad": OPEN_ANGLE_RAD,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[SUCCESS_JOINT_NAME]),
        },
    )


@configclass
class OpenCabinetTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success once the door swings ``OPEN_SUCCESS_FRACTION`` of the target angle."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": OPEN_SUCCESS_FRACTION * OPEN_ANGLE_RAD,
            "mode": "displacement",
            "reduce": "any",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[SUCCESS_JOINT_NAME]),
        },
    )


@configclass
class OpenCabinetEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Open-cabinet env backed by storage-unit asset 45249."""

    articulation_usd_path: str = CABINET_USD_PATH
    articulation_scale: tuple[float, float, float] = CABINET_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = CABINET_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [SUCCESS_JOINT_NAME]

    rewards: OpenCabinetRewardsCfg = OpenCabinetRewardsCfg()
    terminations: OpenCabinetTerminationsCfg = OpenCabinetTerminationsCfg()

    def __post_init__(self):
        # Slight initial opening so the policy doesn't have to handle a hard
        # contact at t=0. Set BEFORE super so it's wired into the articulation.
        self.articulation_init_joint_pos = {SUCCESS_JOINT_NAME: INIT_OPEN_RAD}
        super().__post_init__()
        # Light damping/friction keeps the passive door from swinging wildly
        # while preserving direct manipulation through contact.
        self.scene.articulation.actuators = {
            "cabinet_joint_damping": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim={".*": 2.0},
                stiffness={".*": 0.0},
                damping={".*": 0.15},
                friction={".*": 0.05},
            ),
        }


# Backward-compat alias.
OpenCabinetEnvCfg = OpenCabinetEnvFloatingDexHandRightCfg
