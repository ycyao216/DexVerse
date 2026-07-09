# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift-lid task: open a kettle's lid past 80% of its hinge range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The reward is
``joint_range_progress`` (progress relative to the full joint range), and
the success condition is a ratio-mode ``joint_relative_move``.
"""

from __future__ import annotations

from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "kettle"
KETTLE_USD_PATH = str(ASSET_DIR / "101305.usd")
KETTLE_SCALE = (0.2, 0.2, 0.2)
KETTLE_HALF_HEIGHT_EST = 0.075
LID_JOINT_NAME = "joint_0"
LID_SUCCESS_RATIO = 0.8


@configclass
class LiftLidRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to lid joint progress over its range."""

    lid_lift = RewTerm(
        func=mdp.joint_range_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[LID_JOINT_NAME]),
        },
    )


@configclass
class LiftLidTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the lid joint has covered ``LID_SUCCESS_RATIO`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": LID_SUCCESS_RATIO,
            "mode": "ratio",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[LID_JOINT_NAME]),
        },
    )


@configclass
class LiftLidEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Lift-lid env (floating dex-hand variant)."""

    articulation_usd_path: str = KETTLE_USD_PATH
    articulation_scale: tuple[float, float, float] = KETTLE_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.1, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = KETTLE_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [LID_JOINT_NAME]
    # success_threshold left at None: ratio-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: LiftLidRewardsCfg = LiftLidRewardsCfg()
    terminations: LiftLidTerminationsCfg = LiftLidTerminationsCfg()


# Backward-compat alias.
LiftLidEnvCfg = LiftLidEnvFloatingDexHandRightCfg
