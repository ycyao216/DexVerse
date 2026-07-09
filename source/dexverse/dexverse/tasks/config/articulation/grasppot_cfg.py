# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-pot task: open a kitchen pot's lid past 80% of its hinge range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. Reward is a
fixed-displacement ``joint_displacement_reward`` (any lift past 5 cm
counts), and success is a ratio-mode ``joint_relative_move``.
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

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "kitchenpot"
POT_USD_PATH = str(ASSET_DIR / "100015.usd")
POT_SCALE = (0.2, 0.2, 0.2)
POT_HALF_HEIGHT_EST = 0.07
LID_JOINT_NAME = "joint_0"
LID_OPEN_DISPLACEMENT_M = 0.05
LID_SUCCESS_RATIO = 0.8


@configclass
class GraspPotRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward when the lid joint displacement exceeds ``LID_OPEN_DISPLACEMENT_M``."""

    lid_open = RewTerm(
        func=mdp.joint_displacement_reward,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[LID_JOINT_NAME]),
            "threshold_m": LID_OPEN_DISPLACEMENT_M,
        },
    )


@configclass
class GraspPotTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the lid has rotated through ``LID_SUCCESS_RATIO`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[LID_JOINT_NAME]),
            "threshold": LID_SUCCESS_RATIO,
            "mode": "ratio",
        },
    )


@configclass
class GraspPotEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Grasp-pot env (floating dex-hand variant)."""

    articulation_usd_path: str = POT_USD_PATH
    articulation_scale: tuple[float, float, float] = POT_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.2, 0.15, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = POT_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [LID_JOINT_NAME]
    # success_threshold left at None: ratio-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: GraspPotRewardsCfg = GraspPotRewardsCfg()
    terminations: GraspPotTerminationsCfg = GraspPotTerminationsCfg()


# Backward-compat alias.
GraspPotEnvCfg = GraspPotEnvFloatingDexHandRightCfg
